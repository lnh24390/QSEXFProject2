"""The labelling canvas: zoom / pan / prompt / polygon editing."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (QBrush, QColor, QImage, QPainter, QPen, QPixmap,
                           QTransform)
from PySide6.QtWidgets import (QGraphicsPixmapItem, QGraphicsRectItem,
                               QGraphicsScene, QGraphicsView, QRubberBand)

from ..core.annotation import Shape
from .items import CrosshairItem, MaskPreviewItem, PromptPointItem, ShapeItem

# tools
SELECT = "select"
SAM_POINT = "sam_point"
SAM_BOX = "sam_box"
POLYGON = "polygon"
BOX2SEG = "box2seg"
RECT = "rect"

TOOL_HINTS = {
    SELECT: "선택/편집 — 클릭 선택, 더블클릭 정점편집, Alt+클릭 정점추가, Del 삭제",
    SAM_POINT: "SAM 클릭 — 좌클릭 포함(+), 우클릭 제외(−), Enter 확정, Esc 취소",
    SAM_BOX: "SAM 박스 — 드래그로 객체를 감싸면 마스크 생성, Enter 확정",
    POLYGON: "수동 폴리곤 — 좌클릭 점 추가, 우클릭/Enter 닫기, Backspace 취소",
    BOX2SEG: "박스 클릭 변환 — 점선 박스를 하나씩 클릭하면 SAM이 마스크로 바꿉니다"
             " (여러 개를 한 번에 바꾸려면 선택 후 Ctrl+Shift+G)",
    RECT: "박스 라벨 — 드래그로 박스를 그립니다. 여러 개를 계속 그릴 수 있습니다",
}


class CanvasView(QGraphicsView):
    promptChanged = Signal(list, object)       # [(x,y,label)], box|None
    polygonFinished = Signal(list)             # [(x,y), ...]
    boxFinished = Signal(list)                 # [x1, y1, x2, y2], a box label
    convertRequested = Signal(object)          # ShapeItem
    selectionChangedSig = Signal(object)       # ShapeItem | None
    cursorMoved = Signal(float, float)
    zoomChanged = Signal(float)
    shapeEdited = Signal(object)               # ShapeItem
    editAboutToStart = Signal(object)          # ShapeItem, before the mutation

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.setResizeAnchor(QGraphicsView.NoAnchor)
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setMouseTracking(True)
        self.setBackgroundBrush(QBrush(QColor("#141517")))
        self.setFocusPolicy(Qt.StrongFocus)

        self.base = QGraphicsPixmapItem()
        self.base.setZValue(0)
        self._scene.addItem(self.base)
        self.preview = MaskPreviewItem()
        self._scene.addItem(self.preview)
        self.crosshair = CrosshairItem()
        self._scene.addItem(self.crosshair)

        self.shape_items: List[ShapeItem] = []
        self.prompt_points: List[Tuple[float, float, int]] = []
        self._prompt_items: List[PromptPointItem] = []
        self.prompt_box: Optional[List[float]] = None
        self._box_item: Optional[QGraphicsRectItem] = None
        self._poly_points: List[Tuple[float, float]] = []
        self._poly_item: Optional[QGraphicsRectItem] = None

        self.tool = SELECT
        self._panning = False
        self._pan_start = QPoint()
        self._space = False
        self._dragging_box = False
        self._box_origin = QPointF()
        self._moving_item: Optional[ShapeItem] = None
        self._move_last = QPointF()
        self.image_size = (0, 0)
        self.fill_alpha = 70
        self.outline_width = 1.8
        self._scene.selectionChanged.connect(self._on_selection)

    # ---- image ------------------------------------------------------------
    def set_image(self, rgb: Optional[np.ndarray]) -> None:
        self.clear_prompt()
        self.clear_shapes()
        if rgb is None:
            self.base.setPixmap(QPixmap())
            self.image_size = (0, 0)
            self._scene.setSceneRect(QRectF(0, 0, 1, 1))
            return
        h, w = rgb.shape[:2]
        # QImage does not own the buffer -> copy() before the ndarray moves on
        img = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy()
        self.base.setPixmap(QPixmap.fromImage(img))
        self.image_size = (w, h)
        self._scene.setSceneRect(QRectF(0, 0, w, h))
        self.fit_to_view()

    def fit_to_view(self) -> None:
        if self.image_size[0] == 0:
            return
        self.resetTransform()
        self.fitInView(QRectF(0, 0, *self.image_size), Qt.KeepAspectRatio)
        self.zoomChanged.emit(self.zoom)

    def zoom_to(self, factor: float) -> None:
        self.resetTransform()
        self.scale(factor, factor)
        self.zoomChanged.emit(self.zoom)

    @property
    def zoom(self) -> float:
        return float(self.transform().m11())

    # ---- shapes -----------------------------------------------------------
    def add_shape_item(self, shape: Shape, color: str) -> ShapeItem:
        item = ShapeItem(shape, color, self.fill_alpha, self.outline_width)
        item.edit_began = self.editAboutToStart.emit
        self._scene.addItem(item)
        self.shape_items.append(item)
        return item

    def remove_shape_item(self, item: ShapeItem) -> None:
        item.clear_handles()
        if item.scene():
            self._scene.removeItem(item)
        if item in self.shape_items:
            self.shape_items.remove(item)

    def clear_shapes(self) -> None:
        for it in list(self.shape_items):
            self.remove_shape_item(it)
        self.shape_items = []

    def selected_items(self) -> List[ShapeItem]:
        return [i for i in self.shape_items if i.isSelected()]

    def item_for_shape(self, shape: Shape) -> Optional[ShapeItem]:
        for i in self.shape_items:
            if i.shape_data is shape or i.shape_data.uid == shape.uid:
                return i
        return None

    def set_outline_width(self, w: float) -> None:
        self.outline_width = float(w)
        for i in self.shape_items:
            i.outline_width = self.outline_width
            i.apply_style()

    def set_fill_alpha(self, a: int) -> None:
        self.fill_alpha = int(a)
        for i in self.shape_items:
            i.fill_alpha = self.fill_alpha
            i.apply_style()

    def set_shapes_visible(self, visible: bool) -> None:
        for i in self.shape_items:
            i.setVisible(visible)

    # ---- prompts ----------------------------------------------------------
    def add_prompt_point(self, x: float, y: float, positive: bool) -> None:
        self.prompt_points.append((x, y, 1 if positive else 0))
        it = PromptPointItem(x, y, positive)
        self._scene.addItem(it)
        self._prompt_items.append(it)
        self.promptChanged.emit(list(self.prompt_points), self.prompt_box)

    def undo_prompt_point(self) -> None:
        if not self.prompt_points:
            return
        self.prompt_points.pop()
        it = self._prompt_items.pop()
        self._scene.removeItem(it)
        self.promptChanged.emit(list(self.prompt_points), self.prompt_box)

    def set_prompt_box(self, box: Optional[List[float]]) -> None:
        self.prompt_box = box
        if self._box_item is not None:
            self._scene.removeItem(self._box_item)
            self._box_item = None
        if box:
            r = QGraphicsRectItem(QRectF(QPointF(box[0], box[1]),
                                         QPointF(box[2], box[3])))
            pen = QPen(QColor("#4c8dff"), 1.5, Qt.DashLine)
            pen.setCosmetic(True)
            r.setPen(pen)
            r.setZValue(38)
            self._scene.addItem(r)
            self._box_item = r
        self.promptChanged.emit(list(self.prompt_points), self.prompt_box)

    def clear_prompt(self) -> None:
        for it in self._prompt_items:
            self._scene.removeItem(it)
        self._prompt_items = []
        self.prompt_points = []
        if self._box_item is not None:
            self._scene.removeItem(self._box_item)
            self._box_item = None
        self.prompt_box = None
        self.preview.clear()
        self._clear_poly()

    def has_prompt(self) -> bool:
        return bool(self.prompt_points or self.prompt_box)

    def show_mask(self, mask: Optional[np.ndarray], color: str = "#4c8dff") -> None:
        if mask is None:
            self.preview.clear()
        else:
            self.preview.set_mask(mask, color)

    # ---- manual polygon ---------------------------------------------------
    def _clear_poly(self) -> None:
        self._poly_points = []
        if self._poly_item is not None:
            self._scene.removeItem(self._poly_item)
            self._poly_item = None

    def _redraw_poly(self) -> None:
        from PySide6.QtGui import QPainterPath, QPolygonF
        from PySide6.QtWidgets import QGraphicsPathItem
        if self._poly_item is None:
            self._poly_item = QGraphicsPathItem()
            pen = QPen(QColor("#ffb300"), 1.8)
            pen.setCosmetic(True)
            pen.setJoinStyle(Qt.RoundJoin)
            pen.setCapStyle(Qt.RoundCap)
            self._poly_item.setPen(pen)
            self._poly_item.setBrush(QBrush(QColor(255, 179, 0, 50)))
            self._poly_item.setZValue(25)
            self._scene.addItem(self._poly_item)
        path = QPainterPath()
        if self._poly_points:
            path.addPolygon(QPolygonF([QPointF(x, y) for x, y in self._poly_points]))
        self._poly_item.setPath(path)

    def finish_polygon(self) -> None:
        if len(self._poly_points) >= 3:
            self.polygonFinished.emit(list(self._poly_points))
        self._clear_poly()

    # ---- tools ------------------------------------------------------------
    def set_tool(self, tool: str) -> None:
        self.tool = tool
        self.clear_prompt()
        editable = tool == SELECT
        for i in self.shape_items:
            i.setFlag(ShapeItem.GraphicsItemFlag.ItemIsSelectable, True)
            if not editable:
                i.set_editing(False)
        self.crosshair.setVisible(tool in (SAM_POINT, SAM_BOX, POLYGON, RECT))
        cursors = {SELECT: Qt.ArrowCursor, SAM_POINT: Qt.CrossCursor,
                   SAM_BOX: Qt.CrossCursor, POLYGON: Qt.CrossCursor,
                   RECT: Qt.CrossCursor,
                   BOX2SEG: Qt.PointingHandCursor}
        self.setCursor(cursors.get(tool, Qt.ArrowCursor))

    # ---- events -----------------------------------------------------------
    def _scene_pos(self, event) -> QPointF:
        return self.mapToScene(event.position().toPoint())

    def _in_image(self, p: QPointF) -> bool:
        w, h = self.image_size
        return 0 <= p.x() < w and 0 <= p.y() < h

    def wheelEvent(self, event):
        if self.image_size[0] == 0:
            return
        factor = 1.18 if event.angleDelta().y() > 0 else 1 / 1.18
        new_zoom = self.zoom * factor
        if not (0.02 < new_zoom < 80):
            return
        old = self.mapToScene(event.position().toPoint())
        self.scale(factor, factor)
        new = self.mapToScene(event.position().toPoint())
        d = new - old
        self.translate(d.x(), d.y())
        self.zoomChanged.emit(self.zoom)

    def mousePressEvent(self, event):
        pos = self._scene_pos(event)
        btn = event.button()

        if btn == Qt.MiddleButton or (self._space and btn == Qt.LeftButton):
            self._panning = True
            self._pan_start = event.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        if self.image_size[0] and not self._in_image(pos) and self.tool != SELECT:
            return

        if self.tool == SAM_POINT and btn in (Qt.LeftButton, Qt.RightButton):
            self.add_prompt_point(pos.x(), pos.y(), btn == Qt.LeftButton)
            event.accept()
            return

        if self.tool in (SAM_BOX, RECT) and btn == Qt.LeftButton:
            self._dragging_box = True
            self._box_origin = pos
            self.set_prompt_box([pos.x(), pos.y(), pos.x(), pos.y()])
            event.accept()
            return

        if self.tool == POLYGON:
            if btn == Qt.LeftButton:
                self._poly_points.append((pos.x(), pos.y()))
                self._redraw_poly()
            elif btn == Qt.RightButton:
                self.finish_polygon()
            event.accept()
            return

        if self.tool == BOX2SEG and btn == Qt.LeftButton:
            item = self._shape_at(pos)
            if item is not None:
                self.convertRequested.emit(item)
            event.accept()
            return

        if self.tool == SELECT and btn == Qt.LeftButton:
            item = self._shape_at(pos)
            if item is not None:
                if event.modifiers() & Qt.AltModifier:
                    item.notify_edit_begin()
                    item.set_editing(True)
                    item.insert_vertex_near(pos)
                    self.shapeEdited.emit(item)
                    event.accept()
                    return
                if not (event.modifiers() & Qt.ShiftModifier):
                    self._scene.clearSelection()
                item.setSelected(True)
                item.notify_edit_begin()   # a drag may follow; snapshot first
                self._moving_item = item
                self._move_last = pos
            else:
                self._scene.clearSelection()
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.tool == SELECT:
            item = self._shape_at(self._scene_pos(event))
            if item is not None:
                item.setSelected(True)
                item.set_editing(True)
                event.accept()
                return
        if self.tool == POLYGON:
            self.finish_polygon()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event):
        pos = self._scene_pos(event)
        if self._in_image(pos):
            self.cursorMoved.emit(pos.x(), pos.y())
            if self.crosshair.isVisible():
                self.crosshair.move_to(pos.x(), pos.y(), *self.image_size)

        if self._panning:
            delta = event.position().toPoint() - self._pan_start
            self._pan_start = event.position().toPoint()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y())
            event.accept()
            return

        if self._dragging_box:
            x1, y1 = self._box_origin.x(), self._box_origin.y()
            self.set_prompt_box([min(x1, pos.x()), min(y1, pos.y()),
                                 max(x1, pos.x()), max(y1, pos.y())])
            event.accept()
            return

        if self._moving_item is not None and event.buttons() & Qt.LeftButton:
            d = pos - self._move_last
            self._move_last = pos
            for it in self.selected_items() or [self._moving_item]:
                it.move_by(d.x(), d.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._panning:
            self._panning = False
            self.setCursor(Qt.OpenHandCursor if self._space else Qt.ArrowCursor)
            self.set_tool(self.tool) if not self._space else None
            event.accept()
            return
        if self._dragging_box:
            self._dragging_box = False
            b = self.prompt_box
            too_small = b and (abs(b[2] - b[0]) < 4 or abs(b[3] - b[1]) < 4)
            if too_small:
                self.set_prompt_box(None)          # accidental click
            elif self.tool == RECT and b:
                # A box *label*, not a prompt: hand it over and drop the
                # rubber band so the next drag starts clean. That is what
                # lets several boxes be drawn one after another.
                box = list(b)
                self.set_prompt_box(None)
                self.boxFinished.emit(box)
            event.accept()
            return
        if self._moving_item is not None:
            self.shapeEdited.emit(self._moving_item)
            self._moving_item = None
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space = True
            self.setCursor(Qt.OpenHandCursor)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space = False
            self.set_tool(self.tool)
            event.accept()
            return
        super().keyReleaseEvent(event)

    # ---- helpers ----------------------------------------------------------
    def _shape_at(self, pos: QPointF) -> Optional[ShapeItem]:
        """Topmost shape under the cursor (smallest wins for nested shapes)."""
        hits = [i for i in self.shape_items
                if i.isVisible() and i.path().contains(pos)]
        if not hits:
            for it in self.items(self.mapFromScene(pos)):
                if isinstance(it, ShapeItem):
                    return it
            return None
        hits.sort(key=lambda i: i.path().boundingRect().width() *
                  i.path().boundingRect().height())
        return hits[0]

    def _on_selection(self) -> None:
        sel = self.selected_items()
        self.selectionChangedSig.emit(sel[0] if sel else None)
