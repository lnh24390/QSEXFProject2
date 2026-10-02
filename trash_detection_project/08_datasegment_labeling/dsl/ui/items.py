"""QGraphicsItems used by the canvas."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (QBrush, QColor, QImage, QPainter, QPainterPath, QPen,
                           QPixmap, QPolygonF)
from PySide6.QtWidgets import (QGraphicsEllipseItem, QGraphicsItem,
                               QGraphicsPathItem, QGraphicsPixmapItem)

from ..core.annotation import Shape
from .theme import qcolor

HANDLE_R = 4.0


class VertexHandle(QGraphicsEllipseItem):
    """Draggable polygon vertex. Screen-constant size via ItemIgnoresTransformations."""

    def __init__(self, index: int, parent: "ShapeItem"):
        super().__init__(-HANDLE_R, -HANDLE_R, HANDLE_R * 2, HANDLE_R * 2, parent)
        self.index = index
        self.shape_item = parent
        self.setFlags(QGraphicsItem.ItemIsMovable |
                      QGraphicsItem.ItemSendsGeometryChanges |
                      QGraphicsItem.ItemIgnoresTransformations)
        self.setZValue(30)
        self.setBrush(QBrush(QColor("#ffffff")))
        self.setPen(QPen(QColor("#000000"), 1))
        self.setCursor(Qt.SizeAllCursor)
        self.setAcceptHoverEvents(True)

    def hoverEnterEvent(self, e):
        self.setBrush(QBrush(QColor("#4c8dff")))
        super().hoverEnterEvent(e)

    def hoverLeaveEvent(self, e):
        self.setBrush(QBrush(QColor("#ffffff")))
        super().hoverLeaveEvent(e)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged and self.shape_item:
            self.shape_item.on_vertex_moved(self.index, self.pos())
        return super().itemChange(change, value)

    def mousePressEvent(self, e):
        # snapshot before the drag moves anything
        if self.shape_item is not None:
            self.shape_item.notify_edit_begin()
        super().mousePressEvent(e)

    def mouseDoubleClickEvent(self, e):
        """Double-click removes this vertex."""
        if self.shape_item is not None:
            self.shape_item.notify_edit_begin()
        self.shape_item.remove_vertex(self.index)
        e.accept()


class ShapeItem(QGraphicsPathItem):
    """One annotation instance. Coordinates are original image pixels."""

    def __init__(self, shape: Shape, color: str = "#e6194b", fill_alpha: int = 70,
                 outline_width: float = 1.8):
        super().__init__()
        self.shape_data = shape
        self.color = color
        self.fill_alpha = fill_alpha
        self.outline_width = outline_width
        self.handles: List[VertexHandle] = []
        self._editing = False
        self._updating = False
        #: called right *before* a geometry mutation so the window can take an
        #: undo snapshot of the pre-edit state. CanvasView wires this up.
        self.edit_began = None
        self.setFlags(QGraphicsItem.ItemIsSelectable)
        self.setAcceptHoverEvents(True)
        self.setZValue(10)
        self.rebuild_path()
        self.apply_style()

    def notify_edit_begin(self) -> None:
        if self.edit_began is not None:
            self.edit_began(self)

    # ---- geometry ---------------------------------------------------------
    def rebuild_path(self) -> None:
        path = QPainterPath()
        pts = self.shape_data.points
        if len(pts) >= 2:
            path.addPolygon(QPolygonF([QPointF(x, y) for x, y in pts]))
            path.closeSubpath()
        for hole in self.shape_data.holes:
            if len(hole) >= 3:
                sub = QPainterPath()
                sub.addPolygon(QPolygonF([QPointF(x, y) for x, y in hole]))
                sub.closeSubpath()
                path = path.subtracted(sub)
        self.setPath(path)

    def apply_style(self, selected: Optional[bool] = None) -> None:
        sel = self.isSelected() if selected is None else selected
        dashed = self.shape_data.shape_type == "rectangle"
        w = float(self.outline_width)
        pen = QPen(qcolor(self.color), w + 1.0 if sel else w)
        pen.setCosmetic(True)                     # constant width at any zoom
        # SAM contours have hundreds of near-collinear vertices with sharp
        # turns. The default bevel join notches every one of them, which
        # reads as a ragged, broken outline; round joins/caps make the
        # stroke look like one continuous line.
        pen.setJoinStyle(Qt.RoundJoin)
        pen.setCapStyle(Qt.RoundCap)
        if dashed:
            pen.setStyle(Qt.DashLine)
        if sel:
            pen.setColor(QColor("#ffffff"))
        self.setPen(pen)
        alpha = self.fill_alpha + (40 if sel else 0)
        if dashed:
            # Boxes are drawn fainter than segments so they do not hide the
            # pixels being judged -- but "outline only" (0) must stay 0,
            # otherwise the mode would not actually clear the fill.
            alpha = 0 if self.fill_alpha == 0 else max(10, self.fill_alpha // 3)
        self.setBrush(QBrush(qcolor(self.color, min(alpha, 255))))

    def set_color(self, color: str) -> None:
        self.color = color
        self.apply_style()

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemSelectedHasChanged:
            self.apply_style(bool(value))
            if not value:
                self.set_editing(False)
        return super().itemChange(change, value)

    # ---- vertex editing ---------------------------------------------------
    def set_editing(self, on: bool) -> None:
        if on == self._editing:
            return
        self._editing = on
        self.clear_handles()
        if on:
            for i, (x, y) in enumerate(self.shape_data.points):
                h = VertexHandle(i, self)
                h.setPos(x, y)
                self.handles.append(h)

    def clear_handles(self) -> None:
        for h in self.handles:
            h.setParentItem(None)
            if h.scene():
                h.scene().removeItem(h)
        self.handles = []

    def on_vertex_moved(self, index: int, pos: QPointF) -> None:
        if self._updating or index >= len(self.shape_data.points):
            return
        self.shape_data.points[index] = (pos.x(), pos.y())
        if self.shape_data.shape_type == "rectangle":
            self.shape_data.shape_type = "polygon"   # editing breaks the box
        self.rebuild_path()

    def remove_vertex(self, index: int) -> None:
        if len(self.shape_data.points) <= 3:
            return
        del self.shape_data.points[index]
        self.rebuild_path()
        self._editing = False
        self.set_editing(True)

    def insert_vertex_near(self, pos: QPointF) -> None:
        """Add a vertex on the closest edge (Alt+click)."""
        pts = self.shape_data.points
        if len(pts) < 2:
            return
        best_i, best_d = 0, float("inf")
        px, py = pos.x(), pos.y()
        for i in range(len(pts)):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % len(pts)]
            dx, dy = x2 - x1, y2 - y1
            L = dx * dx + dy * dy
            t = 0.0 if L == 0 else max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / L))
            cx, cy = x1 + t * dx, y1 + t * dy
            d = (px - cx) ** 2 + (py - cy) ** 2
            if d < best_d:
                best_d, best_i = d, i
        pts.insert(best_i + 1, (px, py))
        self.rebuild_path()
        self._editing = False
        self.set_editing(True)

    def move_by(self, dx: float, dy: float) -> None:
        self.shape_data.points = [(x + dx, y + dy) for x, y in self.shape_data.points]
        self.shape_data.holes = [[(x + dx, y + dy) for x, y in h]
                                 for h in self.shape_data.holes]
        self.rebuild_path()
        if self._editing:
            self._editing = False
            self.set_editing(True)


class PromptPointItem(QGraphicsEllipseItem):
    """A SAM click: green = foreground, red = background."""

    R = 5.0

    def __init__(self, x: float, y: float, positive: bool = True):
        super().__init__(-self.R, -self.R, self.R * 2, self.R * 2)
        self.positive = positive
        self.setPos(x, y)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations)
        self.setZValue(40)
        self.setBrush(QBrush(QColor("#3ddc84" if positive else "#ff5252")))
        pen = QPen(QColor("#101010"), 1.5)
        self.setPen(pen)


class MaskPreviewItem(QGraphicsPixmapItem):
    """Semi-transparent overlay of the current SAM proposal."""

    def __init__(self):
        super().__init__()
        self.setZValue(20)
        self.setOpacity(0.55)

    def set_mask(self, mask: np.ndarray, color: str = "#4c8dff") -> None:
        if mask is None or mask.size == 0:
            self.clear()
            return
        h, w = mask.shape[:2]
        c = QColor(color)
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        m = mask > 0
        rgba[..., 0][m] = c.red()
        rgba[..., 1][m] = c.green()
        rgba[..., 2][m] = c.blue()
        rgba[..., 3][m] = 255
        # edge highlight so thin structures stay visible
        img = QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()
        self.setPixmap(QPixmap.fromImage(img))
        self.setVisible(True)

    def clear(self) -> None:
        self.setPixmap(QPixmap())
        self.setVisible(False)


class CrosshairItem(QGraphicsPathItem):
    """Thin crosshair that follows the cursor in prompt modes."""

    def __init__(self):
        super().__init__()
        self.setZValue(35)
        pen = QPen(QColor(255, 255, 255, 110), 1)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setVisible(False)

    def move_to(self, x: float, y: float, w: int, h: int) -> None:
        path = QPainterPath()
        path.moveTo(0, y)
        path.lineTo(w, y)
        path.moveTo(x, 0)
        path.lineTo(x, h)
        self.setPath(path)
