"""Shapes of the current image."""
from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QHBoxLayout, QHeaderView,
                               QPushButton, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from ...core.annotation import Shape
from ..panels.class_panel import color_icon


class ShapePanel(QWidget):
    shapeSelected = Signal(object)         # Shape | None
    deleteRequested = Signal(list)         # [Shape]
    classAssignRequested = Signal(list)    # [Shape] -> assign current class

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self.shapes: List[Shape] = []

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["#", "클래스", "출처", "점수"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setShowGrid(False)
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(1, QHeaderView.Stretch)
        h.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(3, QHeaderView.ResizeToContents)

        self.btn_assign = QPushButton("선택 도형에 현재 클래스 적용")
        self.btn_assign.setToolTip(
            "1) [클래스] 패널에서 클래스를 고르고\n"
            "2) 캔버스나 이 표에서 도형을 선택한 뒤\n"
            "3) 이 버튼을 누릅니다. (숫자키 1-9 로 클래스 전환)")
        self.btn_del = QPushButton("삭제")
        self.btn_del.setObjectName("danger")
        # Both act on the current selection; showing them disabled says so
        # far more clearly than a button that silently does nothing.
        self.btn_assign.setEnabled(False)
        self.btn_del.setEnabled(False)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.btn_assign, 1)
        row.addWidget(self.btn_del)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)
        lay.addWidget(self.table, 1)
        lay.addLayout(row)

        self.table.itemSelectionChanged.connect(self._on_select)
        self.btn_del.clicked.connect(
            lambda: self.deleteRequested.emit(self.selected_shapes()))
        self.btn_assign.clicked.connect(
            lambda: self.classAssignRequested.emit(self.selected_shapes()))

    def set_project(self, project) -> None:
        self.project = project

    def set_shapes(self, shapes: List[Shape]) -> None:
        self.shapes = list(shapes)
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.shapes))
        for r, s in enumerate(self.shapes):
            cls = self.project.class_by_id(s.class_id) if self.project else None
            name = cls.name if cls else f"id{s.class_id}"
            it0 = QTableWidgetItem(str(r + 1))
            it1 = QTableWidgetItem(name)
            if cls:
                it1.setIcon(color_icon(cls.color))
            it2 = QTableWidgetItem("박스" if s.shape_type == "rectangle" else s.source)
            it3 = QTableWidgetItem(f"{s.score:.2f}")
            for c, it in enumerate((it0, it1, it2, it3)):
                it.setData(Qt.UserRole, r)
                self.table.setItem(r, c, it)
        self.table.blockSignals(False)

    def selected_shapes(self) -> List[Shape]:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        return [self.shapes[r] for r in rows if r < len(self.shapes)]

    def select_shape(self, shape: Optional[Shape]) -> None:
        self.table.blockSignals(True)
        self.table.clearSelection()
        if shape is not None:
            for r, s in enumerate(self.shapes):
                if s is shape or s.uid == shape.uid:
                    self.table.selectRow(r)
                    break
        self.table.blockSignals(False)

    def _on_select(self) -> None:
        sel = self.selected_shapes()
        self.btn_assign.setEnabled(bool(sel))
        self.btn_del.setEnabled(bool(sel))
        self.shapeSelected.emit(sel[0] if sel else None)
