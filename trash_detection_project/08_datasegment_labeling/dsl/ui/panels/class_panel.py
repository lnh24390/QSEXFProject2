"""Class list: create / rename / recolour / delete label classes."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (QColorDialog, QHBoxLayout, QInputDialog, QLabel,
                               QListWidget, QListWidgetItem, QMessageBox,
                               QPushButton, QVBoxLayout, QWidget)

from ...core.project import Project
from ..theme import FG_DIM


def color_icon(hex_color: str, size: int = 14) -> QIcon:
    pm = QPixmap(size, size)
    pm.fill(QColor(hex_color))
    return QIcon(pm)


class ClassPanel(QWidget):
    currentClassChanged = Signal(int)      # class id
    classesChanged = Signal()              # list or colours edited

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project: Optional[Project] = None

        self.list = QListWidget()
        self.list.setAlternatingRowColors(False)

        self.btn_add = QPushButton("추가")
        self.btn_edit = QPushButton("이름")
        self.btn_color = QPushButton("색")
        self.btn_del = QPushButton("삭제")
        self.btn_del.setObjectName("danger")
        for b in (self.btn_add, self.btn_edit, self.btn_color, self.btn_del):
            b.setFixedHeight(26)

        hint = QLabel("숫자키 1–9 로 클래스 전환")
        hint.setStyleSheet(f"color: {FG_DIM}; font-size: 11px;")

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(3)
        for b in (self.btn_add, self.btn_edit, self.btn_color, self.btn_del):
            row.addWidget(b)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)
        lay.addWidget(self.list, 1)
        lay.addLayout(row)
        lay.addWidget(hint)

        self.btn_add.clicked.connect(self.add_class)
        self.btn_edit.clicked.connect(self.rename_class)
        self.btn_color.clicked.connect(self.change_color)
        self.btn_del.clicked.connect(self.delete_class)
        self.list.currentRowChanged.connect(self._on_row)
        self.list.itemDoubleClicked.connect(lambda _i: self.rename_class())

    # ---- data -------------------------------------------------------------
    def set_project(self, project: Optional[Project]) -> None:
        self.project = project
        self.reload()

    def reload(self) -> None:
        keep = self.current_class_id()
        self.list.blockSignals(True)
        self.list.clear()
        if self.project:
            for i, c in enumerate(self.project.classes):
                item = QListWidgetItem(color_icon(c.color), f"{i + 1}. {c.name}")
                item.setData(Qt.UserRole, c.id)
                self.list.addItem(item)
        self.list.blockSignals(False)
        if self.list.count():
            row = 0
            if keep is not None:
                for i in range(self.list.count()):
                    if self.list.item(i).data(Qt.UserRole) == keep:
                        row = i
                        break
            self.list.setCurrentRow(row)

    def current_class_id(self) -> Optional[int]:
        item = self.list.currentItem()
        return None if item is None else int(item.data(Qt.UserRole))

    def select_index(self, idx: int) -> None:
        if 0 <= idx < self.list.count():
            self.list.setCurrentRow(idx)

    def _on_row(self, row: int) -> None:
        if row >= 0:
            item = self.list.item(row)
            if item:
                self.currentClassChanged.emit(int(item.data(Qt.UserRole)))

    # ---- actions ----------------------------------------------------------
    def add_class(self) -> None:
        if not self.project:
            return
        name, ok = QInputDialog.getText(self, "클래스 추가", "클래스 이름:")
        name = name.strip()
        if not ok or not name:
            return
        if self.project.class_by_name(name):
            QMessageBox.warning(self, "중복", f"'{name}' 클래스가 이미 있습니다.")
            return
        c = self.project.add_class(name)
        self.project.save()
        self.reload()
        for i in range(self.list.count()):
            if self.list.item(i).data(Qt.UserRole) == c.id:
                self.list.setCurrentRow(i)
                break
        self.classesChanged.emit()

    def rename_class(self) -> None:
        cid = self.current_class_id()
        if not self.project or cid is None:
            return
        c = self.project.class_by_id(cid)
        name, ok = QInputDialog.getText(self, "클래스 이름 변경", "새 이름:", text=c.name)
        name = name.strip()
        if not ok or not name or name == c.name:
            return
        c.name = name
        self.project.save()
        self.reload()
        self.classesChanged.emit()

    def change_color(self) -> None:
        cid = self.current_class_id()
        if not self.project or cid is None:
            return
        c = self.project.class_by_id(cid)
        col = QColorDialog.getColor(QColor(c.color), self, "클래스 색상")
        if not col.isValid():
            return
        c.color = col.name()
        self.project.save()
        self.reload()
        self.classesChanged.emit()

    def delete_class(self) -> None:
        cid = self.current_class_id()
        if not self.project or cid is None:
            return
        c = self.project.class_by_id(cid)
        n = sum(int(e.get("cls", {}).get(str(cid), 0))
                for e in self.project.index.values())
        msg = f"'{c.name}' 클래스를 삭제할까요?"
        if n:
            msg += f"\n\n이 클래스로 라벨된 도형 {n}개도 함께 삭제됩니다."
        if QMessageBox.question(self, "클래스 삭제", msg) != QMessageBox.Yes:
            return
        self.project.remove_class(cid)
        self.project.save()
        self.reload()
        self.classesChanged.emit()
