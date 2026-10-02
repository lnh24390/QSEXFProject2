"""Image list: virtualised rows, thumbnails only for what is on screen."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import (QAbstractItemModel, QAbstractListModel, QModelIndex,
                            QSize, Qt, Signal,
                            QTimer)
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QLineEdit,
                               QListView, QTreeView, QVBoxLayout, QWidget)

from ...core.image_cache import LRUCountCache
from ...core.project import Project
from ..theme import FG_DIM, OK, WARN
from ..workers import ThumbnailPool

VIEW_THUMB = "썸네일"
VIEW_LIST = "목록"
VIEW_TREE = "폴더"

FILTER_ALL = "전체"
FILTER_TODO = "미라벨만"
FILTER_DONE = "라벨됨"
FILTER_VERIFIED = "검증됨"


def filter_rows(project, text: str = "", mode: str = FILTER_ALL):
    """Relative paths that pass the search box and the state filter.

    Shared by the flat and the folder model so the two views can never
    disagree about what is currently visible.
    """
    if project is None:
        return []
    t = text.strip().lower()
    out = []
    for rel in project.images:
        if t and t not in rel.lower():
            continue
        if mode == FILTER_TODO and project.is_labeled(rel):
            continue
        if mode == FILTER_DONE and not project.is_labeled(rel):
            continue
        if mode == FILTER_VERIFIED and not project.is_verified(rel):
            continue
        out.append(rel)
    return out


def row_style(project, rel: str):
    """(colour, tooltip, display text) shared by both models."""
    n = project.shape_count(rel)
    name = Path(rel).name
    disp = f"{name}   ({n})" if n else name
    if project.is_verified(rel):
        col = QColor(OK)
        state = "검증됨"
    elif project.is_labeled(rel):
        col = QColor(WARN)
        state = "라벨됨"
    else:
        col = QColor(FG_DIM)
        state = "미라벨"
    return col, f"{rel}\n{state} · 도형 {n}개", disp


class ImageListModel(QAbstractListModel):
    """Rows are relative paths. Nothing is decoded until a row is displayed."""

    def __init__(self, thumb_size: int = 96, cache_items: int = 512, parent=None,
                 show_thumbs: bool = True):
        super().__init__(parent)
        self.project: Optional[Project] = None
        self.rows: List[str] = []
        self.thumb_size = thumb_size
        self.show_thumbs = show_thumbs
        self.cache = LRUCountCache(cache_items)
        self.pool = ThumbnailPool()      # threads scale with the CPU
        self.pool.ready.connect(self._thumb_ready)
        self._placeholder = self._make_placeholder()
        self._row_of: dict = {}

    def _make_placeholder(self) -> QPixmap:
        pm = QPixmap(self.thumb_size, self.thumb_size)
        pm.fill(QColor("#2a2c30"))
        return pm

    # ---- data -------------------------------------------------------------
    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.rows)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid() or self.project is None:
            return None
        row = index.row()
        if row >= len(self.rows):
            return None
        rel = self.rows[row]

        if role == Qt.DisplayRole:
            n = self.project.shape_count(rel)
            name = Path(rel).name
            return f"{name}   ({n})" if n else name
        if role == Qt.DecorationRole:
            if not self.show_thumbs:
                # Name-only mode: no icon means no decode is ever requested,
                # which is cheaper than a cached thumbnail, not just smaller.
                return None
            # Called by the view only for rows it is about to paint.
            pm = self.cache.get(rel)
            if pm is None:
                self.pool.request(rel, self.project.abs_image(rel), self.thumb_size)
                return QIcon(self._placeholder)
            return QIcon(pm)
        if role == Qt.ForegroundRole:
            if self.project.is_verified(rel):
                return QColor(OK)
            if self.project.is_labeled(rel):
                return QColor(WARN)
            return QColor(FG_DIM)
        if role == Qt.ToolTipRole:
            state = ("검증됨" if self.project.is_verified(rel)
                     else "라벨됨" if self.project.is_labeled(rel) else "미라벨")
            return f"{rel}\n{state} · 도형 {self.project.shape_count(rel)}개"
        if role == Qt.UserRole:
            return rel
        return None

    def _thumb_ready(self, rel: str, qimg) -> None:
        if qimg is None:
            self.cache.put(rel, self._placeholder)
        else:
            self.cache.put(rel, QPixmap.fromImage(qimg))
        row = self._row_of.get(rel)
        if row is not None and row < len(self.rows) and self.rows[row] == rel:
            idx = self.index(row, 0)
            self.dataChanged.emit(idx, idx, [Qt.DecorationRole])

    # ---- population -------------------------------------------------------
    def set_project(self, project: Optional[Project]) -> None:
        self.project = project
        self.cache.clear()
        self.pool.clear()
        self.apply_filter()

    def apply_filter(self, text: str = "", mode: str = FILTER_ALL) -> None:
        self.beginResetModel()
        if self.project is None:
            self.rows = []
        else:
            self.rows = filter_rows(self.project, text, mode)
        self._row_of = {rel: i for i, rel in enumerate(self.rows)}
        self.endResetModel()

    def row_for(self, rel: str) -> int:
        return self._row_of.get(rel, -1)

    def refresh_row(self, rel: str) -> None:
        row = self._row_of.get(rel)
        if row is None:
            return
        idx = self.index(row, 0)
        self.dataChanged.emit(idx, idx)


class ImageTreeModel(QAbstractItemModel):
    """Images in a real folder hierarchy, each level collapsible.

    Nodes are rows in flat parallel lists, not allocated item objects: a
    dataset of tens of thousands of images would otherwise pay for one
    QStandardItem each (DEVELOPMENT.md 3.1). A node is an index; `internalId`
    carries `node index + 1` so that 0 can mean "child of the invisible
    root", which is the one id Qt hands back for top-level rows.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project: Optional[Project] = None
        self._name: List[str] = []            # display name of each node
        self._parent: List[int] = []          # parent node index, -1 = top
        self._kids: List[List[int]] = []      # child node indices
        self._rel: List[Optional[str]] = []   # image path, None for folders
        self._roots: List[int] = []
        self._node_of: Dict[str, int] = {}    # rel -> node index
        self._order: List[str] = []           # rels in display order

    # ---- building ---------------------------------------------------------
    def _add(self, name: str, parent: int, rel: Optional[str]) -> int:
        i = len(self._name)
        self._name.append(name)
        self._parent.append(parent)
        self._kids.append([])
        self._rel.append(rel)
        (self._kids[parent] if parent >= 0 else self._roots).append(i)
        return i

    def set_project(self, project: Optional[Project]) -> None:
        self.project = project
        self.apply_filter()

    def apply_filter(self, text: str = "", mode: str = FILTER_ALL) -> None:
        self.beginResetModel()
        self._name, self._parent, self._kids, self._rel = [], [], [], []
        self._roots, self._node_of, self._order = [], {}, []
        folders: Dict[tuple, int] = {}

        def folder_node(parts: tuple) -> int:
            """The node for a folder path, creating parents as needed."""
            if not parts:
                return -1
            hit = folders.get(parts)
            if hit is not None:
                return hit
            node = self._add(parts[-1], folder_node(parts[:-1]), None)
            folders[parts] = node
            return node

        for rel in filter_rows(self.project, text, mode):
            q = Path(rel)
            parent = folder_node(tuple(q.parts[:-1]))
            self._node_of[rel] = self._add(q.name, parent, rel)
            self._order.append(rel)
        self.endResetModel()

    # ---- structure --------------------------------------------------------
    def _children(self, parent: QModelIndex) -> List[int]:
        if not parent.isValid():
            return self._roots
        return self._kids[int(parent.internalId()) - 1]

    def index(self, row, column, parent=QModelIndex()):
        if not self.hasIndex(row, column, parent):
            return QModelIndex()
        kids = self._children(parent)
        if row >= len(kids):
            return QModelIndex()
        return self.createIndex(row, column, kids[row] + 1)

    def parent(self, index):
        if not index.isValid():
            return QModelIndex()
        node = int(index.internalId()) - 1
        up = self._parent[node]
        if up < 0:
            return QModelIndex()
        grand = self._parent[up]
        siblings = self._kids[grand] if grand >= 0 else self._roots
        return self.createIndex(siblings.index(up), 0, up + 1)

    def rowCount(self, parent=QModelIndex()) -> int:
        if parent.isValid() and parent.column() > 0:
            return 0
        return len(self._children(parent))

    def columnCount(self, parent=QModelIndex()) -> int:
        return 1

    def _leaf_count(self, node: int) -> int:
        if self._rel[node] is not None:
            return 1
        return sum(self._leaf_count(k) for k in self._kids[node])

    def data(self, index, role: int = Qt.DisplayRole):
        if not index.isValid() or self.project is None:
            return None
        node = int(index.internalId()) - 1
        rel = self._rel[node]
        if rel is None:                                   # folder row
            if role == Qt.DisplayRole:
                return f"{self._name[node]}   [{self._leaf_count(node)}]"
            if role == Qt.ForegroundRole:
                return QColor(FG_DIM)
            if role == Qt.ToolTipRole:
                return self._name[node]
            return None
        if role == Qt.DisplayRole:
            return row_style(self.project, rel)[2]
        if role == Qt.ForegroundRole:
            return row_style(self.project, rel)[0]
        if role == Qt.ToolTipRole:
            return row_style(self.project, rel)[1]
        if role == Qt.UserRole:
            return rel
        return None

    # ---- lookups used by the panel ---------------------------------------
    @property
    def rows(self) -> List[str]:
        return list(self._order)

    @property
    def folders(self) -> List[str]:
        return [n for n, r in zip(self._name, self._rel) if r is None]

    def index_for(self, rel: str) -> QModelIndex:
        node = self._node_of.get(rel)
        if node is None:
            return QModelIndex()
        up = self._parent[node]
        siblings = self._kids[up] if up >= 0 else self._roots
        return self.createIndex(siblings.index(node), 0, node + 1)

    def rel_at_offset(self, rel: str, delta: int) -> Optional[str]:
        """Next/previous image in display order, crossing folder boundaries."""
        if not self._order:
            return None
        try:
            i = self._order.index(rel)
        except ValueError:
            return self._order[0]
        return self._order[max(0, min(len(self._order) - 1, i + delta))]

    def refresh_row(self, rel: str) -> None:
        idx = self.index_for(rel)
        if idx.isValid():
            self.dataChanged.emit(idx, idx)


class FilePanel(QWidget):
    imageSelected = Signal(str)

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.model = ImageListModel(settings.thumb_size, settings.thumb_cache_items)

        self.search = QLineEdit()
        self.search.setPlaceholderText("파일명 검색…")
        self.search.setClearButtonEnabled(True)
        self.filter = QComboBox()
        self.filter.addItems([FILTER_ALL, FILTER_TODO, FILTER_DONE, FILTER_VERIFIED])
        if settings.hide_completed:
            self.filter.setCurrentText(FILTER_TODO)

        self.view = QListView()
        self.view.setModel(self.model)
        self.view.setUniformItemSizes(True)            # enables fast virtualisation
        self.view.setIconSize(QSize(settings.thumb_size, settings.thumb_size))
        self.view.setLayoutMode(QListView.Batched)
        self.view.setBatchSize(40)
        self.view.setResizeMode(QListView.Adjust)
        self.view.setSelectionMode(QListView.SingleSelection)
        self.view.setVerticalScrollMode(QListView.ScrollPerPixel)
        self.view.setSpacing(1)

        # Folder view: the same rows, grouped and collapsible. Useful when
        # a dropped dataset spans several image folders.
        self.tree_model = ImageTreeModel()
        self.tree = QTreeView()
        self.tree.setModel(self.tree_model)
        self.tree.setHeaderHidden(True)
        self.tree.setUniformRowHeights(True)
        self.tree.setSelectionMode(QTreeView.SingleSelection)
        self.tree.setVerticalScrollMode(QTreeView.ScrollPerPixel)
        self.tree.setVisible(False)

        self.mode = QComboBox()
        self.mode.addItems([VIEW_THUMB, VIEW_LIST, VIEW_TREE])
        want = getattr(settings, "file_view_mode", VIEW_THUMB)
        if want in (VIEW_THUMB, VIEW_LIST, VIEW_TREE):
            self.mode.setCurrentText(want)

        self.count_label = QLabel("0 / 0")
        self.count_label.setStyleSheet(f"color: {FG_DIM};")

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(self.search, 1)
        top.addWidget(self.filter)
        top.addWidget(self.mode)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)
        lay.addLayout(top)
        lay.addWidget(self.view, 1)
        lay.addWidget(self.tree, 1)
        lay.addWidget(self.count_label)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(200)
        self._debounce.timeout.connect(self._refilter)
        self.search.textChanged.connect(lambda _: self._debounce.start())
        self.filter.currentTextChanged.connect(lambda _: self._refilter())
        self.view.selectionModel().currentChanged.connect(self._on_current)
        self.tree.selectionModel().currentChanged.connect(self._on_current)
        self.mode.currentTextChanged.connect(self._on_mode)
        self._apply_mode()

    # ---- view mode --------------------------------------------------------
    @property
    def _view(self):
        return self.tree if self.mode.currentText() == VIEW_TREE else self.view

    @property
    def _model(self):
        return (self.tree_model if self.mode.currentText() == VIEW_TREE
                else self.model)

    def _apply_mode(self) -> None:
        m = self.mode.currentText()
        self.model.show_thumbs = (m == VIEW_THUMB)
        self.view.setIconSize(
            QSize(self.settings.thumb_size, self.settings.thumb_size)
            if m == VIEW_THUMB else QSize(0, 0))
        self.view.setVisible(m != VIEW_TREE)
        self.tree.setVisible(m == VIEW_TREE)

    def _on_mode(self, _text: str) -> None:
        keep = self.current_rel()
        self.settings.file_view_mode = self.mode.currentText()
        self.settings.save()
        self._apply_mode()
        self._refilter()
        if keep:
            self.select(keep, emit=False)

    # ---- api --------------------------------------------------------------
    def set_project(self, project: Optional[Project]) -> None:
        self.model.set_project(project)
        self.tree_model.set_project(project)
        self._refilter()

    def _refilter(self) -> None:
        keep = self.current_rel()
        text, mode = self.search.text(), self.filter.currentText()
        self.model.apply_filter(text, mode)
        self.tree_model.apply_filter(text, mode)
        if self.mode.currentText() == VIEW_TREE:
            self.tree.expandAll()
        self._update_count()
        if keep:
            self.select(keep, emit=False)

    def _update_count(self) -> None:
        p = self.model.project
        total = len(p.images) if p else 0
        st = p.stats() if p else {"labeled": 0, "verified": 0}
        self.count_label.setText(
            f"표시 {len(self.model.rows)} / 전체 {total}   ·   "
            f"라벨 {st['labeled']} · 검증 {st.get('verified', 0)}")

    def current_rel(self) -> Optional[str]:
        idx = self._view.currentIndex()
        if not idx.isValid():
            return None
        return self._model.data(idx, Qt.UserRole)

    def select(self, rel: str, emit: bool = True) -> bool:
        view, model = self._view, self._model
        if view is self.tree:
            idx = model.index_for(rel)
            if idx.isValid():
                self.tree.expand(idx.parent())
        else:
            row = model.row_for(rel)
            idx = model.index(row, 0) if row >= 0 else QModelIndex()
        if not idx.isValid():
            return False
        blocked = view.selectionModel().blockSignals(not emit)
        view.setCurrentIndex(idx)
        view.scrollTo(idx, QListView.EnsureVisible)
        view.selectionModel().blockSignals(blocked)
        return True

    def step(self, delta: int) -> None:
        if self._view is self.tree:
            cur = self.current_rel()
            nxt = self.tree_model.rel_at_offset(cur, delta) if cur else None
            if nxt and nxt != cur:
                self.select(nxt)
            return
        row = self.view.currentIndex().row()
        new = max(0, min(len(self.model.rows) - 1, row + delta))
        if new != row:
            self.view.setCurrentIndex(self.model.index(new, 0))

    def refresh_row(self, rel: str) -> None:
        self.model.refresh_row(rel)
        self.tree_model.refresh_row(rel)
        self._update_count()

    def drop_row(self, rel: str) -> None:
        """Remove a finished image from view when '미라벨만' is active."""
        if self.filter.currentText() != FILTER_TODO:
            self.refresh_row(rel)
            return
        self._refilter()

    def _on_current(self, cur, _prev) -> None:
        if cur.isValid():
            rel = self._model.data(cur, Qt.UserRole)
            if rel:
                self.imageSelected.emit(rel)
