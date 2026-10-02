"""Main window: wires canvas, panels, SAM worker and the training runner."""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PySide6.QtCore import QSize, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (QApplication, QDockWidget, QFileDialog,
                               QInputDialog, QLabel, QMainWindow, QMessageBox,
                               QScrollArea, QTabWidget, QToolBar, QWidget)

from ..core.annotation import ImageAnnotation, Shape
from ..core.image_cache import LRUImageCache
from ..core.mask_utils import mask_to_polygons
from ..core.project import IMAGE_EXTS, Project
from ..core.settings import Settings, resolve_device, resolve_under_app
from ..sam import catalog
from ..sam.downloader import cuda_guidance, probe_hardware
from ..train.jobs import TrainJob
from ..train.runner import TrainingProcess
from .canvas import (BOX2SEG, POLYGON, RECT, SAM_BOX, SAM_POINT, SELECT,
                     TOOL_HINTS, CanvasView)
from .dialogs.data_io import (AutoLabelDialog, ExportDatasetDialog,
                              ImportLabelsDialog)
from .dialogs.model_manager import ModelManagerDialog
from .items import ShapeItem
from .panels.class_panel import ClassPanel
from .panels.file_panel import FILTER_TODO, FilePanel
from .panels.sam_panel import SamPanel
from .panels.shape_panel import ShapePanel
from .panels.train_panel import TrainPanel
from .theme import FG_DIM
from .workers import SamWorker

UNDO_LIMIT = 40


class MainWindow(QMainWindow):
    # -> SAM worker thread
    samLoad = Signal(str, str, str, str)        # key, weights_dir, device, precision
    samUnload = Signal()
    samSetImage = Signal(object, str)
    samPredict = Signal(list, object, int, int, int)
    samPredictText = Signal(str, int, int, int)
    samBatch = Signal(object, list, int, float, float, bool, int)
    samBatchText = Signal(object, list, str, int, float, float, int)
    samBatchAuto = Signal(object, list, int, float, float, int, int)
    samMaxSide = Signal(int)
    samCancelBatch = Signal()

    # <- training subprocess stdout pump (non-GUI thread)
    trainLine = Signal(str)
    trainMetric = Signal(object)
    trainDone = Signal(int)

    def __init__(self, settings: Optional[Settings] = None):
        super().__init__()
        self.settings = settings or Settings.load()
        self.setWindowTitle("DataSegmentLabeling — SAM 세그먼트 라벨링 & 학습")
        self.setMinimumSize(1280, 720)
        self.setAcceptDrops(True)          # drop a dataset folder to open it

        self.project: Optional[Project] = None
        self.current_rel: Optional[str] = None
        self.annotation: Optional[ImageAnnotation] = None
        self.current_class_id: Optional[int] = None
        self.cache = LRUImageCache(self.settings.image_cache_bytes)
        self.hw = probe_hardware(self.settings.weights_path)

        self._dirty = False
        self._req = 0
        self._candidates: Optional[np.ndarray] = None
        self._cand_scores: Optional[np.ndarray] = None
        self._cand_index = 0
        self._pending_convert: Dict[int, ShapeItem] = {}
        self._pending_text: Dict[int, bool] = {}
        self._undo: List[list] = []
        self._redo: List[list] = []
        self._sam_ready_for: Optional[str] = None
        self.trainer: Optional[TrainingProcess] = None
        self._temp_projects: List[Path] = []     # read-only sessions

        self._build_ui()
        self._build_actions()
        self._fit_to_screen(1680, 960)
        self._start_sam_thread()
        self._update_title()
        self._restore_last_project()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        self.canvas = CanvasView()
        self.canvas.fill_alpha = self.settings.fill_alpha
        self.canvas.outline_width = self.settings.outline_width
        self.setCentralWidget(self.canvas)

        self.file_panel = FilePanel(self.settings)
        self.class_panel = ClassPanel()
        self.shape_panel = ShapePanel()
        self.sam_panel = SamPanel(self.settings)
        self.train_panel = TrainPanel(self.settings)

        self.dock_files = self._dock("이미지", self.file_panel, Qt.LeftDockWidgetArea, 300)
        self.dock_class = self._dock("클래스", self.class_panel, Qt.RightDockWidgetArea, 300)
        self.dock_shape = self._dock("도형", self.shape_panel, Qt.RightDockWidgetArea, 300)

        self.right_tabs = QTabWidget()
        self.right_tabs.addTab(self.sam_panel, "SAM")
        self.right_tabs.addTab(self.train_panel, "학습")
        self.dock_tools = self._dock("도구", self.right_tabs, Qt.RightDockWidgetArea, 360)
        self.splitDockWidget(self.dock_class, self.dock_shape, Qt.Vertical)
        self.splitDockWidget(self.dock_shape, self.dock_tools, Qt.Vertical)
        self.resizeDocks([self.dock_class, self.dock_shape, self.dock_tools],
                         [180, 220, 560], Qt.Vertical)

        sb = self.statusBar()
        self.lbl_hint = QLabel(TOOL_HINTS[SELECT])
        self.lbl_pos = QLabel("—")
        self.lbl_zoom = QLabel("100%")
        self.lbl_mem = QLabel("")
        self.lbl_sam = QLabel("SAM: 미로딩")
        for w in (self.lbl_pos, self.lbl_zoom, self.lbl_mem, self.lbl_sam):
            w.setStyleSheet(f"color: {FG_DIM};")
        sb.addWidget(self.lbl_hint, 1)
        for w in (self.lbl_pos, self.lbl_zoom, self.lbl_mem, self.lbl_sam):
            sb.addPermanentWidget(w)

        # signals
        self.file_panel.imageSelected.connect(self.open_image)
        self.class_panel.currentClassChanged.connect(self._on_class_changed)
        self.class_panel.classesChanged.connect(self._on_classes_changed)
        self.shape_panel.shapeSelected.connect(self._on_shape_row_selected)
        self.shape_panel.deleteRequested.connect(self.delete_shapes)
        self.shape_panel.classAssignRequested.connect(self.assign_class)

        self.canvas.promptChanged.connect(self._on_prompt)
        self.canvas.polygonFinished.connect(self._on_manual_polygon)
        self.canvas.boxFinished.connect(self._on_manual_box)
        self.canvas.convertRequested.connect(self._on_convert_request)
        self.canvas.selectionChangedSig.connect(self._on_canvas_selection)
        self.canvas.cursorMoved.connect(
            lambda x, y: self.lbl_pos.setText(f"({int(x)}, {int(y)})"))
        self.canvas.zoomChanged.connect(
            lambda z: self.lbl_zoom.setText(f"{z * 100:.0f}%"))
        # snapshot *before* the geometry changes, so Ctrl+Z restores the shape
        # as it was when the drag/insert/delete started
        self.canvas.editAboutToStart.connect(lambda _i: self._snapshot())
        self.canvas.shapeEdited.connect(lambda _i: self._mark_dirty())

        self.sam_panel.loadRequested.connect(self._load_sam)
        self.sam_panel.unloadRequested.connect(self._unload_sam)
        self.sam_panel.openManagerRequested.connect(self.open_model_manager)
        self.sam_panel.textPromptRequested.connect(self._on_text_prompt)
        self.sam_panel.batchConvertRequested.connect(self._start_batch_convert)
        self.sam_panel.batchTextRequested.connect(self._start_batch_text)
        self.sam_panel.autoSegmentRequested.connect(self._start_auto_segment)
        self.sam_panel.cancelBatchRequested.connect(self.samCancelBatch.emit)
        self.sam_panel.settingChanged.connect(self.settings.save)

        self.train_panel.exportRequested.connect(self.export_dataset)
        self.train_panel.trainRequested.connect(self.start_training)
        self.train_panel.stopRequested.connect(self.stop_training)
        self.train_panel.openRunDirRequested.connect(self._open_runs)

        # The trainer's stdout pump is a plain thread; these signals hop the
        # data back onto the GUI thread before any widget is touched.
        self.trainLine.connect(self.train_panel.append_log)
        self.trainMetric.connect(self.train_panel.update_metrics)
        self.trainDone.connect(self._on_train_done)

        self._mem_timer = QTimer(self)
        self._mem_timer.timeout.connect(self._update_memory_label)
        self._mem_timer.start(1500)

        self._prefetch_timer = QTimer(self)
        self._prefetch_timer.setSingleShot(True)
        self._prefetch_timer.timeout.connect(self._do_prefetch)

    def _fit_to_screen(self, want_w: int, want_h: int) -> None:
        """Open at the preferred size, but never bigger than the screen offers.

        A hard-coded 1680x960 does not fit a 1366x768 laptop: the window
        opens with its lower edge under the taskbar, and a QMainWindow has no
        scrollbar of its own, so the status bar and the bottom of the docks
        become unreachable. Clamp to the available area (which already
        excludes the taskbar) and centre what is left.
        """
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:                      # offscreen platform in tests
            self.resize(want_w, want_h)
            return
        avail = screen.availableGeometry()
        w = max(self.minimumWidth(), min(want_w, avail.width()))
        h = max(self.minimumHeight(), min(want_h, avail.height()))
        self.resize(w, h)
        self.move(avail.x() + max(0, (avail.width() - w) // 2),
                  avail.y() + max(0, (avail.height() - h) // 2))

    def _dock(self, title: str, widget: QWidget, area, width: int) -> QDockWidget:
        """A dock whose contents scroll instead of being clipped.

        Three docks stack down the right edge, and their panels together are
        taller than a 720px window. Without the scroll area the overflow is
        simply unreachable -- a dock clips its widget and offers no way to
        pan. Wrapping also lets the dock shrink below its panel's sizeHint,
        which is what makes a 1280x720 window possible at all.
        """
        scroll = QScrollArea()
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)          # panel follows the dock width
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        d = QDockWidget(title, self)
        d.setWidget(scroll)
        d.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        # A minimum *width* is a layout hint; a minimum height would defeat
        # the scroll area, so the dock is free to become short.
        d.setMinimumWidth(width)
        self.addDockWidget(area, d)
        return d

    # -------------------------------------------------------------- actions
    def _act(self, text, slot, shortcut=None, checkable=False, tip=""):
        a = QAction(text, self)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        a.setCheckable(checkable)
        a.setToolTip(tip or text)
        a.setStatusTip(tip or text)
        a.triggered.connect(slot)
        return a

    def _build_actions(self) -> None:
        m = self.menuBar()

        # --- file
        fm = m.addMenu("파일(&F)")
        self.a_new = self._act("새 프로젝트…", self.new_project, "Ctrl+N")
        self.a_open = self._act("프로젝트 열기…", self.open_project, "Ctrl+O")
        self.a_open_dir = self._act("이미지 폴더 열기…", self.open_image_folder,
                                    "Ctrl+Shift+O")
        self.a_open_pair = self._act(
            "이미지+라벨 함께 열기…", lambda: self.open_images_with_labels(False),
            tip="이미지 폴더와 라벨 폴더를 차례로 골라 한 번에 엽니다")
        self.a_open_view = self._act(
            "이미지+라벨 열기 (열람 전용)…",
            lambda: self.open_images_with_labels(True),
            tip="라벨을 보기만 합니다. 앱 폴더에 프로젝트를 남기지 않습니다")
        self.a_save = self._act("저장", self.save_current, "Ctrl+S")
        self.a_add_img = self._act("이미지 추가…", self.add_images)
        self.a_add_dir = self._act("이미지 폴더 지정…", self.set_image_dir)
        self.a_import = self._act("기존 라벨 가져오기…", self.import_labels, "Ctrl+I")
        self.a_export = self._act("데이터셋 내보내기…", lambda: self.export_dataset(),
                                  "Ctrl+E")
        self.a_quit = self._act("종료", self.close, "Ctrl+Q")
        for a in (self.a_new, self.a_open, self.a_open_dir, self.a_open_pair,
                  self.a_open_view, self.a_save, None,
                  self.a_add_img,
                  self.a_add_dir, None, self.a_import, self.a_export, None,
                  self.a_quit):
            fm.addSeparator() if a is None else fm.addAction(a)

        # --- edit
        em = m.addMenu("편집(&E)")
        self.a_undo = self._act("실행 취소", self.undo, "Ctrl+Z")
        self.a_redo = self._act("다시 실행", self.redo, "Ctrl+Shift+Z")
        self.a_del = self._act("선택 삭제", self.delete_selected, "Delete")
        self.a_all = self._act("모두 선택", self.select_all, "Ctrl+A")
        self.a_clear = self._act("이 이미지 라벨 비우기", self.clear_image_labels)
        self.a_verify = self._act("검증 완료 표시", self.toggle_verified, "Ctrl+D")
        for a in (self.a_undo, self.a_redo, None, self.a_del, self.a_all, None,
                  self.a_clear, self.a_verify):
            em.addSeparator() if a is None else em.addAction(a)

        # --- tools
        tm = m.addMenu("도구(&T)")
        self.tool_group = QActionGroup(self)
        self.tool_group.setExclusive(True)
        self.tool_actions = {}
        for key, label, sc in ((SELECT, "선택/편집", "1"),
                               (SAM_POINT, "SAM 클릭", "2"),
                               (SAM_BOX, "SAM 박스", "3"),
                               (POLYGON, "수동 폴리곤", "4"),
                               (BOX2SEG, "박스→세그먼트", "5"),
                               (RECT, "박스 라벨", "6")):
            a = self._act(f"{label}  ({sc})", lambda _c=False, k=key: self.set_tool(k),
                          f"Ctrl+{sc}", checkable=True, tip=TOOL_HINTS[key])
            self.tool_group.addAction(a)
            tm.addAction(a)
            self.tool_actions[key] = a
        self.tool_actions[SELECT].setChecked(True)
        tm.addSeparator()
        self.a_seg2box = self._act(
            "세그→박스", self.shapes_to_boxes, "Ctrl+Shift+B",
            tip="선택한 세그먼트를 바운딩 박스로 바꿉니다 "
                "(원본 폴리곤을 보관해 되돌릴 수 있습니다)")
        self.a_box2seg_sel = self._act(
            "박스→세그", self.boxes_to_shapes, "Ctrl+Shift+G",
            tip="선택한 박스를 세그먼트로 바꿉니다 "
                "(세그먼트에서 온 박스는 원본 그대로 복원)")
        tm.addAction(self.a_seg2box)
        tm.addAction(self.a_box2seg_sel)
        tm.addSeparator()
        self.a_accept = self._act("마스크 확정", self.accept_mask, "Return")
        self.a_cancel = self._act("프롬프트 취소", self.cancel_prompt, "Escape")
        self.a_next_cand = self._act("다음 후보", lambda: self.cycle_candidate(1), "]")
        self.a_prev_cand = self._act("이전 후보", lambda: self.cycle_candidate(-1), "[")
        for a in (self.a_accept, self.a_cancel, self.a_next_cand, self.a_prev_cand):
            tm.addAction(a)
        tm.addSeparator()
        tm.addAction(self._act("SAM 모델 관리자…", self.open_model_manager, "Ctrl+M"))
        tm.addSeparator()
        tm.addAction(self._act(
            "학습한 모델로 자동 라벨링…", self.auto_label,
            tip="이 프로젝트에서 학습한 체크포인트로 라벨을 미리 채웁니다"))

        # --- view
        vm = m.addMenu("보기(&V)")
        self.a_fit = self._act("화면 맞춤", self.canvas.fit_to_view, "Ctrl+0")
        self.a_zoom_in = self._act("확대", lambda: self.canvas.zoom_to(self.canvas.zoom * 1.25), "Ctrl+=")
        self.a_zoom_out = self._act("축소", lambda: self.canvas.zoom_to(self.canvas.zoom / 1.25), "Ctrl+-")
        self.a_hide = self._act("라벨 숨기기", self.toggle_shapes, "H", checkable=True)
        self.a_prev = self._act("이전 이미지", lambda: self.file_panel.step(-1), "A")
        self.a_next = self._act("다음 이미지", lambda: self.file_panel.step(1), "D")
        self.a_outline = self._act("외곽선 굵기…", self._set_outline_width)
        self.a_fill = self._act("채우기 투명도…", self._set_fill_alpha)
        self.a_outline_only = self._act("외곽선만 보기", self._toggle_outline_only,
                                        "Ctrl+Shift+F", checkable=True)
        self.a_outline_only.setChecked(self.settings.fill_alpha == 0)
        for a in (self.a_fit, self.a_zoom_in, self.a_zoom_out, None, self.a_hide,
                  self.a_outline_only, self.a_outline, self.a_fill,
                  None, self.a_prev, self.a_next):
            vm.addSeparator() if a is None else vm.addAction(a)

        # --- options
        om = m.addMenu("설정(&S)")
        self.a_autosave = self._act("자동 저장", self._toggle_autosave, checkable=True)
        self.a_autosave.setChecked(self.settings.auto_save)
        self.a_autoadv = self._act("저장 후 다음 미라벨로 이동", self._toggle_autoadvance,
                                   checkable=True)
        self.a_autoadv.setChecked(self.settings.auto_advance)
        self.a_hidedone = self._act("완료 이미지 목록에서 숨기기", self._toggle_hide_done,
                                    checkable=True)
        self.a_hidedone.setChecked(self.settings.hide_completed)
        self.a_cache = self._act("이미지 캐시 크기…", self._set_cache_size)
        self.a_trainpy = self._act("학습용 파이썬 경로…", self._set_train_python)
        for a in (self.a_autosave, self.a_autoadv, self.a_hidedone, None,
                  self.a_cache, self.a_trainpy):
            om.addSeparator() if a is None else om.addAction(a)

        # --- help
        hm = m.addMenu("도움말(&H)")
        hm.addAction(self._act("단축키", self.show_shortcuts, "F1"))
        hm.addAction(self._act("이 PC 정보", self.show_hardware))
        hm.addAction(self._act("선택 패키지 설치 상태…", self.show_dependencies))

        # --- toolbar
        tb = QToolBar("main")
        tb.setIconSize(QSize(16, 16))
        tb.setMovable(False)
        self.addToolBar(tb)
        tb.addAction(self.a_open)
        tb.addAction(self.a_save)
        tb.addSeparator()
        for key in (SELECT, SAM_POINT, SAM_BOX, POLYGON, RECT, BOX2SEG):
            tb.addAction(self.tool_actions[key])
        tb.addSeparator()
        tb.addAction(self.a_seg2box)
        tb.addAction(self.a_box2seg_sel)
        tb.addSeparator()
        tb.addAction(self.a_undo)
        tb.addAction(self.a_del)
        tb.addSeparator()
        tb.addAction(self.a_prev)
        tb.addAction(self.a_next)
        tb.addAction(self.a_verify)
        tb.addSeparator()
        tb.addAction(self.a_import)
        tb.addAction(self.a_export)

        for i in range(1, 10):                # class hotkeys
            a = QAction(self)
            a.setShortcut(QKeySequence(str(i)))
            a.triggered.connect(lambda _c=False, n=i - 1: self.class_panel.select_index(n))
            self.addAction(a)

    # ---------------------------------------------------------- SAM thread
    def _start_sam_thread(self) -> None:
        self.sam_thread = QThread(self)
        self.sam = SamWorker()
        self.sam.moveToThread(self.sam_thread)
        self.samLoad.connect(self.sam.load_model)
        self.samUnload.connect(self.sam.unload_model)
        self.samSetImage.connect(self.sam.set_image)
        self.samPredict.connect(self.sam.predict)
        self.samPredictText.connect(self.sam.predict_text)
        self.samBatch.connect(self.sam.batch_convert)
        self.samBatchText.connect(self.sam.batch_text)
        self.samBatchAuto.connect(self.sam.batch_auto)
        self.samMaxSide.connect(self.sam.set_max_side)
        self.samCancelBatch.connect(self.sam.cancel_batch)
        self.sam.loaded.connect(self._on_sam_loaded)
        self.sam.capabilities.connect(self._on_sam_capabilities)
        self.sam.loadFailed.connect(self._on_sam_failed)
        self.sam.imageReady.connect(self._on_sam_image_ready)
        self.sam.result.connect(self._on_sam_result)
        self.sam.failed.connect(self._on_sam_error)
        self.sam.busy.connect(self._on_sam_busy)
        self.sam.batchProgress.connect(self.sam_panel.set_batch_progress)
        self.sam.batchFinished.connect(self._on_batch_finished)
        self.sam_thread.start()
        self.samMaxSide.emit(int(self.settings.sam_max_side))

    def _load_sam(self, key: str, device: str, precision: str) -> None:
        spec = catalog.get(key)
        if spec and not spec.is_installed(Path(self.settings.weights_path)):
            if QMessageBox.question(
                    self, "다운로드 필요",
                    f"{spec.display} 체크포인트가 없습니다.\n모델 관리자를 열까요?"
                    ) == QMessageBox.Yes:
                self.open_model_manager()
            return
        self.settings.sam_backend = key
        self.settings.device = device
        self.settings.sam_precision = precision
        self.settings.save()
        self.lbl_sam.setText("SAM: 로딩 중…")
        self._sam_ready_for = None
        self.samLoad.emit(key, str(self.settings.weights_path),
                          resolve_device(device), precision)

    def _unload_sam(self) -> None:
        self.samUnload.emit()
        self._sam_ready_for = None
        self.lbl_sam.setText("SAM: 미로딩")
        self.sam_panel.status.setText("모델 해제됨")
        self._on_sam_capabilities(True, True, True)   # no model = no limits

    def _on_sam_capabilities(self, points: bool, box: bool, _text: bool) -> None:
        """Grey out prompt tools the loaded backend cannot serve."""
        for key, ok in ((SAM_POINT, points), (SAM_BOX, box),
                        (BOX2SEG, box)):          # box->segment needs boxes
            act = self.tool_actions.get(key)
            if act is None:
                continue
            act.setEnabled(ok)
            if not ok:
                act.setToolTip("이 모델이 지원하지 않는 프롬프트입니다")
                if self.canvas.tool == key:
                    self.set_tool(SELECT)
            else:
                act.setToolTip("")

    def _on_sam_loaded(self, text: str) -> None:
        self.sam_panel.set_loaded(text)
        self.lbl_sam.setText(f"SAM: {text}")
        self._sam_ready_for = None

    def _on_sam_failed(self, msg: str) -> None:
        self.sam_panel.set_error(msg.split("\n")[0])
        self.lbl_sam.setText("SAM: 로딩 실패")
        QMessageBox.warning(self, "SAM 로딩 실패", msg)

    def _on_sam_busy(self, busy: bool) -> None:
        QApplication.setOverrideCursor(Qt.BusyCursor) if busy \
            else QApplication.restoreOverrideCursor()

    def _on_sam_error(self, msg: str) -> None:
        self.statusBar().showMessage(msg, 8000)
        self.sam_panel.set_error(msg.split("\n")[0])

    def _on_sam_image_ready(self, key: str) -> None:
        self._sam_ready_for = key
        self.lbl_sam.setText(self.lbl_sam.text().split("  |")[0] + "  | 인코딩 완료")

    # ------------------------------------------------------------ projects
    def _restore_last_project(self) -> None:
        for p in list(self.settings.recent_projects):
            f = Path(p)
            if (f / "project.json").exists():
                self.load_project(f)
                return

    def new_project(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "새 프로젝트 폴더 선택")
        if not d:
            return
        root = Path(d)
        if (root / "project.json").exists():
            self.load_project(root)
            return
        name, ok = QInputDialog.getText(self, "프로젝트 이름", "이름:", text=root.name)
        if not ok:
            return
        img_dir = QFileDialog.getExistingDirectory(
            self, "이미지 폴더 선택 (취소하면 <프로젝트>/images 사용)", str(root))
        project = Project.create(root, name or root.name,
                                 image_dir=img_dir or "images")
        self._set_project(project)

    def open_project(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "프로젝트 열기", "",
                                           "프로젝트 (project.json)")
        if f:
            self.load_project(Path(f).parent)

    def open_image_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "이미지 폴더 열기")
        if d:
            self.load_image_folder(Path(d))

    # ---- drag & drop --------------------------------------------------
    def dragEnterEvent(self, event):
        """Accept a folder; anything else is left to the default handling."""
        md = event.mimeData()
        if md.hasUrls() and any(Path(u.toLocalFile()).is_dir()
                                for u in md.urls() if u.isLocalFile()):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        dirs = [Path(u.toLocalFile()) for u in event.mimeData().urls()
                if u.isLocalFile() and Path(u.toLocalFile()).is_dir()]
        if not dirs:
            super().dropEvent(event)
            return
        event.acceptProposedAction()
        self.open_dataset_folder(dirs[0])

    def open_dataset_folder(self, root: Path) -> bool:
        """Open a dropped dataset folder, guessing its layout.

        A dropped folder often holds several subsets (`set_a/images`,
        `set_b/images`, ...) or a train/val/test split. Taking only the first
        pair would quietly import a fraction of the data, so every image
        folder under the root is opened at once and every label folder is
        imported. Replaces whatever project is open (saving it first).
        """
        from ..core.importers import find_dataset_pairs
        root = Path(root)
        self.save_current()
        pairs = find_dataset_pairs(root)
        if not pairs:
            QMessageBox.warning(
                self, "열 수 없음",
                f"{root}\n\n이미지를 찾지 못했습니다.\n"
                "이미지가 든 폴더를 올려주세요.")
            return False

        # Always open the dropped root, never the images/ subfolder inside
        # it. Descending would strip the folder out of every relative path,
        # which flattens the file panel's folder view to a single node -- the
        # structure the user dropped the folder to keep.
        if not self.load_image_folder(root, name=root.name):
            return False

        sources, seen = [], set()
        for p in pairs:
            q = p["label_src"]
            if q is None:
                continue
            key = str(Path(q).resolve())
            if key not in seen:
                seen.add(key)
                sources.append(Path(q))

        n_img = len(self.project.images)
        if not sources:
            self.statusBar().showMessage(
                f"{root.name} — 이미지 {n_img}장 (라벨 폴더를 찾지 못했습니다)", 12000)
            return True
        self.statusBar().showMessage(
            f"{root.name} — 이미지 {n_img}장 · 라벨 폴더 {len(sources)}곳을 "
            "찾았습니다. 검증 후 가져오세요.", 12000)
        self.import_labels(sources=sources)
        return True

    def open_images_with_labels(self, read_only: bool = False) -> None:
        """Open an image folder and its labels in one flow.

        Doing it as two separate steps (open folder, then import) means
        the labels are only checked *after* a project already exists,
        which is the wrong order when the whole point is to look at an
        existing image+label pair.
        """
        d = QFileDialog.getExistingDirectory(
            self, "이미지 폴더 선택 (다음 화면에서 라벨 폴더를 고릅니다)")
        if not d:
            return
        if not self.load_image_folder(Path(d), read_only=read_only):
            return
        self.import_labels()          # opens pre-filled with this folder

    def load_image_folder(self, img_dir: Path, read_only: bool = False,
                          name: str = "") -> bool:
        """Label a folder of images directly, wherever on disk it lives.

        The chosen folder is never written to: project.json and labels/ go to
        projects/<name>/ under APP_DIR, so a read-only, network or shared
        image folder works. Re-opening the same folder reuses its project,
        so labelling can be resumed.

        Split from the dialog above so it can be driven by a test and by the
        command line without a file picker.
        """
        img_dir = Path(img_dir)
        n = sum(1 for f in img_dir.rglob("*")
                if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
        if n == 0:
            QMessageBox.warning(
                self, "이미지 없음",
                f"{img_dir}\n\n이 폴더에서 이미지를 찾지 못했습니다.\n"
                f"지원 형식: {', '.join(sorted(IMAGE_EXTS))}")
            return False

        # Two different folders can share a basename, so the project folder
        # carries a short digest of the full path to keep them apart.
        # Name it after the dataset, not the "images" subfolder it happens
        # to live in, or every dropped dataset ends up called images_<hash>.
        label = name or img_dir.name
        key = hashlib.sha1(str(img_dir.resolve()).encode("utf-8")).hexdigest()[:8]
        if read_only:
            # A scratch project in a temp dir: nothing is added under
            # projects/, and it is removed when the app closes. The image
            # folder is never written to either way.
            import tempfile
            root = Path(tempfile.mkdtemp(prefix="dsl_view_")) / label
            self._temp_projects.append(root.parent)
        else:
            root = resolve_under_app("projects") / f"{label}_{key}"
        try:
            if (root / "project.json").exists():
                self.load_project(root)
            else:
                project = Project.create(root, label,
                                         image_dir=str(img_dir.resolve()))
                project.scan_images()
                project.rebuild_index()
                project.save()
                self._set_project(project)
        except (OSError, ValueError) as e:
            QMessageBox.warning(self, "열기 실패", f"{img_dir}\n{e}")
            return False
        self.statusBar().showMessage(
            (f"열람 전용 — 이미지 {n}장 (종료 시 라벨은 남지 않습니다)"
             if read_only else
             f"이미지 {n}장 · 라벨 저장 위치: {root / 'labels'}"), 15000)
        return True

    def load_project(self, root: Path) -> None:
        try:
            project = Project.load(Path(root))
        except (OSError, ValueError) as e:
            QMessageBox.warning(self, "열기 실패", f"{root}\n{e}")
            return
        project.scan_images()
        self._set_project(project)

    def _set_project(self, project: Project) -> None:
        self.save_current()
        self.project = project
        self.current_rel = None
        self.annotation = None
        self.cache.clear()
        self._undo.clear()
        self._redo.clear()
        self.file_panel.set_project(project)
        self.class_panel.set_project(project)
        self.shape_panel.set_project(project)
        self.shape_panel.set_shapes([])
        self.canvas.set_image(None)
        self.settings.push_recent(str(project.root))
        self.settings.save()
        self._update_title()
        if not project.classes:
            self.statusBar().showMessage(
                "클래스를 먼저 추가하세요 (오른쪽 [클래스] 패널 → 추가)", 8000)
        if project.images:
            self.file_panel.select(project.images[0])

    def _update_title(self) -> None:
        if self.project:
            st = self.project.stats()
            self.setWindowTitle(
                f"DataSegmentLabeling — {self.project.name}  "
                f"[{st['labeled']}/{st['images']} 라벨 · 도형 {st['shapes']}]")
        else:
            self.setWindowTitle("DataSegmentLabeling — 프로젝트를 열어주세요")

    def add_images(self) -> None:
        if not self.project:
            QMessageBox.information(self, "프로젝트 없음", "먼저 프로젝트를 만들어 주세요.")
            return
        files, _ = QFileDialog.getOpenFileNames(
            self, "이미지 선택", "",
            "이미지 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)")
        if not files:
            return
        added = self.project.import_images(files, copy=True)
        self.project.save()
        self.file_panel.set_project(self.project)
        self.statusBar().showMessage(f"{len(added)}장 추가", 5000)
        self._update_title()

    def set_image_dir(self) -> None:
        if not self.project:
            return
        d = QFileDialog.getExistingDirectory(self, "이미지 폴더 선택",
                                             str(self.project.image_root))
        if not d:
            return
        self.project.image_dir = d
        self.project.scan_images()
        self.project.rebuild_index()
        self.project.save()
        self.file_panel.set_project(self.project)
        self._update_title()

    # -------------------------------------------------------------- images
    def open_image(self, rel: str) -> None:
        if rel == self.current_rel:
            return
        self.save_current()
        if not self.project:
            return
        self.current_rel = rel
        self._undo.clear()
        self._redo.clear()
        self._candidates = None
        self._sam_ready_for = None

        rgb = self.cache.load(self.project.abs_image(rel))
        if rgb is None:
            self.statusBar().showMessage(f"이미지를 읽지 못했습니다: {rel}", 6000)
            self.canvas.set_image(None)
            return

        self.annotation = self.project.load_annotation(rel)
        self.annotation.width, self.annotation.height = rgb.shape[1], rgb.shape[0]
        self.canvas.set_image(rgb)
        self._rebuild_shape_items()
        self._dirty = False
        self._trim_cache()
        self._prefetch_neighbors()
        n = len(self.annotation.shapes)
        boxes = sum(1 for s in self.annotation.shapes if s.shape_type == "rectangle")
        msg = f"{rel}  ({rgb.shape[1]}×{rgb.shape[0]})  도형 {n}개"
        if boxes:
            msg += f" · 변환 대기 박스 {boxes}개"
        self.statusBar().showMessage(msg, 4000)

    def _rebuild_shape_items(self) -> None:
        self.canvas.clear_shapes()
        if not self.annotation or not self.project:
            return
        for s in self.annotation.shapes:
            c = self.project.class_by_id(s.class_id)
            self.canvas.add_shape_item(s, c.color if c else "#888888")
        self.shape_panel.set_shapes(self.annotation.shapes)

    def _trim_cache(self) -> None:
        """Keep only the current image and its prefetch window resident."""
        if not self.project or not self.current_rel:
            return
        n = max(0, int(self.settings.prefetch_neighbors))
        try:
            i = self.project.images.index(self.current_rel)
        except ValueError:
            return
        window = self.project.images[max(0, i - n): i + n + 1]
        self.cache.keep_only([str(self.project.abs_image(r)) for r in window])
        del i, n

    def _window_rels(self) -> List[str]:
        """Current image plus its prefetch neighbours -- the only ones allowed in RAM."""
        if not self.project or not self.current_rel:
            return []
        n = max(0, int(self.settings.prefetch_neighbors))
        try:
            i = self.project.images.index(self.current_rel)
        except ValueError:
            return [self.current_rel]
        lo = max(0, i - n)
        hi = min(len(self.project.images), i + n + 1)
        return self.project.images[lo:hi]

    def _prefetch_neighbors(self) -> None:
        """Coalesced: fast browsing schedules exactly one prefetch pass."""
        if not self.project or int(self.settings.prefetch_neighbors) <= 0:
            return
        self._prefetch_timer.start(150)

    def _do_prefetch(self) -> None:
        # the window is recomputed at fire time, so stale targets are dropped
        for rel in self._window_rels():
            if rel != self.current_rel:
                self.cache.load(self.project.abs_image(rel))
        self._trim_cache()

    def _update_memory_label(self) -> None:
        thumbs = len(self.file_panel.model.cache)
        self.lbl_mem.setText(f"캐시 {self.cache.info()} · 썸네일 {thumbs}")

    # --------------------------------------------------------------- saving
    def _mark_dirty(self) -> None:
        self._dirty = True
        if self.annotation:
            self.shape_panel.set_shapes(self.annotation.shapes)

    def save_current(self) -> None:
        if not (self.project and self.annotation and self.current_rel):
            return
        if not self._dirty:
            return
        self.project.save_annotation(self.annotation)
        self.project.save_index()
        self._dirty = False
        self.file_panel.refresh_row(self.current_rel)
        self._update_title()

    def _after_shape_change(self) -> None:
        self._mark_dirty()
        if self.settings.auto_save:
            self.save_current()

    def toggle_verified(self) -> None:
        if not self.annotation:
            return
        self.annotation.verified = not self.annotation.verified
        self._dirty = True
        self.save_current()
        rel = self.current_rel
        state = "검증 완료" if self.annotation.verified else "검증 해제"
        self.statusBar().showMessage(f"{rel}: {state}", 3000)
        # finished images are released from RAM immediately (DEVELOPMENT.md 3.5)
        if self.annotation.verified and rel:
            self.cache.evict(str(self.project.abs_image(rel)))
            if self.settings.hide_completed:
                self.file_panel.drop_row(rel)
            if self.settings.auto_advance:
                nxt = self.project.next_unlabeled(rel)
                if nxt:
                    self.file_panel.select(nxt)

    # ---------------------------------------------------------------- undo
    def _snapshot(self) -> None:
        if not self.annotation:
            return
        state = [s.to_dict() for s in self.annotation.shapes]
        # Geometry edits snapshot on mouse-press, before we know whether the
        # user will actually drag. A plain selection click, or the press that
        # precedes a double-click, would otherwise pile up no-op undo steps.
        if self._undo and self._undo[-1] == state:
            return
        self._undo.append(state)
        del self._undo[:-UNDO_LIMIT]
        self._redo.clear()

    def _restore(self, snap: list) -> None:
        if self.annotation is None:
            return
        self.annotation.shapes = [Shape.from_dict(d) for d in snap]
        self._rebuild_shape_items()
        self._after_shape_change()

    def undo(self) -> None:
        if not self._undo or self.annotation is None:
            return
        self._redo.append([s.to_dict() for s in self.annotation.shapes])
        self._restore(self._undo.pop())

    def redo(self) -> None:
        if not self._redo or self.annotation is None:
            return
        self._undo.append([s.to_dict() for s in self.annotation.shapes])
        self._restore(self._redo.pop())

    # --------------------------------------------------------------- tools
    def set_tool(self, tool: str) -> None:
        self.canvas.set_tool(tool)
        self.tool_actions[tool].setChecked(True)
        self.lbl_hint.setText(TOOL_HINTS[tool])
        self._candidates = None

    def toggle_shapes(self, checked: bool) -> None:
        self.canvas.set_shapes_visible(not checked)

    def select_all(self) -> None:
        for it in self.canvas.shape_items:
            it.setSelected(True)

    def delete_selected(self) -> None:
        items = self.canvas.selected_items()
        if items:
            self.delete_shapes([i.shape_data for i in items])

    def delete_shapes(self, shapes: List[Shape]) -> None:
        if not shapes or self.annotation is None:
            return
        self._snapshot()
        uids = {s.uid for s in shapes}
        self.annotation.shapes = [s for s in self.annotation.shapes
                                  if s.uid not in uids]
        self._rebuild_shape_items()
        self._after_shape_change()
        self.statusBar().showMessage(f"{len(uids)}개 삭제", 3000)

    def assign_class(self, shapes: List[Shape]) -> None:
        """Re-label the selected shapes with the class currently highlighted.

        Both preconditions used to fail silently, which is why this reads as
        a broken button rather than a missing step (DEVELOPMENT.md 4: never
        swallow, say it in the status bar).
        """
        if not shapes:
            self.statusBar().showMessage(
                "적용할 도형이 없습니다 — 캔버스나 [도형] 표에서 먼저 선택하세요.", 6000)
            return
        if self.current_class_id is None:
            self.statusBar().showMessage(
                "현재 클래스가 없습니다 — [클래스] 패널에서 클래스를 고르세요"
                " (없으면 [추가]).", 6000)
            return
        self._snapshot()
        for s in shapes:
            s.class_id = self.current_class_id
        self._rebuild_shape_items()
        self._after_shape_change()
        c = self.project.class_by_id(self.current_class_id) if self.project else None
        self.statusBar().showMessage(
            f"도형 {len(shapes)}개 → '{c.name if c else self.current_class_id}'", 5000)

    def clear_image_labels(self) -> None:
        if self.annotation is None or not self.annotation.shapes:
            return
        if QMessageBox.question(self, "라벨 비우기",
                                "이 이미지의 모든 도형을 삭제할까요?") != QMessageBox.Yes:
            return
        self._snapshot()
        self.annotation.shapes = []
        self._rebuild_shape_items()
        self._after_shape_change()

    def _on_class_changed(self, cid: int) -> None:
        self.current_class_id = cid
        if self.project:
            c = self.project.class_by_id(cid)
            if c:
                self.statusBar().showMessage(f"현재 클래스: {c.name}", 2000)

    def _on_classes_changed(self) -> None:
        # A deleted class is already stripped from the label files on disk; the
        # image open right now is only in memory, so drop its shapes too or the
        # next save writes the deleted class straight back.
        if self.project and self.annotation:
            live = {c.id for c in self.project.classes}
            keep = [s for s in self.annotation.shapes if s.class_id in live]
            self.annotation.shapes = keep
        self._rebuild_shape_items()      # also refreshes the shape table
        self.file_panel.model.layoutChanged.emit()
        self._update_title()

    def _on_canvas_selection(self, item: Optional[object]) -> None:
        """Canvas emits a ShapeItem; the shape table speaks Shape."""
        self.shape_panel.select_shape(
            item.shape_data if item is not None else None)

    def _on_shape_row_selected(self, shape: Optional[Shape]) -> None:
        for it in self.canvas.shape_items:
            it.setSelected(shape is not None and it.shape_data.uid == shape.uid)

    # ------------------------------------------------------------- prompts
    def _current_rgb(self) -> Optional[np.ndarray]:
        if not (self.project and self.current_rel):
            return None
        return self.cache.load(self.project.abs_image(self.current_rel))

    def _ensure_encoded(self) -> bool:
        """Encode the current image if the SAM worker has not seen it yet."""
        if self.sam.backend is None and not self.sam.model_key:
            self.statusBar().showMessage(
                "먼저 [SAM] 탭에서 모델을 불러오세요.", 5000)
            return False
        key = self.current_rel or ""
        if self._sam_ready_for == key:
            return True
        rgb = self._current_rgb()
        if rgb is None:
            return False
        self.samSetImage.emit(rgb, key)
        self._sam_ready_for = key          # queued; predict runs after encoding
        return True

    def _on_prompt(self, points: list, box) -> None:
        if not points and not box:
            self.canvas.show_mask(None)
            self._candidates = None
            return
        if self.canvas.tool not in (SAM_POINT, SAM_BOX):
            return
        if not self._ensure_encoded():
            return
        h, w = (self.annotation.height, self.annotation.width) if self.annotation \
            else (0, 0)
        self._req += 1
        self.samPredict.emit(list(points), box, self._req, h, w)

    def _on_text_prompt(self, text: str) -> None:
        if not text:
            return
        if not self._ensure_encoded():
            return
        h, w = (self.annotation.height, self.annotation.width) if self.annotation \
            else (0, 0)
        self._req += 1
        self._pending_text[self._req] = True
        self.samPredictText.emit(text, self._req, h, w)

    def shapes_to_boxes(self) -> None:
        """Selected segments -> their bounding boxes.

        The polygon is stashed in meta first, so converting back restores it
        exactly instead of re-running SAM and getting a slightly different
        mask. That is what makes the round trip free rather than lossy.
        """
        items = [i for i in self.canvas.selected_items()
                 if i.shape_data.shape_type != "rectangle"]
        if not items:
            self.statusBar().showMessage(
                "박스로 바꿀 세그먼트를 먼저 선택하세요 (Ctrl+A 로 전체 선택).", 6000)
            return
        self._snapshot()
        for it in items:
            sh = it.shape_data
            x1, y1, x2, y2 = sh.bbox()
            sh.meta = dict(sh.meta or {})
            sh.meta["src_polygon"] = [[float(x), float(y)] for x, y in sh.points]
            if sh.holes:
                sh.meta["src_holes"] = [[[float(x), float(y)] for x, y in h]
                                        for h in sh.holes]
            sh.points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            sh.holes = []
            sh.shape_type = "rectangle"
            it.rebuild_path()
            it.apply_style()
        self.shape_panel.set_shapes(self.annotation.shapes)
        self._after_shape_change()
        self.statusBar().showMessage(
            f"세그먼트 {len(items)}개 → 박스 · Ctrl+Z 로 되돌리기", 5000)

    def boxes_to_shapes(self) -> None:
        """Selected boxes -> segments.

        A box that was made from a segment still carries that polygon, so it
        comes back instantly and exactly. Anything else (an imported or hand
        drawn box) has no polygon to restore and goes to SAM.
        """
        items = [i for i in self.canvas.selected_items()
                 if i.shape_data.shape_type == "rectangle"]
        if not items:
            self.statusBar().showMessage(
                "세그먼트로 바꿀 박스를 먼저 선택하세요 (Ctrl+A 로 전체 선택).", 6000)
            return
        restored, need_sam = 0, []
        self._snapshot()
        for it in items:
            sh = it.shape_data
            poly = (sh.meta or {}).get("src_polygon")
            if not poly:
                need_sam.append(it)
                continue
            sh.points = [(float(x), float(y)) for x, y in poly]
            sh.holes = [[(float(x), float(y)) for x, y in h]
                        for h in (sh.meta.get("src_holes") or [])]
            sh.shape_type = "polygon"
            sh.meta.pop("src_polygon", None)
            sh.meta.pop("src_holes", None)
            it.rebuild_path()
            it.apply_style()
            restored += 1
        if restored:
            self.shape_panel.set_shapes(self.annotation.shapes)
            self._after_shape_change()
        if need_sam:
            if not self._ensure_encoded():
                self.statusBar().showMessage(
                    f"{restored}개 복원. 나머지 {len(need_sam)}개는 SAM 이 필요합니다 "
                    "— [SAM] 탭에서 모델을 불러오세요.", 8000)
                return
            h, w = self.annotation.height, self.annotation.width
            for it in need_sam:
                x1, y1, x2, y2 = it.shape_data.bbox()
                self._req += 1
                self._pending_convert[self._req] = it
                self.samPredict.emit([], [x1, y1, x2, y2], self._req, h, w)
        msg = []
        if restored:
            msg.append(f"{restored}개 원본 복원")
        if need_sam:
            msg.append(f"{len(need_sam)}개 SAM 변환 중")
        self.statusBar().showMessage(" · ".join(msg) + " · Ctrl+Z 로 되돌리기", 5000)

    def _on_convert_request(self, item: ShapeItem) -> None:
        if item.shape_data.shape_type != "rectangle":
            self.statusBar().showMessage(
                "점선 박스만 클릭 변환할 수 있습니다. 세그먼트를 박스로 바꾸려면 "
                "도구 → 세그먼트 → 박스 (Ctrl+Shift+B).", 6000)
            return
        if not self._ensure_encoded():
            return
        h, w = self.annotation.height, self.annotation.width
        x1, y1, x2, y2 = item.shape_data.bbox()
        self._req += 1
        self._pending_convert[self._req] = item
        self.samPredict.emit([], [x1, y1, x2, y2], self._req, h, w)

    @Slot(object, object, int)
    def _on_sam_result(self, masks, scores, req_id: int) -> None:
        if masks is None or len(masks) == 0:
            self.statusBar().showMessage("마스크를 찾지 못했습니다.", 4000)
            return

        item = self._pending_convert.pop(req_id, None)
        if item is not None:
            self._apply_conversion(item, masks[0], float(scores[0]))
            return

        if self._pending_text.pop(req_id, None):
            self._add_masks_as_shapes(masks, scores, source="text")
            return

        if req_id != self._req:
            return                               # a newer prompt superseded this
        self._candidates = masks
        self._cand_scores = scores
        self._cand_index = 0
        self._show_candidate()

    def _show_candidate(self) -> None:
        if self._candidates is None or len(self._candidates) == 0:
            return
        i = self._cand_index % len(self._candidates)
        color = "#4c8dff"
        if self.project and self.current_class_id is not None:
            c = self.project.class_by_id(self.current_class_id)
            if c:
                color = c.color
        self.canvas.show_mask(self._candidates[i], color)
        self.statusBar().showMessage(
            f"후보 {i + 1}/{len(self._candidates)}  "
            f"점수 {float(self._cand_scores[i]):.3f}  —  Enter 확정 / [ ] 후보전환 / Esc 취소")

    def cycle_candidate(self, delta: int) -> None:
        if self._candidates is None:
            return
        self._cand_index = (self._cand_index + delta) % len(self._candidates)
        self._show_candidate()

    def cancel_prompt(self) -> None:
        self.canvas.clear_prompt()
        self._candidates = None
        self.statusBar().clearMessage()

    def accept_mask(self) -> None:
        if self._candidates is None or self.annotation is None:
            return
        if self.current_class_id is None:
            QMessageBox.information(self, "클래스 필요", "먼저 클래스를 선택하세요.")
            return
        mask = self._candidates[self._cand_index % len(self._candidates)]
        score = float(self._cand_scores[self._cand_index % len(self._cand_scores)])
        added = self._add_masks_as_shapes(mask[None], np.array([score]))
        self.canvas.clear_prompt()
        self._candidates = None
        if added and self.settings.auto_advance and self.settings.auto_save:
            pass

    def _on_manual_box(self, box: list) -> None:
        """A box drawn with the box-label tool becomes a rectangle shape.

        Stored as four corner points like every other shape, so exporters and
        the bbox derivation need no special case. Drawing does not clear what
        is already on the image, so boxes and segments accumulate together.
        """
        if self.annotation is None:
            self.statusBar().showMessage("먼저 이미지를 여세요.", 4000)
            return
        if self.current_class_id is None:
            QMessageBox.information(self, "클래스 필요", "먼저 클래스를 선택하세요.")
            return
        x1, y1, x2, y2 = (float(v) for v in box)
        w = self.annotation.width or 0
        h = self.annotation.height or 0
        if w and h:                       # never let a label leave the image
            x1, x2 = max(0.0, min(x1, w)), max(0.0, min(x2, w))
            y1, y2 = max(0.0, min(y1, h)), max(0.0, min(y2, h))
        if abs(x2 - x1) < 2 or abs(y2 - y1) < 2:
            return
        self._snapshot()
        shape = Shape(class_id=self.current_class_id,
                      points=[(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
                      shape_type="rectangle", source="manual")
        self.annotation.shapes.append(shape)
        c = self.project.class_by_id(self.current_class_id) if self.project else None
        self.canvas.add_shape_item(shape, c.color if c else "#888888")
        self.shape_panel.set_shapes(self.annotation.shapes)
        self._mark_dirty()
        self.statusBar().showMessage(
            f"박스 {len(self.annotation.shapes)}개 — 계속 드래그해 더 그릴 수 있습니다",
            4000)

    def _add_masks_as_shapes(self, masks, scores, source: str = "") -> int:
        if self.annotation is None or self.current_class_id is None:
            return 0
        self._snapshot()
        src = source or self.sam.model_key or "sam"
        added = 0
        for m, sc in zip(masks, scores):
            polys = mask_to_polygons(
                m, self.settings.poly_epsilon, self.settings.min_area,
                keep_largest_only=self.settings.keep_largest_only,
                smooth=self.settings.poly_smooth)
            for outer, holes in polys:
                shape = Shape(class_id=self.current_class_id, points=outer,
                              holes=holes, score=float(sc),
                              source=src if source != "text" else
                              f"{self.sam.model_key or 'sam'}:text")
                self.annotation.shapes.append(shape)
                c = self.project.class_by_id(self.current_class_id)
                self.canvas.add_shape_item(shape, c.color if c else "#888888")
                added += 1
        if added:
            self.shape_panel.set_shapes(self.annotation.shapes)
            self._after_shape_change()
            self.statusBar().showMessage(f"{added}개 도형 추가", 3000)
        else:
            if self._undo:
                self._undo.pop()
            self.statusBar().showMessage(
                "마스크가 최소 면적보다 작습니다. [SAM] 탭에서 최소 면적을 줄여보세요.", 5000)
        return added

    def _apply_conversion(self, item: ShapeItem, mask, score: float) -> None:
        polys = mask_to_polygons(mask, self.settings.poly_epsilon,
                                 self.settings.min_area, keep_largest_only=True,
                                 smooth=self.settings.poly_smooth)
        if not polys:
            self.statusBar().showMessage("이 박스에서 마스크를 얻지 못했습니다.", 4000)
            return
        self._snapshot()
        outer, holes = polys[0]
        s = item.shape_data
        s.meta = dict(s.meta or {})
        s.meta.setdefault("src_bbox", list(s.bbox()))
        s.points = outer
        s.holes = holes
        s.shape_type = "polygon"
        s.score = float(score)
        s.source = self.sam.model_key or "sam"
        item.rebuild_path()
        item.apply_style()
        self.shape_panel.set_shapes(self.annotation.shapes)
        self._after_shape_change()
        self.statusBar().showMessage(
            f"박스 → 세그먼트 변환 완료 (점수 {score:.3f}) · Ctrl+Z 로 되돌리기", 4000)

    def _on_manual_polygon(self, points: list) -> None:
        if self.annotation is None:
            return
        if self.current_class_id is None:
            QMessageBox.information(self, "클래스 필요", "먼저 클래스를 선택하세요.")
            return
        self._snapshot()
        shape = Shape(class_id=self.current_class_id,
                      points=[(float(x), float(y)) for x, y in points],
                      source="manual")
        self.annotation.shapes.append(shape)
        c = self.project.class_by_id(self.current_class_id)
        self.canvas.add_shape_item(shape, c.color if c else "#888888")
        self.shape_panel.set_shapes(self.annotation.shapes)
        self._after_shape_change()

    # -------------------------------------------------------- batch convert
    def _start_batch_convert(self, scope: str) -> None:
        if not self.project:
            return
        if self.sam.model_key == "":
            QMessageBox.information(self, "모델 필요", "먼저 SAM 모델을 불러오세요.")
            return
        if scope == "image":
            rels = [self.current_rel] if self.current_rel else []
        else:
            # the worker skips images without rectangles, so labelled is enough
            rels = self.project.labeled_images()
        if not rels:
            self.statusBar().showMessage("변환할 박스가 없습니다.", 4000)
            return
        if scope == "project":
            if QMessageBox.question(
                    self, "전체 변환",
                    f"라벨된 이미지 {len(rels)}장의 박스를 모두 세그먼트로 변환할까요?\n"
                    "시간이 오래 걸릴 수 있으며 중간에 [중지]할 수 있습니다."
                    ) != QMessageBox.Yes:
                return
        self.save_current()
        self.sam_panel.set_batch_running(True)
        self.samBatch.emit(self.project, rels, -1,
                           float(self.settings.poly_epsilon),
                           float(self.settings.min_area), True,
                           int(self.settings.poly_smooth))

    def _start_auto_segment(self, scope: str) -> None:
        """Segment everything the model can find, with no prompt.

        There are no class names in an unprompted run, so every mask goes to
        the current class and the user sorts them out afterwards. Saying that
        up front is the point of the confirmation below.
        """
        if not self.project or not self.project.images:
            self.statusBar().showMessage("먼저 이미지를 불러오세요.", 5000)
            return
        if self.current_class_id is None:
            QMessageBox.information(self, "클래스 필요", "먼저 클래스를 선택하세요.")
            return
        if scope == "image":
            if not self.current_rel:
                self.statusBar().showMessage("먼저 이미지를 여세요.", 5000)
                return
            rels = [self.current_rel]
        else:
            rels = list(self.project.images)
        c = self.project.class_by_id(self.current_class_id)
        if QMessageBox.question(
                self, "자동 분할",
                f"이미지 {len(rels)}장에서 찾을 수 있는 모든 영역을 분할합니다.\n"
                f"클래스는 알 수 없으므로 전부 "
                f"'{c.name if c else self.current_class_id}' 로 들어갑니다 — "
                "이후 도형을 골라 클래스를 바꾸세요.\n계속할까요?") != QMessageBox.Yes:
            return
        self.save_current()
        self.sam_panel.set_batch_running(True)
        self.samBatchAuto.emit(self.project, rels, int(self.current_class_id),
                               float(self.settings.poly_epsilon),
                               float(self.settings.min_area),
                               int(self.settings.poly_smooth), 200)

    def _start_batch_text(self, text: str) -> None:
        """Apply one text prompt to every image in the project.

        SAM 3 turns a phrase into every matching instance, so this is the
        fastest way to pre-label a whole set when the class has a name the
        model knows. Results are proposals: they are saved unverified.
        """
        if not self.project or not self.project.images:
            self.statusBar().showMessage("먼저 이미지를 불러오세요.", 5000)
            return
        if not text:
            self.statusBar().showMessage("찾을 문구를 입력하세요.", 5000)
            return
        if self.current_class_id is None:
            QMessageBox.information(self, "클래스 필요", "먼저 클래스를 선택하세요.")
            return
        rels = self.project.images
        c = self.project.class_by_id(self.current_class_id)
        if QMessageBox.question(
                self, "프로젝트 전체 적용",
                f"'{text}' 를 이미지 {len(rels)}장에 적용해 "
                f"'{c.name if c else self.current_class_id}' 클래스로 추가할까요?\n"
                "시간이 오래 걸릴 수 있으며 [중지] 할 수 있습니다.") != QMessageBox.Yes:
            return
        self.save_current()
        self.sam_panel.set_batch_running(True)
        self.samBatchText.emit(self.project, list(rels), text,
                               int(self.current_class_id),
                               float(self.settings.poly_epsilon),
                               float(self.settings.min_area),
                               int(self.settings.poly_smooth))

    def _on_batch_finished(self, converted: int, images: int) -> None:
        self.sam_panel.set_batch_running(False)
        self.project.save_index() if self.project else None
        self._sam_ready_for = None
        if self.current_rel:
            rel = self.current_rel
            self.current_rel = None
            self.open_image(rel)
        self.file_panel.set_project(self.project)
        self.file_panel.select(self.current_rel or "", emit=False)
        self._update_title()
        QMessageBox.information(
            self, "일괄 변환 완료",
            f"이미지 {images}장에서 박스 {converted}개를 세그먼트로 변환했습니다.")

    # ----------------------------------------------------------- data i/o
    def auto_label(self) -> None:
        """Pre-label with a trained checkpoint, then reload what changed."""
        if not self.project:
            QMessageBox.information(self, "프로젝트 없음", "먼저 프로젝트를 열어주세요.")
            return
        if not self.project.images:
            QMessageBox.information(self, "이미지 없음", "먼저 이미지를 불러오세요.")
            return
        self.save_current()
        dlg = AutoLabelDialog(self.project, self.settings, self)
        dlg.exec()
        if not dlg.result_data:
            return
        self.project.load_index()
        self.class_panel.reload()
        self.file_panel.set_project(self.project)
        if self.current_rel:
            rel = self.current_rel
            self.current_rel = None            # force a reload of the canvas
            self.open_image(rel)
        self._update_title()

    def import_labels(self, label_src: str = "", sources=None) -> None:
        if not self.project:
            QMessageBox.information(self, "프로젝트 없음", "먼저 프로젝트를 열어주세요.")
            return
        self.save_current()
        dlg = ImportLabelsDialog(self.project, self, sources=sources)
        if sources:
            dlg.path.setText(f"{len(sources)}곳: "
                             + ", ".join(Path(q).name for q in sources[:4])
                             + (" …" if len(sources) > 4 else ""))
            dlg.path.setReadOnly(True)
            dlg.check_data()
        elif label_src:
            dlg.path.setText(str(label_src))
            dlg.check_data()          # show the match report straight away
        dlg.exec()
        if dlg.result_data:
            self.project.load_index()
            self.class_panel.reload()
            self.file_panel.set_project(self.project)
            if self.current_rel:
                rel = self.current_rel
                self.current_rel = None
                self.open_image(rel)
            self._update_title()
            self.set_tool(BOX2SEG)
            self._offer_conversion(dlg.result_data)

    def _offer_conversion(self, r: dict) -> None:
        """Ask how the freshly imported boxes should become segments.

        The import itself only stores boxes; turning them into masks is
        a separate, slow step. Asking here means the user does not have
        to go find the batch buttons in the SAM panel to discover the
        automatic path even exists.
        """
        boxes = int(r.get("boxes", 0))
        if boxes <= 0:
            self.statusBar().showMessage("가져오기 완료 — 변환할 박스가 없습니다.", 8000)
            return
        box = QMessageBox(self)
        box.setWindowTitle("세그먼트로 변환")
        box.setText(f"박스 {boxes}개를 가져왔습니다. 어떻게 변환할까요?")
        box.setInformativeText(
            "· 수동: 점선 박스를 하나씩 클릭해 확인하며 변환합니다.\n"
            "· 현재 이미지: 지금 열린 이미지의 박스를 한 번에.\n"
            "· 프로젝트 전체: 모든 이미지를 백그라운드로. 중간에 중지할 수 있습니다.")
        b_manual = box.addButton("수동으로 하나씩", QMessageBox.AcceptRole)
        b_img = box.addButton("현재 이미지 자동", QMessageBox.AcceptRole)
        b_all = box.addButton("프로젝트 전체 자동", QMessageBox.AcceptRole)
        box.addButton("나중에", QMessageBox.RejectRole)
        box.exec()
        clicked = box.clickedButton()
        if clicked is b_img:
            self._start_batch_convert("image")
        elif clicked is b_all:
            self._start_batch_convert("project")
        elif clicked is b_manual:
            self.statusBar().showMessage(
                "[박스→세그먼트] 도구입니다 — 점선 박스를 클릭하면 변환됩니다.", 8000)
        else:
            self.statusBar().showMessage(
                "가져오기 완료 — 나중에 [SAM] 탭에서 일괄 변환할 수 있습니다.", 8000)

    def export_dataset(self, fmt: str = "yolo") -> None:
        if not self.project:
            return
        self.save_current()
        dlg = ExportDatasetDialog(self.project, self.settings, fmt or "yolo", self)
        dlg.exec()
        if dlg.result_data:
            self.train_panel.set_dataset(dlg.result_data.get("root", ""))
            self.right_tabs.setCurrentWidget(self.train_panel)

    def open_model_manager(self) -> None:
        dlg = ModelManagerDialog(self.settings, self)
        dlg.modelsChanged.connect(self.sam_panel.reload_models)
        dlg.exec()
        self.sam_panel.reload_models()

    # ---------------------------------------------------------- training
    def start_training(self, model_key: str, variant: str, hyper: dict,
                       dataset: str) -> None:
        if self.trainer is not None and self.trainer.running:
            QMessageBox.information(self, "실행 중", "이미 학습이 진행 중입니다.")
            return
        if not dataset or not Path(dataset).exists():
            QMessageBox.warning(self, "데이터셋 없음",
                                "먼저 데이터셋을 내보내고 경로를 지정하세요.")
            return
        classes = [c.name for c in self.project.classes] if self.project else []
        job = TrainJob(model_key=model_key, variant=variant,
                       dataset_dir=str(dataset), classes=classes,
                       device=self.train_panel.device.text().strip() or "0",
                       hyper=hyper,
                       # absolute: the trainer subprocess runs with a
                       # different cwd, so "runs" would land elsewhere
                       project_dir=str(self.settings.runs_path))
        job.fill_defaults()
        self.train_panel.set_running(True)
        self.train_panel.append_log(f"$ python -m dsl.train.cli  ({job.run_name})")
        self.trainer = TrainingProcess(
            job,
            on_line=self.trainLine.emit,
            on_metric=self.trainMetric.emit,
            on_done=self.trainDone.emit,
            python_exe=self.settings.train_python)
        try:
            self.trainer.start()
        except (OSError, RuntimeError) as e:
            self.train_panel.set_running(False)
            QMessageBox.warning(self, "학습 시작 실패", str(e))

    def _on_train_done(self, code: int) -> None:
        self.train_panel.set_running(False)
        msg = "학습 완료" if code == 0 else f"학습 종료 (코드 {code})"
        self.train_panel.append_log(f"=== {msg} ===")
        self.statusBar().showMessage(msg, 8000)

    def stop_training(self) -> None:
        if self.trainer is not None and self.trainer.running:
            self.train_panel.append_log("=== 중지 요청 ===")
            self.trainer.stop()

    def _open_runs(self) -> None:
        d = Path("runs").resolve()
        d.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(str(d))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(d)])
        else:
            subprocess.Popen(["xdg-open", str(d)])

    # ---------------------------------------------------------- settings
    def _toggle_autosave(self, on: bool) -> None:
        self.settings.auto_save = on
        self.settings.save()

    def _toggle_autoadvance(self, on: bool) -> None:
        self.settings.auto_advance = on
        self.settings.save()

    def _toggle_hide_done(self, on: bool) -> None:
        self.settings.hide_completed = on
        self.settings.save()
        self.file_panel.filter.setCurrentText(FILTER_TODO if on else "전체")

    def _set_cache_size(self) -> None:
        mb, ok = QInputDialog.getInt(
            self, "이미지 캐시", "원본 이미지 캐시 상한 (MB):",
            int(self.settings.image_cache_mb), 64, 32768, 128)
        if not ok:
            return
        self.settings.image_cache_mb = mb
        self.settings.save()
        self.cache.budget = self.settings.image_cache_bytes
        self._trim_cache()

    def _set_outline_width(self) -> None:
        w, ok = QInputDialog.getDouble(
            self, "외곽선 굵기", "화면 픽셀 단위 (줌과 무관하게 일정):",
            float(self.settings.outline_width), 0.5, 8.0, 1)
        if not ok:
            return
        self.settings.outline_width = float(w)
        self.settings.save()
        self.canvas.set_outline_width(w)

    def _set_fill_alpha(self) -> None:
        a, ok = QInputDialog.getInt(
            self, "채우기 투명도", "0 = 외곽선만, 255 = 불투명:",
            int(self.settings.fill_alpha), 0, 255, 5)
        if not ok:
            return
        self._apply_fill_alpha(int(a))

    def _toggle_outline_only(self, on: bool) -> None:
        """Fill hides the image under it; outline-only keeps both readable.

        The previous opacity is remembered so the toggle is reversible
        without the user having to recall the number they had set.
        """
        if on:
            self._prev_fill_alpha = max(10, int(self.settings.fill_alpha))
            self._apply_fill_alpha(0)
        else:
            self._apply_fill_alpha(getattr(self, "_prev_fill_alpha", 70))

    def _apply_fill_alpha(self, a: int) -> None:
        self.settings.fill_alpha = int(a)
        self.settings.save()
        self.canvas.set_fill_alpha(a)
        self.a_outline_only.setChecked(a == 0)

    def show_shortcuts(self) -> None:
        QMessageBox.information(self, "단축키", SHORTCUTS_TEXT)

    def _set_train_python(self) -> None:
        """Point training at an outside interpreter.

        Only the frozen build strictly needs this (it cannot run `-m` on
        itself and ships no torch), but pointing a source checkout at another
        environment is useful too -- e.g. keeping the GUI light while a
        CUDA-heavy env does the training.
        """
        from PySide6.QtWidgets import QFileDialog
        cur = self.settings.train_python
        path, _ = QFileDialog.getOpenFileName(
            self, "학습에 사용할 python.exe 선택",
            cur or str(Path(sys.executable).parent),
            "python (python.exe python)" if sys.platform == "win32" else "python (*)")
        if not path:
            return
        self.settings.train_python = path
        self.settings.save()
        self.statusBar().showMessage(f"학습용 파이썬: {path}", 8000)

    def show_dependencies(self) -> None:
        """What optional packages are installed, and how to add the rest.

        Nothing is installed from here on purpose: these are multi-GB
        downloads that belong in the user's own environment, not inside the
        app (DEVELOPMENT.md 2.2). The dialog hands over the exact commands.
        """
        from ..core.deps import missing, report
        box = QMessageBox(self)
        box.setWindowTitle("선택 패키지 설치 상태")
        n = len(missing())
        box.setText("모두 설치되어 있습니다." if not n
                    else f"설치되지 않은 선택 패키지 {n}개가 있습니다.\n"
                         "아래 명령을 앱 바깥의 터미널에서 실행하세요.")
        box.setDetailedText(report())
        box.exec()

    def show_hardware(self) -> None:
        hw = probe_hardware(self.settings.weights_path)
        text = (hw.summary() +
                f"\ntorch: {hw.torch_version or '미설치'}"
                f"\n권장 장치: {hw.device}")
        if hw.torch_error:
            text += f"\n\ntorch 오류:\n  {hw.torch_error}"
        # Only says anything when there is a GPU going unused (DEVELOPMENT.md 2.2);
        # a CPU-only machine is not misconfigured and is told nothing.
        guide = cuda_guidance(hw)
        if guide:
            text += "\n\n" + guide
        QMessageBox.information(self, "이 PC 정보", text)

    # ------------------------------------------------------------- closing
    def closeEvent(self, event):
        self.save_current()
        if self.trainer is not None and self.trainer.running:
            if QMessageBox.question(
                    self, "학습 진행 중",
                    "학습이 실행 중입니다. 중지하고 종료할까요?") != QMessageBox.Yes:
                event.ignore()
                return
            self.trainer.stop()
        if self.project:
            self.project.save()
        self.settings.save()
        # Read-only sessions leave nothing behind; their labels only ever
        # lived in a temp folder.
        for d in self._temp_projects:
            shutil.rmtree(d, ignore_errors=True)
        self._temp_projects.clear()
        self.samUnload.emit()
        self.sam_thread.quit()
        self.sam_thread.wait(4000)
        super().closeEvent(event)


SHORTCUTS_TEXT = """[도구]
Ctrl+1 선택/편집    Ctrl+2 SAM 클릭    Ctrl+3 SAM 박스
Ctrl+4 수동 폴리곤  Ctrl+5 박스→세그먼트  Ctrl+6 박스 라벨

[SAM 클릭]
좌클릭 포함(+)  ·  우클릭 제외(−)  ·  [ ] 후보 전환
Enter 확정      ·  Esc 취소

[세그먼트 ↔ 박스 변환]  도형을 선택한 뒤
Ctrl+Shift+B 세그먼트→박스   Ctrl+Shift+G 박스→세그먼트

[편집]
1–9 클래스 선택   Del 삭제   Ctrl+A 모두 선택
Ctrl+Z 취소   Ctrl+Shift+Z 재실행
Alt+클릭 정점 추가   정점 더블클릭 삭제   더블클릭 정점 편집 모드

[탐색/보기]
A 이전 이미지   D 다음 이미지   Ctrl+D 검증 완료 표시
휠 확대/축소    Ctrl+= 확대   Ctrl+- 축소   Ctrl+0 화면 맞춤
Space+드래그 또는 휠클릭 이동
H 라벨 숨기기   Ctrl+Shift+F 외곽선만 보기

[파일]
Ctrl+N 새 프로젝트  Ctrl+O 프로젝트 열기  Ctrl+Shift+O 이미지 폴더 열기
Ctrl+S 저장  Ctrl+I 라벨 가져오기  Ctrl+E 데이터셋 내보내기
Ctrl+M 모델 관리자  Ctrl+Q 종료  F1 이 도움말
"""
