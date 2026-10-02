"""SAM control panel: pick a checkpoint, load it, prompt with text, batch-convert."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QProgressBar, QPushButton, QSpinBox, QVBoxLayout,
                               QWidget)

from ...sam import catalog
from ...sam.downloader import probe_hardware
from ..theme import ERR, FG_DIM, OK, WARN


class SamPanel(QWidget):
    loadRequested = Signal(str, str, str)       # key, device, precision
    unloadRequested = Signal()
    textPromptRequested = Signal(str)
    batchConvertRequested = Signal(str)          # "image" | "project"
    batchTextRequested = Signal(str)             # the prompt text
    autoSegmentRequested = Signal(str)           # "image" | "project"
    cancelBatchRequested = Signal()
    openManagerRequested = Signal()
    settingChanged = Signal()

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.hw = probe_hardware(settings.weights_path)

        # ---- model picker --------------------------------------------------
        self.combo = QComboBox()
        self.btn_manager = QPushButton("모델 관리자")
        self.btn_manager.setToolTip("체크포인트 다운로드 / 삭제 / 추천")

        self.device = QComboBox()
        self.device.addItems(["auto", "cuda", "cpu", "mps"])
        self.device.setCurrentText(settings.device)
        self.precision = QComboBox()
        self.precision.addItems(["fp16", "fp32", "bf16", "int8"])
        self.precision.setCurrentText(settings.sam_precision)

        self.btn_load = QPushButton("모델 불러오기")
        self.btn_load.setObjectName("primary")
        self.btn_unload = QPushButton("해제")
        self.status = QLabel("모델 미로딩")
        self.status.setWordWrap(True)
        self.status.setStyleSheet(f"color: {FG_DIM};")

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(4)
        form.addRow("모델", self.combo)
        row_dev = QHBoxLayout()
        row_dev.addWidget(self.device, 1)
        row_dev.addWidget(self.precision, 1)
        form.addRow("장치/정밀도", row_dev)
        row_btn = QHBoxLayout()
        row_btn.addWidget(self.btn_load, 2)
        row_btn.addWidget(self.btn_unload, 1)
        row_btn.addWidget(self.btn_manager, 1)

        g_model = QGroupBox("SAM 모델")
        vm = QVBoxLayout(g_model)
        vm.setSpacing(4)
        vm.addLayout(form)
        vm.addLayout(row_btn)
        vm.addWidget(self.status)

        # ---- mask options --------------------------------------------------
        self.epsilon = QDoubleSpinBox()
        self.epsilon.setRange(0.0, 0.05)
        self.epsilon.setSingleStep(0.001)
        self.epsilon.setDecimals(4)
        self.epsilon.setValue(settings.poly_epsilon)
        self.epsilon.setToolTip("0에 가까울수록 윤곽이 촘촘합니다(점 개수 증가).")
        self.smooth = QSpinBox()
        self.smooth.setRange(0, 20)
        self.smooth.setValue(int(settings.poly_smooth))
        self.smooth.setToolTip(
            "마스크 경계의 1픽셀 계단을 둥글게 폅니다. 0이면 계단 그대로.\n"
            "2~4 정도면 형태를 바꾸지 않고 외곽선이 매끄러워집니다.")
        self.min_area = QSpinBox()
        self.min_area.setRange(0, 100000)
        self.min_area.setValue(int(settings.min_area))
        self.keep_largest = QCheckBox("가장 큰 덩어리만 사용")
        self.keep_largest.setChecked(settings.keep_largest_only)
        self.multimask = QCheckBox("여러 후보 생성 ( [ ] 키로 전환 )")
        self.multimask.setChecked(True)

        g_mask = QGroupBox("마스크 → 폴리곤")
        fm = QFormLayout(g_mask)
        fm.setSpacing(4)
        fm.addRow("외곽선 부드럽게", self.smooth)
        fm.addRow("단순화(ε)", self.epsilon)
        fm.addRow("최소 면적(px)", self.min_area)
        fm.addRow(self.keep_largest)
        fm.addRow(self.multimask)

        # ---- text prompt (SAM 3) ------------------------------------------
        self.text = QLineEdit()
        self.text.setPlaceholderText("예: car, person, scratch …")
        self.btn_text = QPushButton("텍스트로 찾기")
        self.btn_text_all = QPushButton("프로젝트 전체에 적용")
        self.btn_text_all.setToolTip(
            "같은 문구를 프로젝트의 모든 이미지에 적용합니다.\n"
            "결과는 검수 대상으로 저장됩니다.")
        self.text_hint = QLabel("SAM 3 계열 모델에서만 동작합니다.")
        self.text_hint.setStyleSheet(f"color: {FG_DIM}; font-size: 11px;")
        g_text = QGroupBox("텍스트 프롬프트")
        vt = QVBoxLayout(g_text)
        vt.setSpacing(4)
        vt.addWidget(self.text)
        vt.addWidget(self.btn_text)
        vt.addWidget(self.btn_text_all)
        vt.addWidget(self.text_hint)

        # ---- box -> segment -----------------------------------------------
        self.btn_auto_img = QPushButton("이 이미지 전부 자동 분할")
        self.btn_auto_img.setToolTip(
            "프롬프트 없이 찾을 수 있는 모든 것을 분할합니다.\n"
            "클래스는 알 수 없으므로 현재 클래스로 넣습니다 — 후보 제안에 가깝습니다.")
        self.btn_auto_all = QPushButton("프로젝트 전체 자동 분할")
        self.btn_conv_img = QPushButton("현재 이미지 박스 전체 변환")
        self.btn_conv_all = QPushButton("프로젝트 전체 변환")
        self.btn_cancel = QPushButton("중지")
        self.btn_cancel.setEnabled(False)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.batch_status = QLabel("")
        self.batch_status.setStyleSheet(f"color: {FG_DIM}; font-size: 11px;")

        g_conv = QGroupBox("박스 → 세그먼트 변환")
        vc = QVBoxLayout(g_conv)
        vc.setSpacing(4)
        vc.addWidget(QLabel("가져온 점선 박스를 SAM 마스크로 승격합니다."))
        vc.addWidget(self.btn_auto_img)
        vc.addWidget(self.btn_auto_all)
        vc.addWidget(self.btn_conv_img)
        rowc = QHBoxLayout()
        rowc.addWidget(self.btn_conv_all, 2)
        rowc.addWidget(self.btn_cancel, 1)
        vc.addLayout(rowc)
        vc.addWidget(self.progress)
        vc.addWidget(self.batch_status)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)
        lay.addWidget(g_model)
        lay.addWidget(g_mask)
        lay.addWidget(g_text)
        lay.addWidget(g_conv)
        lay.addStretch(1)

        self.reload_models()
        self.btn_load.clicked.connect(self._emit_load)
        self.btn_unload.clicked.connect(self.unloadRequested.emit)
        self.btn_manager.clicked.connect(self.openManagerRequested.emit)
        self.btn_text.clicked.connect(
            lambda: self.textPromptRequested.emit(self.text.text().strip()))
        self.btn_text_all.clicked.connect(
            lambda: self.batchTextRequested.emit(self.text.text().strip()))
        self.text.returnPressed.connect(
            lambda: self.textPromptRequested.emit(self.text.text().strip()))
        self.btn_auto_img.clicked.connect(
            lambda: self.autoSegmentRequested.emit("image"))
        self.btn_auto_all.clicked.connect(
            lambda: self.autoSegmentRequested.emit("project"))
        self.btn_conv_img.clicked.connect(
            lambda: self.batchConvertRequested.emit("image"))
        self.btn_conv_all.clicked.connect(
            lambda: self.batchConvertRequested.emit("project"))
        self.btn_cancel.clicked.connect(self.cancelBatchRequested.emit)
        for w in (self.epsilon, self.min_area, self.smooth):
            w.valueChanged.connect(lambda _v: self._push_settings())
        for w in (self.keep_largest, self.multimask):
            w.toggled.connect(lambda _v: self._push_settings())
        self.combo.currentIndexChanged.connect(self._sync_precision)

    # ---- model list -------------------------------------------------------
    def reload_models(self) -> None:
        wd = self.settings.weights_path   # APP_DIR-based, never the CWD
        catalog.apply_overrides(wd)
        cur = self.current_key()
        self.combo.blockSignals(True)
        self.combo.clear()
        for fam in catalog.FAMILY_ORDER:
            for m in catalog.by_family(fam):
                installed = m.is_installed(wd)
                mark = "✔ " if installed else "⬇ "
                self.combo.addItem(f"{mark}{m.display}", m.key)
                idx = self.combo.count() - 1
                if not installed:
                    self.combo.setItemData(idx, "다운로드가 필요합니다",
                                           Qt.ToolTipRole)
        self.combo.blockSignals(False)
        target = cur or self.settings.sam_backend
        i = self.combo.findData(target)
        if i < 0:
            installed = catalog.installed_models(wd)
            i = self.combo.findData(installed[0].key) if installed else 0
        self.combo.setCurrentIndex(max(0, i))
        self._sync_precision()

    def current_key(self) -> Optional[str]:
        return self.combo.currentData()

    def _sync_precision(self) -> None:
        spec = catalog.get(self.current_key() or "")
        if not spec:
            return
        cur = self.precision.currentText()
        self.precision.blockSignals(True)
        self.precision.clear()
        self.precision.addItems(spec.quant_modes)
        self.precision.setCurrentText(cur if cur in spec.quant_modes
                                      else spec.quant_modes[0])
        self.precision.blockSignals(False)
        self.text.setEnabled(spec.text_prompt)
        self.btn_text.setEnabled(spec.text_prompt)
        self.btn_text_all.setEnabled(spec.text_prompt)
        note = spec.notes or ""
        installed = spec.is_installed(self.settings.weights_path)
        head = "" if installed else "⬇ 아직 다운로드되지 않았습니다. "
        self.status.setText(head + note)
        self.status.setStyleSheet(f"color: {FG_DIM if installed else WARN};")

    def _emit_load(self) -> None:
        key = self.current_key()
        if not key:
            return
        self.loadRequested.emit(key, self.device.currentText(),
                                self.precision.currentText())

    def _push_settings(self) -> None:
        self.settings.poly_epsilon = float(self.epsilon.value())
        self.settings.poly_smooth = int(self.smooth.value())
        self.settings.min_area = float(self.min_area.value())
        self.settings.keep_largest_only = self.keep_largest.isChecked()
        self.settingChanged.emit()

    # ---- status feedback --------------------------------------------------
    def set_loaded(self, text: str) -> None:
        self.status.setText(f"● {text}")
        self.status.setStyleSheet(f"color: {OK};")

    def set_error(self, text: str) -> None:
        self.status.setText(text)
        self.status.setStyleSheet(f"color: {ERR};")

    def set_batch_running(self, running: bool) -> None:
        self.btn_conv_img.setEnabled(not running)
        self.btn_conv_all.setEnabled(not running)
        self.btn_cancel.setEnabled(running)
        self.progress.setVisible(running)
        if not running:
            self.progress.setValue(0)

    def set_batch_progress(self, done: int, total: int, msg: str) -> None:
        self.progress.setMaximum(max(1, total))
        self.progress.setValue(done)
        self.batch_status.setText(f"{done}/{total}  {msg}")
