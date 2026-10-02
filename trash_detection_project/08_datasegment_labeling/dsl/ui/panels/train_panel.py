"""Dataset export + model training, with a bounded live log."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QPlainTextEdit, QProgressBar, QPushButton,
                               QSpinBox, QVBoxLayout, QWidget)

from ...core.exporters import EXPORT_LABELS
from ...train import registry
from ...train.registry import TrainerSpec
from ..theme import ERR, FG_DIM, OK, WARN


class TrainPanel(QWidget):
    exportRequested = Signal(str)                    # format key
    trainRequested = Signal(str, str, dict, str)     # model_key, variant, hyper, dataset
    stopRequested = Signal()
    openRunDirRequested = Signal()

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._max_lines = int(settings.log_max_lines)

        # ---- dataset -------------------------------------------------------
        self.fmt = QComboBox()
        for k, v in EXPORT_LABELS.items():
            self.fmt.addItem(v, k)
        self.btn_export = QPushButton("데이터셋 내보내기…")
        self.dataset = QLineEdit()
        self.dataset.setPlaceholderText("내보낸 데이터셋 경로")
        g_ds = QGroupBox("1. 데이터셋")
        v1 = QVBoxLayout(g_ds)
        v1.setSpacing(4)
        v1.addWidget(self.fmt)
        v1.addWidget(self.btn_export)
        v1.addWidget(self.dataset)

        # ---- model ---------------------------------------------------------
        self.model = QComboBox()
        for s in registry.all_specs():
            self.model.addItem(f"{s.display}", s.key)
        self.model.setCurrentIndex(max(0, self.model.findData("yolo11-seg")))
        self.variant = QComboBox()
        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.note.setStyleSheet(f"color: {FG_DIM}; font-size: 11px;")

        self.epochs = QSpinBox(); self.epochs.setRange(1, 10000)
        self.imgsz = QSpinBox(); self.imgsz.setRange(64, 4096); self.imgsz.setSingleStep(32)
        self.batch = QSpinBox(); self.batch.setRange(1, 256)
        self.lr = QDoubleSpinBox(); self.lr.setDecimals(6); self.lr.setRange(1e-6, 1.0)
        self.lr.setSingleStep(0.0001)
        self.workers = QSpinBox(); self.workers.setRange(0, 32)
        self.device = QLineEdit("0")
        self.device.setToolTip("GPU 인덱스(0), 다중 GPU(0,1) 또는 cpu")
        self.amp = QCheckBox("AMP (혼합정밀)")
        self.amp.setChecked(True)

        g_model = QGroupBox("2. 모델 / 하이퍼파라미터")
        f2 = QFormLayout(g_model)
        f2.setSpacing(4)
        f2.addRow("모델", self.model)
        f2.addRow("크기", self.variant)
        f2.addRow("epochs", self.epochs)
        f2.addRow("imgsz", self.imgsz)
        f2.addRow("batch", self.batch)
        f2.addRow("lr0", self.lr)
        f2.addRow("workers", self.workers)
        f2.addRow("device", self.device)
        f2.addRow(self.amp)
        f2.addRow(self.note)

        # ---- run -----------------------------------------------------------
        self.btn_train = QPushButton("학습 시작")
        self.btn_train.setObjectName("primary")
        self.btn_stop = QPushButton("중지")
        self.btn_stop.setEnabled(False)
        self.btn_open = QPushButton("결과 폴더")
        self.progress = QProgressBar()
        self.progress.setFormat("%v / %m epoch")
        self.metrics = QLabel("")
        self.metrics.setStyleSheet(f"color: {FG_DIM}; font-size: 11px;")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(self._max_lines)
        self.log.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")

        row = QHBoxLayout()
        row.addWidget(self.btn_train, 2)
        row.addWidget(self.btn_stop, 1)
        row.addWidget(self.btn_open, 1)

        g_run = QGroupBox("3. 학습")
        v3 = QVBoxLayout(g_run)
        v3.setSpacing(4)
        v3.addLayout(row)
        v3.addWidget(self.progress)
        v3.addWidget(self.metrics)
        v3.addWidget(self.log, 1)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)
        lay.addWidget(g_ds)
        lay.addWidget(g_model)
        lay.addWidget(g_run, 1)

        self.model.currentIndexChanged.connect(self._on_model)
        self.btn_export.clicked.connect(
            lambda: self.exportRequested.emit(self.fmt.currentData()))
        self.btn_train.clicked.connect(self._emit_train)
        self.btn_stop.clicked.connect(self.stopRequested.emit)
        self.btn_open.clicked.connect(self.openRunDirRequested.emit)
        self._on_model()

    # ---- model selection --------------------------------------------------
    def current_spec(self) -> Optional[TrainerSpec]:
        return registry.get(self.model.currentData())

    def _on_model(self) -> None:
        spec = self.current_spec()
        if spec is None:
            return
        self.variant.clear()
        self.variant.addItems(spec.variants)
        self.variant.setCurrentText(spec.default_variant)
        d = spec.defaults
        self.epochs.setValue(int(d.get("epochs", 100)))
        self.imgsz.setValue(int(d.get("imgsz", 640)))
        self.batch.setValue(int(d.get("batch", 8)))
        self.lr.setValue(float(d.get("lr0", 0.01)))
        self.workers.setValue(int(d.get("workers", 2)))
        self.amp.setChecked(bool(d.get("amp", True)))

        want = spec.dataset_format
        i = self.fmt.findData(want)
        if i >= 0:
            self.fmt.setCurrentIndex(i)

        missing = registry.missing_packages(spec)
        parts = [spec.notes] if spec.notes else []
        parts.append(f"필요 데이터셋 포맷: {want}")
        if missing:
            parts.append("설치 필요: uv pip install " + " ".join(missing))
        self.note.setText("  ·  ".join(parts))
        self.note.setStyleSheet(
            f"color: {WARN if missing else FG_DIM}; font-size: 11px;")

    def _emit_train(self) -> None:
        spec = self.current_spec()
        if spec is None:
            return
        hyper = {"epochs": self.epochs.value(), "imgsz": self.imgsz.value(),
                 "batch": self.batch.value(), "lr0": self.lr.value(),
                 "workers": self.workers.value(), "amp": self.amp.isChecked()}
        self.trainRequested.emit(spec.key, self.variant.currentText(), hyper,
                                 self.dataset.text().strip())

    # ---- feedback ---------------------------------------------------------
    def set_dataset(self, path: str) -> None:
        self.dataset.setText(str(path))

    def suggest_batch(self, vram_gb: float) -> None:
        spec = self.current_spec()
        if spec:
            self.batch.setValue(registry.recommend_batch(
                spec, vram_gb, self.imgsz.value()))

    def append_log(self, line: str) -> None:
        self.log.appendPlainText(line)
        self.log.moveCursor(QTextCursor.End)

    def set_running(self, running: bool) -> None:
        self.btn_train.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        self.btn_export.setEnabled(not running)
        if running:
            self.progress.setValue(0)
            self.metrics.setText("준비 중…")

    def update_metrics(self, rec: dict) -> None:
        ep = int(rec.get("epoch", 0))
        total = int(rec.get("epochs", 0)) or 1
        self.progress.setMaximum(total)
        self.progress.setValue(ep)
        bits = []
        for k, v in (rec.get("loss") or {}).items():
            if v is not None:
                bits.append(f"{k} {v:.4f}")
        for k, v in (rec.get("metrics") or {}).items():
            try:
                bits.append(f"{k} {float(v):.4f}")
            except (TypeError, ValueError):
                continue
        if rec.get("time"):
            bits.append(f"{rec['time']}s/epoch")
        self.metrics.setText(f"epoch {ep}/{total}   " + "   ".join(bits[:8]))
        self.metrics.setStyleSheet(f"color: {OK}; font-size: 11px;")
