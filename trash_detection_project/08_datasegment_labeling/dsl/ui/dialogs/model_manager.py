"""SAM checkpoint manager: probe the PC, recommend, download, remove."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QDialog, QHBoxLayout,
                               QHeaderView, QInputDialog, QLabel, QMessageBox,
                               QProgressBar, QPushButton, QTableWidget,
                               QTableWidgetItem, QVBoxLayout)

from ...sam import catalog
from ...sam.downloader import (DownloadError, install, missing_packages,
                               probe_hardware, recommend, uninstall)
from ..theme import ERR, FG_DIM, OK, WARN


class _DownloadWorker(QObject):
    progress = Signal(int, int, float)
    finished = Signal(str, str)          # key, error ("" = ok)

    def __init__(self, key: str, weights_dir: str):
        super().__init__()
        self.key = key
        self.weights_dir = weights_dir
        self._cancel = False

    def cancel(self):
        self._cancel = True

    @Slot()
    def run(self):
        try:
            install(self.key, self.weights_dir,
                    progress=lambda d, t, s: self.progress.emit(d, t, s),
                    cancel=lambda: self._cancel)
            self.finished.emit(self.key, "")
        except DownloadError as e:
            self.finished.emit(self.key, str(e))
        except Exception as e:
            self.finished.emit(self.key, f"예상치 못한 오류: {e}")


class ModelManagerDialog(QDialog):
    modelsChanged = Signal()

    COLS = ["모델", "계열", "크기", "최소 VRAM", "상태", "비고"]

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SAM 모델 관리자")
        # Fit the screen rather than a fixed 920x560: on a small display the
        # buttons along the bottom would otherwise sit off-screen, and a
        # QDialog offers no way to scroll itself back into view.
        _scr = self.screen() or QApplication.primaryScreen()
        if _scr is not None:
            _av = _scr.availableGeometry()
            self.resize(min(920, int(_av.width() * 0.9)),
                        min(560, int(_av.height() * 0.9)))
        else:
            self.resize(920, 560)
        self.settings = settings
        self.weights_dir = str(settings.weights_path)
        self.hw = probe_hardware(self.weights_dir)
        self.recs = {r.spec.key: r for r in recommend(self.hw, allow_experimental=True)}
        self._thread: Optional[QThread] = None
        self._worker: Optional[_DownloadWorker] = None

        self.hw_label = QLabel("[이 PC] " + self.hw.summary())
        self.hw_label.setStyleSheet(f"color: {FG_DIM};")
        rec_txt = " · ".join(f"{i + 1}. {r.spec.key} ({r.precision})"
                             for i, r in enumerate(list(self.recs.values())[:3]))
        self.rec_label = QLabel("추천: " + (rec_txt or "없음"))
        self.rec_label.setStyleSheet(f"color: {OK};")

        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.Stretch)
        for c in range(1, len(self.COLS) - 1):
            h.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(len(self.COLS) - 1, QHeaderView.Stretch)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.status = QLabel("")
        self.status.setWordWrap(True)

        self.btn_dl = QPushButton("다운로드")
        self.btn_dl.setObjectName("primary")
        self.btn_rec = QPushButton("추천 모델 받기")
        self.btn_del = QPushButton("삭제")
        self.btn_del.setObjectName("danger")
        self.btn_edit = QPushButton("URL/저장소 수정")
        self.btn_cancel = QPushButton("중지")
        self.btn_cancel.setEnabled(False)
        self.btn_close = QPushButton("닫기")

        row = QHBoxLayout()
        row.addWidget(self.btn_dl)
        row.addWidget(self.btn_rec)
        row.addWidget(self.btn_del)
        row.addWidget(self.btn_edit)
        row.addWidget(self.btn_cancel)
        row.addStretch(1)
        row.addWidget(self.btn_close)

        lay = QVBoxLayout(self)
        lay.addWidget(self.hw_label)
        lay.addWidget(self.rec_label)
        lay.addWidget(self.table, 1)
        lay.addWidget(self.progress)
        lay.addWidget(self.status)
        lay.addLayout(row)

        self.btn_dl.clicked.connect(lambda: self.download(self.selected_key()))
        self.btn_rec.clicked.connect(self.download_recommended)
        self.btn_del.clicked.connect(self.remove_selected)
        self.btn_edit.clicked.connect(self.edit_source)
        self.btn_cancel.clicked.connect(self.cancel_download)
        self.btn_close.clicked.connect(self.accept)
        self.table.itemDoubleClicked.connect(
            lambda _i: self.download(self.selected_key()))

        self.reload()

    # ---- table ------------------------------------------------------------
    def reload(self) -> None:
        catalog.apply_overrides(Path(self.weights_dir))
        keep = self.selected_key()
        specs = [m for fam in catalog.FAMILY_ORDER for m in catalog.by_family(fam)]
        self.table.setRowCount(len(specs))
        self._keys = []
        for r, m in enumerate(specs):
            self._keys.append(m.key)
            installed = m.is_installed(Path(self.weights_dir))
            rec = self.recs.get(m.key)
            title = m.display + ("  ★추천" if rec else "")
            note = m.notes or ""
            if rec:
                note = rec.reason + ("  |  " + note if note else "")
            miss = missing_packages(m)
            if miss:
                note = f"uv pip install {' '.join(miss)}  |  " + note
            fits = (not self.hw.has_cuda) or m.min_vram_gb <= self.hw.vram_gb + 0.5
            cells = [title, catalog.FAMILY_LABELS.get(m.family, m.family),
                     f"{m.size_mb:,}MB",
                     f"{m.min_vram_gb:.1f}GB" if m.min_vram_gb else "-",
                     "설치됨" if installed else "미설치", note]
            for c, text in enumerate(cells):
                it = QTableWidgetItem(text)
                if c == 4:
                    it.setForeground(Qt.GlobalColor.green if installed
                                     else Qt.GlobalColor.gray)
                if not fits:
                    it.setToolTip("이 PC의 VRAM보다 큰 모델입니다. CPU로 동작하거나 실패할 수 있습니다.")
                self.table.setItem(r, c, it)
        if keep and keep in self._keys:
            self.table.selectRow(self._keys.index(keep))
        elif self._keys:
            self.table.selectRow(0)

    def selected_key(self) -> Optional[str]:
        r = self.table.currentRow()
        if r < 0 or r >= len(getattr(self, "_keys", [])):
            return None
        return self._keys[r]

    # ---- actions ----------------------------------------------------------
    def download_recommended(self) -> None:
        if not self.recs:
            QMessageBox.information(self, "추천 없음", "추천 가능한 모델이 없습니다.")
            return
        key = next(iter(self.recs))
        self.download(key)

    def download(self, key: Optional[str]) -> None:
        if not key or self._thread is not None:
            return
        spec = catalog.get(key)
        if spec is None:
            return
        if spec.is_installed(Path(self.weights_dir)):
            self.status.setText(f"{spec.display} 은 이미 설치되어 있습니다.")
            return
        miss = missing_packages(spec)
        if miss:
            ans = QMessageBox.question(
                self, "패키지 필요",
                f"{spec.display} 을 사용하려면 먼저 설치가 필요합니다:\n\n"
                f"    uv pip install {' '.join(miss)}\n\n"
                "체크포인트만 먼저 내려받을까요?")
            if ans != QMessageBox.Yes:
                return

        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.btn_dl.setEnabled(False)
        self.btn_rec.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.status.setText(f"{spec.display} 다운로드 중…")
        self.status.setStyleSheet(f"color: {FG_DIM};")

        self._thread = QThread(self)
        self._worker = _DownloadWorker(key, self.weights_dir)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._thread.start()

    def cancel_download(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.status.setText("중지 요청됨…")

    def _on_progress(self, done: int, total: int, speed: float) -> None:
        if total:
            self.progress.setMaximum(1000)
            self.progress.setValue(int(done / total * 1000))
            self.progress.setFormat(
                f"{done / 1048576:.0f} / {total / 1048576:.0f} MB  "
                f"({speed / 1048576:.1f} MB/s)")
        else:
            self.progress.setMaximum(0)

    def _on_finished(self, key: str, error: str) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(3000)
            self._thread = None
            self._worker = None
        self.progress.setVisible(False)
        self.btn_dl.setEnabled(True)
        self.btn_rec.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        spec = catalog.get(key)
        name = spec.display if spec else key
        if error:
            self.status.setText(f"{name}: {error}")
            self.status.setStyleSheet(f"color: {ERR};")
        else:
            self.status.setText(f"{name} 다운로드 완료.")
            self.status.setStyleSheet(f"color: {OK};")
            self.modelsChanged.emit()
        self.reload()

    def remove_selected(self) -> None:
        key = self.selected_key()
        spec = catalog.get(key or "")
        if not spec:
            return
        if not spec.is_installed(Path(self.weights_dir)):
            return
        if QMessageBox.question(
                self, "삭제", f"{spec.display} 체크포인트를 삭제할까요?\n"
                f"{spec.local_path(Path(self.weights_dir))}") != QMessageBox.Yes:
            return
        uninstall(key, self.weights_dir)
        self.status.setText(f"{spec.display} 삭제됨.")
        self.modelsChanged.emit()
        self.reload()

    def edit_source(self) -> None:
        """Override a broken URL / HF repo id without touching the source."""
        import json
        key = self.selected_key()
        spec = catalog.get(key or "")
        if not spec:
            return
        cur = spec.url or spec.repo_id
        text, ok = QInputDialog.getText(
            self, "다운로드 경로 수정",
            f"{spec.display}\nURL 또는 HuggingFace 저장소 ID:", text=cur)
        text = text.strip()
        if not ok or not text or text == cur:
            return
        path = catalog.user_overrides_path(Path(self.weights_dir))
        path.parent.mkdir(parents=True, exist_ok=True)
        data: Dict = {}
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                data = {}
        field = "url" if text.startswith("http") else "repo_id"
        other = "repo_id" if field == "url" else "url"
        data.setdefault(key, {})[field] = text
        data[key][other] = ""
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        self.status.setText(f"저장됨: {path}")
        self.reload()

    def closeEvent(self, event):
        if self._thread is not None:
            self.cancel_download()
            self._thread.quit()
            self._thread.wait(3000)
        super().closeEvent(event)
