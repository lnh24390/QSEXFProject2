"""Import existing labels / export training datasets (both run off-thread)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

import cv2

from PySide6.QtCore import QObject, QRectF, QThread, Qt, Signal, Slot
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog,
                               QDoubleSpinBox,
                               QFileDialog, QFormLayout, QHBoxLayout, QLabel,
                               QLineEdit, QMessageBox, QProgressBar,
                               QPushButton, QScrollArea, QSpinBox,
                               QVBoxLayout, QWidget)

from ...core.exporters import EXPORT_LABELS, export_dataset, folder_split
from ...core.image_cache import imread_unicode
from ...core.settings import resolve_device
from ...train.predictor import Predictor, find_checkpoints
from ...core.importers import (FORMAT_LABELS, _ensure_class, analyze_import,
                               analyze_many,
                               describe_analysis, detect_format,
                               import_labels, stem_collisions)
from ..theme import ERR, FG_DIM, OK


class _Task(QObject):
    progress = Signal(int, int, str)
    finished = Signal(object, str)          # result dict, error

    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn

    @Slot()
    def run(self):
        try:
            res = self.fn(lambda d, t, m="": self.progress.emit(d, t, str(m)))
            self.finished.emit(res, "")
        except Exception as e:
            self.finished.emit(None, str(e))


class _TaskDialog(QDialog):
    """Shared plumbing: a form on top, progress + result at the bottom."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(560)
        self.result_data: Optional[Dict] = None
        self._max_h = 720
        self._thread: Optional[QThread] = None
        self._task: Optional[_Task] = None

        self.form = QFormLayout()
        self.form.setSpacing(6)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet(f"color: {FG_DIM};")
        self.btn_run = QPushButton("실행")
        self.btn_run.setObjectName("primary")
        self.btn_close = QPushButton("닫기")

        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self.btn_run)
        row.addWidget(self.btn_close)

        # The form grows with whatever a subclass puts in it (a match
        # report, a preview image), so it scrolls instead of pushing the
        # buttons off the bottom of the screen.
        inner = QWidget()
        inner.setLayout(self.form)
        scroll = QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)

        lay = QVBoxLayout(self)
        lay.addWidget(scroll, 1)
        lay.addWidget(self.progress)
        lay.addWidget(self.status)
        lay.addLayout(row)
        self._scroll = scroll

        self.btn_close.clicked.connect(self.reject)
        self.btn_run.clicked.connect(self.run_task)
        scr = self.screen() or QApplication.primaryScreen()
        if scr is not None:
            self._max_h = int(scr.availableGeometry().height() * 0.85)
        self.setMaximumHeight(self._max_h)
        self.resize(600, min(520, self._max_h))

    def build_fn(self) -> Optional[Callable]:
        raise NotImplementedError

    def run_task(self) -> None:
        if self._thread is not None:
            return
        fn = self.build_fn()
        if fn is None:
            return
        self.btn_run.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setMaximum(0)
        self.status.setText("작업 중…")
        self.status.setStyleSheet(f"color: {FG_DIM};")

        self._thread = QThread(self)
        self._task = _Task(fn)
        self._task.moveToThread(self._thread)
        self._thread.started.connect(self._task.run)
        self._task.progress.connect(self._on_progress)
        self._task.finished.connect(self._on_finished)
        self._thread.start()

    def _on_progress(self, done: int, total: int, msg: str) -> None:
        self.progress.setMaximum(max(1, total))
        self.progress.setValue(done)
        self.status.setText(f"{done}/{total}  {msg}")

    def _on_finished(self, result, error: str) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(3000)
            self._thread = None
            self._task = None
        self.progress.setVisible(False)
        self.btn_run.setEnabled(True)
        if error:
            self.status.setText("실패: " + error)
            self.status.setStyleSheet(f"color: {ERR};")
            return
        self.result_data = result
        self.status.setText(self.describe(result))
        self.status.setStyleSheet(f"color: {OK};")

    def describe(self, result: Dict) -> str:
        return str(result)


class ImportLabelsDialog(_TaskDialog):
    """YOLO / COCO / VOC / LabelMe -> project labels (boxes stay dashed)."""

    def __init__(self, project, parent=None, sources=None):
        super().__init__("기존 라벨 가져오기", parent)
        self.project = project

        self._sources = list(sources or [])
        self.img_dir = QLineEdit()
        self.img_dir.setPlaceholderText("이미지 폴더")
        self.img_dir.setText(str(project.image_root) if project else "")
        btn_img = QPushButton("폴더…")
        rowi = QHBoxLayout()
        rowi.addWidget(self.img_dir, 1)
        rowi.addWidget(btn_img)

        self.btn_check = QPushButton("데이터 검증")
        self.report = QLabel("")
        self.report.setWordWrap(True)
        self.report.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.report.setStyleSheet(
            "background:#1c1f26; border:1px solid #2c313a; padding:8px;")
        self.report.setVisible(False)
        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(120)
        self.preview.setStyleSheet("border:1px solid #2c313a;")
        self.preview.setVisible(False)

        self.path = QLineEdit()
        self.path.setPlaceholderText("라벨 폴더 또는 파일")
        btn_dir = QPushButton("폴더…")
        btn_file = QPushButton("파일…")
        rowp = QHBoxLayout()
        rowp.addWidget(self.path, 1)
        rowp.addWidget(btn_dir)
        rowp.addWidget(btn_file)

        self.fmt = QComboBox()
        self.fmt.addItem("자동 감지", "auto")
        for k, v in FORMAT_LABELS.items():
            self.fmt.addItem(v, k)
        self.merge = QCheckBox("기존 라벨에 합치기 (끄면 덮어쓰기)")
        self.merge.setChecked(True)

        self.form.addRow("이미지 폴더", rowi)
        self.form.addRow("라벨 경로", rowp)
        self.form.addRow("포맷", self.fmt)
        self.form.addRow(self.merge)
        self.form.addRow("", self.btn_check)
        self.form.addRow(self.report)
        self.form.addRow(self.preview)
        self.form.addRow(QLabel(
            "가져온 박스는 점선으로 표시됩니다. 그 뒤 하나씩 클릭해 바꾸거나,\n"
            "이미지 단위·프로젝트 전체로 한 번에 변환할 수 있습니다."))

        btn_img.clicked.connect(self._pick_images)
        btn_dir.clicked.connect(self._pick_dir)
        btn_file.clicked.connect(self._pick_file)
        self.path.textChanged.connect(self._auto_detect)
        self.btn_check.clicked.connect(self.check_data)
        # Importing before looking is how a folder that matches nothing
        # becomes a silent no-op, so the run button waits for the check.
        self.btn_run.setEnabled(False)
        self.btn_run.setText("가져오기")

    def _pick_images(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "이미지 폴더 선택",
                                             self.img_dir.text().strip())
        if d:
            self.img_dir.setText(d)
            self.btn_run.setEnabled(False)
            self.report.setVisible(False)

    def _sync_image_dir(self) -> bool:
        """Point the project at the chosen image folder, if it moved."""
        want = self.img_dir.text().strip()
        if not want:
            return True
        q = Path(want)
        if not q.is_dir():
            QMessageBox.warning(self, "경로 오류",
                                "이미지 폴더가 없습니다:\n" + str(q))
            return False
        if Path(self.project.image_root).resolve() != q.resolve():
            self.project.image_dir = str(q.resolve())
            self.project.scan_images()
            self.project.rebuild_index()
            self.project.save()
        elif not self.project.images:
            self.project.scan_images()
        return True

    def sources(self):
        """Every label folder this dialog will import from."""
        if self._sources:
            return list(self._sources)
        t = self.path.text().strip()
        return [Path(t)] if t else []

    def check_data(self) -> None:
        """Pair images with labels and report before writing anything."""
        srcs = self.sources()
        if not srcs or not all(q.exists() for q in srcs):
            QMessageBox.warning(self, "경로 오류", "라벨 폴더나 파일을 지정하세요.")
            return
        if not self._sync_image_dir():
            return
        try:
            a = (analyze_many(self.project, srcs) if len(srcs) > 1
                 else analyze_import(self.project, srcs[0], self.fmt.currentData()))
        except Exception as e:              # a bad folder must not close it
            QMessageBox.warning(self, "검증 실패", f"{type(e).__name__}: {e}")
            return
        self._analysis = a
        text = describe_analysis(a)
        dup = stem_collisions(self.project)
        if dup:
            n = sum(len(v) for _k, v in dup)
            text += ("\n\n[!] 파일 이름이 겹치는 이미지가 있습니다 "
                     f"({n}장). 라벨은 이름으로 짝지어지므로 엉뚱한 이미지에 "
                     "붙을 수 있습니다:\n  " +
                     "\n  ".join(f"{k}: " + ", ".join(v[:3])
                                    for k, v in dup[:4]))
        self.report.setText(text)
        self.report.setVisible(True)
        ok = a["matched"] > 0
        self.btn_run.setEnabled(ok)
        self._show_preview(a)
        self.status.setText(
            f"검증 완료 — 박스 {a['boxes']}개를 세그먼트로 바꿀 수 있습니다."
            if ok else "매칭된 라벨이 없어 가져올 것이 없습니다.")
        # Grow to show the report if there is room, but never past the
        # screen and never *shrink* -- adjustSize() would collapse the
        # dialog to the scroll area's small hint.
        want = self.layout().sizeHint().height() + 40
        self.resize(max(600, self.width()),
                    max(self.height(), min(want, self._max_h)))

    def _show_preview(self, a: Dict) -> None:
        """Draw one matched image with its boxes, before importing anything.

        Counts alone cannot show that coordinates are read in the right
        convention -- a normalised/absolute mix-up, or boxes belonging to a
        different crop, look fine as numbers and obviously wrong as a picture.
        """
        sample = a.get("sample")
        if not sample:
            self.preview.setVisible(False)
            return
        try:
            bgr = imread_unicode(self.project.abs_image(sample["rel"]))
            rgb = None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            rgb = None
        if rgb is None:
            self.preview.setVisible(False)
            return
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg).scaled(
            360, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        sx, sy = pix.width() / max(1, w), pix.height() / max(1, h)
        painter = QPainter(pix)
        pen = QPen(QColor("#4c8dff"), 2)
        pen.setCosmetic(True)
        painter.setPen(pen)
        for (x1, y1, x2, y2) in sample["boxes"]:
            painter.drawRect(QRectF(x1 * sx, y1 * sy,
                                    (x2 - x1) * sx, (y2 - y1) * sy))
        painter.end()
        self.preview.setPixmap(pix)
        self.preview.setToolTip(
            f"{sample['rel']} — 박스 {len(sample['boxes'])}개 (미리보기)")
        self.preview.setVisible(True)

    def _pick_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "라벨 폴더 선택")
        if d:
            self.path.setText(d)

    def _pick_file(self) -> None:
        f, _ = QFileDialog.getOpenFileName(
            self, "라벨 파일 선택", "",
            "라벨 (*.json *.xml *.txt);;모든 파일 (*.*)")
        if f:
            self.path.setText(f)

    def _auto_detect(self, text: str) -> None:
        p = Path(text)
        if not text or not p.exists():
            return
        fmt = detect_format(p)
        if fmt != "unknown":
            self.status.setText(f"감지된 포맷: {FORMAT_LABELS.get(fmt, fmt)}")
            self.status.setStyleSheet(f"color: {FG_DIM};")

    def build_fn(self):
        srcs = self.sources()
        if not srcs or not all(q.exists() for q in srcs):
            QMessageBox.warning(self, "경로 오류", "존재하지 않는 경로입니다.")
            return None
        if not self._sync_image_dir():
            return None
        fmt = self.fmt.currentData()
        merge = self.merge.isChecked()
        project = self.project

        def run(prog):
            """Import every source, summing the per-source stats."""
            total = {"files": 0, "shapes": 0, "boxes": 0, "polys": 0,
                     "skipped": 0, "duplicates": 0, "format": ""}
            for i, q in enumerate(srcs):
                # Only the first source may overwrite; after that every
                # source must merge, or each one would wipe the last.
                r = import_labels(
                    project, q, fmt, merge if i == 0 else True,
                    progress=lambda d, t, m, _i=i: prog(
                        d, t, f"[{_i + 1}/{len(srcs)}] {m}"))
                for k in ("files", "shapes", "boxes", "polys", "skipped",
                          "duplicates"):
                    total[k] += r.get(k, 0)
                f = r.get("format", "")
                total["format"] = f if not total["format"] or total["format"] == f                     else "mixed"
            total["sources"] = len(srcs)
            return total

        return run

    def describe(self, r: Dict) -> str:
        n_src = r.get("sources", 0)
        head = f"완료 · 라벨 폴더 {n_src}곳 · " if n_src > 1 else "완료 · "
        msg = (f"{head}포맷 {r.get('format')} · 파일 {r.get('files', 0)}개 · "
               f"도형 {r.get('shapes', 0)}개 (박스 {r.get('boxes', 0)}, "
               f"폴리곤 {r.get('polys', 0)}) · 건너뜀 {r.get('skipped', 0)}")
        dup = r.get("duplicates", 0)
        if dup:
            msg += f" · 이미 있어 건너뛴 도형 {dup}개"
        return msg


class AutoLabelDialog(_TaskDialog):
    """Pre-label images with a model trained in this project.

    The predictions are ordinary shapes, marked `source="auto:<run>"` and
    left `verified=False`, so the point of the run is review rather than
    trust: the [도형] panel and the file list show exactly which images the
    model touched.
    """

    def __init__(self, project, settings, parent=None):
        super().__init__("학습한 모델로 자동 라벨링", parent)
        self.project = project
        self.settings = settings

        self.ckpt = QComboBox()
        self._ckpts = find_checkpoints(settings.runs_path)
        for c in self._ckpts:
            self.ckpt.addItem(f"{c.label}", str(c.path))
        self.btn_browse = QPushButton("다른 파일…")

        rowc = QHBoxLayout()
        rowc.addWidget(self.ckpt, 1)
        rowc.addWidget(self.btn_browse)

        self.conf = QDoubleSpinBox()
        self.conf.setRange(0.05, 0.95)
        self.conf.setSingleStep(0.05)
        self.conf.setValue(0.35)
        self.conf.setToolTip(
            "이 값보다 확신이 낮은 예측은 버립니다.\n"
            "높이면 정확하지만 놓치는 것이 늘고, 낮추면 반대입니다.")

        self.target = QComboBox()
        self.target.addItem("라벨이 없는 이미지만", "todo")
        self.target.addItem("모든 이미지", "all")
        self.replace = QCheckBox("기존 라벨을 지우고 새로 채우기")
        self.replace.setToolTip(
            "끄면 기존 도형 뒤에 덧붙입니다. 켜면 그 이미지의 라벨을 비우고\n"
            "예측만 남깁니다 — 되돌리려면 Ctrl+Z 가 아니라 다시 라벨해야 합니다.")

        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.note.setStyleSheet(f"color: {FG_DIM};")

        self.form.addRow("체크포인트", rowc)
        self.form.addRow("신뢰도 임계값", self.conf)
        self.form.addRow("대상", self.target)
        self.form.addRow(self.replace)
        self.form.addRow(self.note)

        self.btn_run.setText("자동 라벨링")
        self.btn_browse.clicked.connect(self._pick_ckpt)
        self.target.currentIndexChanged.connect(self._update_note)
        self.ckpt.currentIndexChanged.connect(self._update_note)
        self._update_note()

    def _pick_ckpt(self) -> None:
        f, _ = QFileDialog.getOpenFileName(
            self, "체크포인트 선택", str(self.settings.runs_path),
            "가중치 (*.pt);;모든 파일 (*.*)")
        if f:
            self.ckpt.addItem(Path(f).name, f)
            self.ckpt.setCurrentIndex(self.ckpt.count() - 1)

    def _targets(self):
        want = self.target.currentData()
        return [r for r in self.project.images
                if want == "all" or not self.project.is_labeled(r)]

    def _update_note(self) -> None:
        n = len(self._targets())
        msg = (f"대상 이미지 {n}장 · 예측은 검수 대상으로 저장됩니다"
               " (검증 완료 표시는 직접 하세요).")
        # Rebuilt on every change, so the "nothing trained yet" warning has to
        # be re-appended here or it vanishes the first time a combo moves.
        if self.ckpt.count() == 0:
            msg += ("\n\n학습된 체크포인트가 없습니다. [학습] 탭에서 먼저 "
                    "학습하거나 [다른 파일…] 로 .pt 를 직접 고르세요.")
        self.note.setText(msg)

    def build_fn(self):
        path = self.ckpt.currentData()
        if not path:
            QMessageBox.warning(self, "체크포인트 없음",
                                "사용할 가중치 파일을 고르세요.")
            return None
        rels = self._targets()
        if not rels:
            QMessageBox.information(self, "대상 없음", "라벨링할 이미지가 없습니다.")
            return None
        project = self.project
        conf = float(self.conf.value())
        replace = self.replace.isChecked()
        eps = float(self.settings.poly_epsilon)
        min_area = float(self.settings.min_area)
        smooth = int(self.settings.poly_smooth)
        device = resolve_device(self.settings.device)

        def run(prog):
            pred = Predictor(Path(path), device=("" if device == "cpu" else "0"))
            pred.load()
            cache: Dict[str, int] = {}
            done = shapes = touched = 0
            try:
                for i, rel in enumerate(rels):
                    prog(i, len(rels), rel)
                    bgr = imread_unicode(project.abs_image(rel))
                    if bgr is None:
                        continue
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    del bgr
                    found = pred.predict(rgb, conf=conf, epsilon=eps,
                                         min_area=min_area, smooth=smooth)
                    h, w = rgb.shape[:2]
                    del rgb                       # one image at a time
                    done += 1
                    if not found:
                        continue
                    ann = project.load_annotation(rel)
                    ann.width, ann.height = w, h
                    if replace:
                        ann.shapes = []
                    for idx, shape in found:
                        name = pred.names.get(idx, f"class{idx}")
                        shape.class_id = _ensure_class(project, name, cache)
                        shape.source = f"auto:{Path(path).parents[1].name}"
                        ann.shapes.append(shape)
                        shapes += 1
                    ann.verified = False          # everything here needs review
                    project.save_annotation(ann)
                    touched += 1
                prog(len(rels), len(rels), "완료")
            finally:
                pred.unload()
            project.save_index()
            return {"images": done, "touched": touched, "shapes": shapes,
                    "classes": sorted(cache)}

        return run

    def describe(self, r: Dict) -> str:
        return (f"완료 · 이미지 {r.get('images', 0)}장 검사 · "
                f"{r.get('touched', 0)}장에 도형 {r.get('shapes', 0)}개 추가 · "
                f"클래스 {', '.join(r.get('classes') or []) or '없음'}"
                "  —  검수 후 Ctrl+D 로 확정하세요.")


class ExportDatasetDialog(_TaskDialog):
    def __init__(self, project, settings, fmt: str = "yolo", parent=None):
        super().__init__("학습 데이터셋 내보내기", parent)
        self.project = project
        self.settings = settings

        self.fmt = QComboBox()
        for k, v in EXPORT_LABELS.items():
            self.fmt.addItem(v, k)
        i = self.fmt.findData(fmt)
        if i >= 0:
            self.fmt.setCurrentIndex(i)

        default_out = Path(settings.last_export_dir) / f"{project.name}_{fmt}"
        self.out = QLineEdit(str(default_out))
        btn_out = QPushButton("…")
        btn_out.setFixedWidth(32)
        rowo = QHBoxLayout()
        rowo.addWidget(self.out, 1)
        rowo.addWidget(btn_out)

        # How the images are divided. Ratio mode reshuffles everything by a
        # hash of the path; folder mode keeps whatever split the dataset
        # already declares in its folder names.
        self.split_mode = QComboBox()
        self.split_mode.addItem("비율로 나누기", "ratio")
        self.split_mode.addItem("원본 폴더 구조 유지", "folder")
        self.split_mode.setToolTip(
            "비율로 나누기 — 경로 해시로 train/val/test 를 새로 나눕니다.\n"
            "  같은 시드면 몇 번을 내보내도 배치가 바뀌지 않습니다.\n"
            "원본 폴더 구조 유지 — 이미지 경로에 들어 있는 train/val/test\n"
            "  폴더 이름을 그대로 따릅니다. 기존 평가 셋을 지켜야 할 때 쓰세요.")

        self.val = QDoubleSpinBox()
        self.val.setRange(0.0, 0.9)
        self.val.setSingleStep(0.05)
        self.val.setDecimals(2)
        self.val.setValue(0.2)
        self.test = QDoubleSpinBox()
        self.test.setRange(0.0, 0.5)
        self.test.setSingleStep(0.05)
        self.test.setDecimals(2)
        self.test.setValue(0.1)
        self.mode = QComboBox()
        self.mode.addItem("이미지 복사", "copy")
        self.mode.addItem("심볼릭 링크(디스크 절약)", "symlink")
        self.only_verified = QCheckBox("검증 완료된 이미지만")

        self.max_side = QSpinBox()
        self.max_side.setRange(0, 8192)
        self.max_side.setSingleStep(32)
        self.max_side.setSpecialValueText("원본 크기")
        self.max_side.setValue(int(getattr(settings, "export_max_side", 0)))
        self.max_side.setToolTip(
            "긴 변이 이 픽셀 수를 넘으면 줄여서 내보냅니다. 0 = 원본 그대로.\n"
            "라벨 좌표도 같은 비율로 함께 조정됩니다.\n"
            "이미 작은 이미지는 늘리지 않습니다.")

        st = project.stats()
        info = QLabel(f"라벨된 이미지 {st['labeled']}장 · 도형 {st['shapes']}개 · "
                      f"클래스 {len(project.classes)}개")
        info.setStyleSheet(f"color: {FG_DIM};")

        # Live read-out of the resulting split, so the counts are visible
        # before the export runs rather than only in the summary afterwards.
        self.split_info = QLabel()
        self.split_info.setWordWrap(True)
        self.split_info.setStyleSheet(f"color: {FG_DIM};")

        self.form.addRow("포맷", self.fmt)
        self.form.addRow("출력 폴더", rowo)
        self.form.addRow("분할 방식", self.split_mode)
        self.form.addRow("val 비율", self.val)
        self.form.addRow("test 비율", self.test)
        self.form.addRow("", self.split_info)
        self.form.addRow("이미지 처리", self.mode)
        self.form.addRow("이미지 긴 변(px)", self.max_side)
        self.form.addRow(self.only_verified)
        self.form.addRow(info)

        btn_out.clicked.connect(self._pick_out)
        self.fmt.currentIndexChanged.connect(self._sync_out)
        self.max_side.valueChanged.connect(self._on_max_side)
        self.split_mode.currentIndexChanged.connect(self._on_split_mode)
        self.val.valueChanged.connect(self._refresh_split_info)
        self.test.valueChanged.connect(self._refresh_split_info)
        self.only_verified.toggled.connect(self._refresh_split_info)
        self._on_max_side(self.max_side.value())
        self._on_split_mode()

    # ---- split mode ------------------------------------------------------
    def _labeled_rels(self):
        only_v = self.only_verified.isChecked()
        return [r for r in self.project.images
                if self.project.is_labeled(r)
                and (not only_v or self.project.is_verified(r))]

    def _folder_counts(self):
        """What the dataset's own folder names say. Paths only -- no pixels."""
        key = (self.only_verified.isChecked(),)
        if getattr(self, "_fc_key", None) != key:
            c = {"train": 0, "val": 0, "test": 0, "none": 0}
            for rel in self._labeled_rels():
                c[folder_split(rel) or "none"] += 1
            self._fc_key, self._fc = key, c
        return self._fc

    def _on_split_mode(self) -> None:
        """Ratio spin boxes mean nothing in folder mode, so grey them out."""
        ratio = self.split_mode.currentData() == "ratio"
        for w in (self.val, self.test):
            w.setEnabled(ratio)
            lbl = self.form.labelForField(w)
            if lbl is not None:
                lbl.setEnabled(ratio)
        self._refresh_split_info()

    def _refresh_split_info(self) -> None:
        if self.split_mode.currentData() == "folder":
            c = self._folder_counts()
            if not (c["train"] or c["val"] or c["test"]):
                self.split_info.setText(
                    "※ 이미지 경로에서 train/val/test 폴더를 "
                    "찾지 못했습니다 — [비율로 나누기] 를 쓰세요.")
                return
            txt = ("원본 폴더 기준 — "
                   f"train {c['train']} / val {c['val']} / test {c['test']}")
            if c["none"]:
                # These would be swept into train unannounced otherwise.
                txt += ("\n※ 경로에 train/val/test 가 없는 "
                        f"{c['none']}장은 train 으로 들어갑니다.")
            self.split_info.setText(txt)
            return

        n = len(self._labeled_rels())
        v, t = self.val.value(), self.test.value()
        if v + t >= 1.0:
            self.split_info.setText(
                "※ val + test 가 1.00 이상입니다 — train 이 비게 됩니다.")
            return
        # An estimate only: the real split is per-file by path hash, so the
        # counts land near these numbers rather than exactly on them.
        n_test = round(n * t)
        n_val = round(n * v)
        self.split_info.setText(
            f"예상 — train {n - n_val - n_test} / val {n_val} / test {n_test} "
            "(해시 분배라 약간 다를 수 있음)"
            "\n추천 비율: ~1천장 val 0.20 / test 0.10 · "
            "1천~1만장 val 0.15 / test 0.10 · 1만장~ val 0.10 / test 0.05")

    def _on_max_side(self, v: int) -> None:
        """Resizing rewrites every image, so a symlink cannot be used."""
        link = self.mode.findData("symlink")
        if link < 0:
            return
        item = self.mode.model().item(link)
        item.setEnabled(v == 0)
        if v and self.mode.currentData() == "symlink":
            self.mode.setCurrentIndex(self.mode.findData("copy"))

    def _pick_out(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "출력 폴더", self.out.text())
        if d:
            self.out.setText(d)

    def _sync_out(self) -> None:
        base = Path(self.settings.last_export_dir)
        self.out.setText(str(base / f"{self.project.name}_{self.fmt.currentData()}"))

    def build_fn(self):
        st = self.project.stats()
        if st["labeled"] == 0:
            QMessageBox.warning(self, "데이터 없음", "라벨된 이미지가 없습니다.")
            return None
        if not self.project.classes:
            QMessageBox.warning(self, "클래스 없음", "클래스를 먼저 만들어 주세요.")
            return None
        fmt = self.fmt.currentData()
        smode = self.split_mode.currentData()
        if smode == "ratio" and self.val.value() + self.test.value() >= 1.0:
            QMessageBox.warning(
                self, "비율 오류",
                "val + test 비율이 1.00 미만이어야 합니다.")
            return None
        if smode == "folder":
            c = self._folder_counts()
            if not (c["train"] or c["val"] or c["test"]):
                QMessageBox.warning(
                    self, "폴더 구조 없음",
                    "이미지 경로에서 train/val/test 폴더를 "
                    "찾지 못했습니다.\n"
                    "[비율로 나누기] 를 선택하세요.")
                return None
        out = Path(self.out.text().strip())
        ms = int(self.max_side.value())
        self.settings.export_max_side = ms
        self.settings.save()
        kw = dict(val_ratio=self.val.value(), test_ratio=self.test.value(),
                  image_mode=self.mode.currentData(),
                  only_verified=self.only_verified.isChecked(),
                  max_side=ms, split_mode=smode)
        project = self.project
        return lambda prog: export_dataset(
            project, fmt, out, progress=lambda d, t, m: prog(d, t, m), **kw)

    def describe(self, r: Dict) -> str:
        c = r.get("counts", {})
        how = ("원본 폴더 유지" if r.get("split_mode") == "folder"
               else "비율 분할")
        extra = ""
        if r.get("unsplit"):
            extra = ("\n※ 폴더를 못 찾아 train 으로 보낸 "
                     f"이미지 {r['unsplit']}장")
        return (f"완료 · {r.get('root')}\n"
                f"{how} · train {c.get('train', 0)} / val {c.get('val', 0)} / "
                f"test {c.get('test', 0)} · 클래스 {len(r.get('classes', []))}개"
                + extra)
