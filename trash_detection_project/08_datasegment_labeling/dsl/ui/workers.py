"""Background workers: SAM inference, thumbnail decoding, batch conversion.

All CUDA work happens on ONE worker thread so the GUI never blocks and the GPU
is never touched from two threads at once.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import os

import numpy as np
from PySide6.QtCore import (QObject, QRunnable, QThreadPool, Qt, QTimer, Signal,
                            Slot)
from PySide6.QtGui import QImage, QPixmap

from ..core.annotation import ImageAnnotation, Shape
from ..core.image_cache import imread_unicode
from ..core.annotation import Shape
from ..core.mask_utils import mask_to_polygons
from ..sam.backends import create_backend
from ..sam.base import SamBackendError


# ---------------------------------------------------------------------------
# SAM worker
# ---------------------------------------------------------------------------
class SamWorker(QObject):
    """Owns the SAM backend. Every method runs on the worker thread."""

    loaded = Signal(str)                     # display name
    capabilities = Signal(bool, bool, bool)  # points, box, text
    loadFailed = Signal(str)
    imageReady = Signal(str)                 # image key encoded
    result = Signal(object, object, int)     # masks, scores, request id
    failed = Signal(str)
    busy = Signal(bool)
    batchProgress = Signal(int, int, str)    # done, total, message
    batchFinished = Signal(int, int)         # converted shapes, images touched

    def __init__(self):
        super().__init__()
        self.backend = None
        self.model_key = ""
        self._scale = 1.0
        self._max_side = 0
        self._cancel_batch = False

    # ---- model ------------------------------------------------------------
    @Slot(str, str, str, str)
    def load_model(self, key: str, weights_dir: str, device: str, precision: str):
        self.busy.emit(True)
        try:
            if self.backend is not None:
                self.backend.unload()
                self.backend = None
            backend = create_backend(key, Path(weights_dir), device, precision)
            backend.load()
            self.backend = backend
            self.model_key = key
            # advertise what this backend can actually be prompted with, so
            # the window can disable tools it does not support
            caps = [n for n, ok in (("점", backend.supports_points),
                                    ("박스", backend.supports_box),
                                    ("텍스트", backend.supports_text)) if ok]
            self.loaded.emit(
                f"{backend.spec.display} [{device}/{backend.precision}]"
                f" · {'/'.join(caps)} 프롬프트")
            self.capabilities.emit(bool(backend.supports_points),
                                   bool(backend.supports_box),
                                   bool(backend.supports_text))
        except SamBackendError as e:
            self.loadFailed.emit(str(e))
        except Exception as e:                       # torch/hydra/hub errors
            self.loadFailed.emit(f"모델 로딩 실패: {e}")
        finally:
            self.busy.emit(False)

    @Slot()
    def unload_model(self):
        if self.backend is not None:
            self.backend.unload()
            self.backend = None
            self.model_key = ""

    @Slot(int)
    def set_max_side(self, px: int):
        self._max_side = int(px)

    # ---- image ------------------------------------------------------------
    def _prepare(self, rgb: np.ndarray) -> np.ndarray:
        self._scale = 1.0
        if self._max_side and max(rgb.shape[:2]) > self._max_side:
            self._scale = self._max_side / max(rgb.shape[:2])
            rgb = cv2.resize(rgb, None, fx=self._scale, fy=self._scale,
                             interpolation=cv2.INTER_AREA)
        return rgb

    @Slot(object, str)
    def set_image(self, rgb: np.ndarray, key: str):
        if self.backend is None or rgb is None:
            return
        self.busy.emit(True)
        try:
            self.backend.set_image(self._prepare(rgb), key)
            self.imageReady.emit(key)
        except Exception as e:
            self.failed.emit(f"이미지 인코딩 실패: {e}")
        finally:
            self.busy.emit(False)

    @Slot()
    def reset_image(self):
        if self.backend is not None:
            self.backend.reset_image()

    # ---- prediction -------------------------------------------------------
    def _upscale(self, masks: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        if self._scale == 1.0 or masks is None or len(masks) == 0:
            return masks
        h, w = shape
        return np.stack([cv2.resize(m.astype(np.uint8), (w, h),
                                    interpolation=cv2.INTER_NEAREST).astype(bool)
                         for m in masks])

    @Slot(list, object, int, int, int)
    def predict(self, points: list, box, req_id: int, img_h: int, img_w: int):
        if self.backend is None:
            self.failed.emit("SAM 모델이 로드되지 않았습니다.")
            return
        self.busy.emit(True)
        try:
            s = self._scale
            pts = [(x * s, y * s, l) for x, y, l in points] if s != 1.0 else points
            bx = [v * s for v in box] if (box and s != 1.0) else box
            res = self.backend.predict(points=pts, box=bx, multimask=True)
            masks = self._upscale(res.masks, (img_h, img_w))
            self.result.emit(masks, res.scores, req_id)
        except SamBackendError as e:
            self.failed.emit(str(e))
        except Exception as e:
            self.failed.emit(f"추론 실패: {e}")
        finally:
            self.busy.emit(False)

    @Slot(str, int, int, int)
    def predict_text(self, text: str, req_id: int, img_h: int, img_w: int):
        if self.backend is None:
            self.failed.emit("SAM 모델이 로드되지 않았습니다.")
            return
        self.busy.emit(True)
        try:
            res = self.backend.predict_text(text)
            self.result.emit(self._upscale(res.masks, (img_h, img_w)),
                             res.scores, req_id)
        except SamBackendError as e:
            self.failed.emit(str(e))
        except Exception as e:
            self.failed.emit(f"텍스트 추론 실패: {e}")
        finally:
            self.busy.emit(False)

    # ---- batch box -> segment --------------------------------------------
    @Slot()
    def cancel_batch(self):
        self._cancel_batch = True

    @Slot(object, list, int, float, float, bool)
    def batch_convert(self, project, rels: list, class_filter: int,
                      epsilon: float, min_area: float, keep_largest: bool,
                      smooth: int = 0):
        """Convert imported rectangles into polygons, one image at a time."""
        if self.backend is None:
            self.failed.emit("SAM 모델이 로드되지 않았습니다.")
            self.batchFinished.emit(0, 0)
            return
        self._cancel_batch = False
        self.busy.emit(True)
        converted = 0
        touched = 0
        try:
            for i, rel in enumerate(rels):
                if self._cancel_batch:
                    break
                self.batchProgress.emit(i, len(rels), rel)
                ann = project.load_annotation(rel)
                boxes = [s for s in ann.shapes if s.shape_type == "rectangle"
                         and (class_filter < 0 or s.class_id == class_filter)]
                if not boxes:
                    continue
                bgr = imread_unicode(project.abs_image(rel))
                if bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                del bgr
                H, W = rgb.shape[:2]
                ann.width, ann.height = W, H
                self.backend.set_image(self._prepare(rgb), key=f"batch::{rel}")
                del rgb

                changed = False
                for shp in boxes:
                    if self._cancel_batch:
                        break
                    x1, y1, x2, y2 = shp.bbox()
                    s = self._scale
                    res = self.backend.predict(
                        points=(), box=[x1 * s, y1 * s, x2 * s, y2 * s],
                        multimask=False)
                    mask = res.best()
                    if mask is None:
                        continue
                    if s != 1.0:
                        mask = cv2.resize(mask.astype(np.uint8), (W, H),
                                          interpolation=cv2.INTER_NEAREST) > 0
                    polys = mask_to_polygons(mask, epsilon, min_area,
                                             keep_largest_only=True,
                                             smooth=smooth)
                    if not polys:
                        continue
                    outer, holes = polys[0]
                    shp.points = outer
                    shp.holes = holes
                    shp.shape_type = "polygon"
                    shp.source = self.model_key or "sam"
                    shp.score = float(res.scores[0]) if len(res.scores) else 1.0
                    shp.meta = dict(shp.meta or {})
                    shp.meta["src_bbox"] = [x1, y1, x2, y2]
                    converted += 1
                    changed = True
                if changed:
                    project.save_annotation(ann)
                    touched += 1
                self.backend.reset_image()
            self.batchProgress.emit(len(rels), len(rels), "완료")
        except Exception as e:
            self.failed.emit(f"일괄 변환 실패: {e}")
        finally:
            self.busy.emit(False)
            self.batchFinished.emit(converted, touched)


    @Slot(object, list, int, float, float, int, int)
    def batch_auto(self, project, rels: list, class_id: int, epsilon: float,
                   min_area: float, smooth: int, max_masks: int):
        """Segment everything, with no prompt, over a list of images.

        The model has no idea what the things are, so every mask lands on the
        class the user picked -- this produces candidates to sort out, not
        finished labels. Saved unverified for exactly that reason.
        """
        if self.backend is None:
            self.failed.emit("SAM 모델이 로드되지 않았습니다.")
            self.batchFinished.emit(0, 0)
            return
        if not getattr(self.backend, "supports_auto", False):
            self.failed.emit(
                "이 백엔드는 자동 분할을 지원하지 않습니다 "
                "(SAM 1 / SAM 2 / Ultralytics 계열에서 동작합니다).")
            self.batchFinished.emit(0, 0)
            return
        self._cancel_batch = False
        self.busy.emit(True)
        added = touched = 0
        try:
            for i, rel in enumerate(rels):
                if self._cancel_batch:
                    break
                self.batchProgress.emit(i, len(rels), rel)
                bgr = imread_unicode(project.abs_image(rel))
                if bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                del bgr
                H, W = rgb.shape[:2]
                small = self._prepare(rgb)
                del rgb
                try:
                    masks = self.backend.auto_masks(small, max_masks)
                except Exception as e:
                    self.failed.emit(f"{rel}: {e}")
                    break
                del small
                if not masks:
                    continue
                ann = project.load_annotation(rel)
                ann.width, ann.height = W, H
                changed = False
                s_ = self._scale
                for m in masks:
                    if self._cancel_batch:
                        break
                    m = np.asarray(m)
                    if s_ != 1.0 or m.shape[:2] != (H, W):
                        m = cv2.resize(m.astype(np.uint8), (W, H),
                                       interpolation=cv2.INTER_NEAREST) > 0
                    polys = mask_to_polygons(m, epsilon, min_area,
                                             keep_largest_only=True,
                                             smooth=smooth)
                    for outer, holes in polys:
                        ann.shapes.append(Shape(
                            class_id=class_id, points=outer, holes=holes,
                            source=f"{self.model_key or 'sam'}:auto"))
                        added += 1
                        changed = True
                if changed:
                    ann.verified = False
                    project.save_annotation(ann)
                    touched += 1
                self.backend.reset_image()
            self.batchProgress.emit(len(rels), len(rels), "완료")
        except Exception as e:
            self.failed.emit(f"자동 분할 실패: {e}")
        finally:
            self.busy.emit(False)
            self.batchFinished.emit(added, touched)

    @Slot(object, list, str, int, float, float, int)
    def batch_text(self, project, rels: list, text: str, class_id: int,
                   epsilon: float, min_area: float, smooth: int = 0):
        """Run one text prompt over many images, one image at a time.

        SAM 3 answers a concept ("car") with every instance it finds, so a
        whole project can be pre-labelled from a single phrase. Each image is
        loaded, prompted and released before the next, exactly like
        batch_convert -- the embedding alone is hundreds of MB.
        """
        if self.backend is None:
            self.failed.emit("SAM 모델이 로드되지 않았습니다.")
            self.batchFinished.emit(0, 0)
            return
        if not getattr(self.backend, "supports_text", False):
            self.failed.emit(
                "이 백엔드는 텍스트 프롬프트를 지원하지 않습니다 (SAM 3 계열 필요).")
            self.batchFinished.emit(0, 0)
            return
        self._cancel_batch = False
        self.busy.emit(True)
        added = touched = 0
        try:
            for i, rel in enumerate(rels):
                if self._cancel_batch:
                    break
                self.batchProgress.emit(i, len(rels), rel)
                bgr = imread_unicode(project.abs_image(rel))
                if bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                del bgr
                H, W = rgb.shape[:2]
                self.backend.set_image(self._prepare(rgb), key=f"text::{rel}")
                del rgb
                try:
                    res = self.backend.predict_text(text)
                except Exception as e:
                    self.failed.emit(f"{rel}: {e}")
                    self.backend.reset_image()
                    continue
                masks = getattr(res, "masks", None)
                if masks is None or len(res) == 0:
                    self.backend.reset_image()
                    continue
                ann = project.load_annotation(rel)
                ann.width, ann.height = W, H
                changed = False
                s_ = self._scale
                for k, m in enumerate(masks):
                    if self._cancel_batch:
                        break
                    if s_ != 1.0:
                        m = cv2.resize(m.astype(np.uint8), (W, H),
                                       interpolation=cv2.INTER_NEAREST) > 0
                    polys = mask_to_polygons(m, epsilon, min_area,
                                             keep_largest_only=False,
                                             smooth=smooth)
                    for outer, holes in polys:
                        ann.shapes.append(Shape(
                            class_id=class_id, points=outer, holes=holes,
                            score=float(res.scores[k]) if k < len(res.scores) else 1.0,
                            source=f"{self.model_key or 'sam'}:text"))
                        added += 1
                        changed = True
                if changed:
                    ann.verified = False       # a prompt is a proposal
                    project.save_annotation(ann)
                    touched += 1
                self.backend.reset_image()
            self.batchProgress.emit(len(rels), len(rels), "완료")
        except Exception as e:
            self.failed.emit(f"텍스트 일괄 적용 실패: {e}")
        finally:
            self.busy.emit(False)
            self.batchFinished.emit(added, touched)


# ---------------------------------------------------------------------------
# thumbnails
# ---------------------------------------------------------------------------
class _ThumbSignals(QObject):
    done = Signal(str, object)          # rel path, QImage


class ThumbnailTask(QRunnable):
    """Decode one thumbnail off the GUI thread, at reduced resolution."""

    def __init__(self, rel: str, path: Path, size: int, signals: _ThumbSignals):
        super().__init__()
        self.rel = rel
        self.path = Path(path)
        self.size = int(size)
        self.signals = signals
        self.setAutoDelete(True)

    def run(self):
        try:
            buf = np.fromfile(str(self.path), dtype=np.uint8)
            if buf.size == 0:
                self.signals.done.emit(self.rel, None)
                return
            # IMREAD_REDUCED decodes at 1/2, 1/4, 1/8 -- far cheaper than a
            # full decode followed by a resize. Pick the factor from the
            # header (about 2ms) instead of guessing 1/8 and re-decoding when
            # it comes out too small: every image under ~800px used to be
            # decoded twice, which cost more than reading the header.
            flags = {8: cv2.IMREAD_REDUCED_COLOR_8,
                     4: cv2.IMREAD_REDUCED_COLOR_4,
                     2: cv2.IMREAD_REDUCED_COLOR_2,
                     1: cv2.IMREAD_COLOR}
            # Guess the starting factor from the bytes already in hand: a
            # small file is a small image, and asking for 1/8 of it returns
            # something below the thumbnail size, forcing a second decode.
            # Reading the real header instead costs ~2ms, which is more than
            # it saves on the large images that dominate a slow folder.
            order = [4, 2, 1] if buf.size < 300_000 else [8, 4, 2, 1]
            img = None
            for f in order:
                img = cv2.imdecode(buf, flags[f])
                if img is not None and (min(img.shape[:2]) >= self.size or f == order[-1]):
                    break
            if img is None:
                self.signals.done.emit(self.rel, None)
                return
            h, w = img.shape[:2]
            sc = self.size / max(h, w)
            if sc < 1.0:
                img = cv2.resize(img, (max(1, int(w * sc)), max(1, int(h * sc))),
                                 interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                          rgb.strides[0], QImage.Format_RGB888).copy()
            self.signals.done.emit(self.rel, qimg)
        except (OSError, ValueError, cv2.error):
            self.signals.done.emit(self.rel, None)


class ThumbnailPool(QObject):
    """Fans thumbnail requests out to a small pool; drops duplicates."""

    ready = Signal(str, object)

    def __init__(self, max_threads: int = 0):
        super().__init__()
        self.pool = QThreadPool()
        # JPEG decoding is CPU bound and parallel, so a fixed 3 left most of
        # the machine idle. Capped anyway: past a point the disk, not the
        # CPU, is the limit, and the labelling thread still needs cores.
        if max_threads <= 0:
            max_threads = min(8, max(3, (os.cpu_count() or 4) // 3))
        self.pool.setMaxThreadCount(max_threads)
        self._signals = _ThumbSignals()
        self._signals.done.connect(self._on_done, Qt.QueuedConnection)
        self._pending: set = set()
        self._seq = 0

    def request(self, rel: str, path: Path, size: int) -> None:
        if rel in self._pending:
            return
        self._pending.add(rel)
        # Newest request wins. The model asks only for rows about to be
        # painted, so "most recent" is "what the user is looking at now" --
        # without this, a fast scroll leaves the visible rows queued behind
        # hundreds of thumbnails that already scrolled past.
        self._seq += 1
        self.pool.start(ThumbnailTask(rel, path, size, self._signals), self._seq)

    def _on_done(self, rel: str, qimg) -> None:
        self._pending.discard(rel)
        self.ready.emit(rel, qimg)

    def clear(self) -> None:
        self.pool.clear()
        self._pending.clear()
