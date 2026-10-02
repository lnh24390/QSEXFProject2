"""Run a model you already trained over unlabelled images.

This is the loop the app is built around: label a little, train, let the
model propose the rest, correct what it got wrong, train again. Predictions
land as ordinary shapes with `source="auto:<model>"` and the model's own
confidence in `score`, so they are told apart from hand work and can be
filtered or deleted wholesale if a run turns out badly.

Nothing here is imported at module load: ultralytics and torch stay optional
(DEVELOPMENT.md 2).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.annotation import Shape
from ..core.mask_utils import mask_to_polygons


@dataclass
class Checkpoint:
    """One trained weights file the user could predict with."""
    path: Path
    model_key: str          # registry key the run belongs to
    run_name: str
    mtime: float

    @property
    def label(self) -> str:
        return f"{self.model_key} / {self.run_name}"


def find_checkpoints(runs_root: Path) -> List[Checkpoint]:
    """Every best.pt under runs/, newest first.

    Layout is runs/<model key>/<run name>/weights/best.pt (DEVELOPMENT.md 5.2),
    so the two directory names above `weights/` name the run.
    """
    runs_root = Path(runs_root)
    out: List[Checkpoint] = []
    if not runs_root.is_dir():
        return out
    for p in runs_root.glob("*/*/weights/best.pt"):
        try:
            mtime = p.stat().st_mtime
        except OSError:
            continue
        out.append(Checkpoint(path=p, model_key=p.parents[2].name,
                              run_name=p.parents[1].name, mtime=mtime))
    out.sort(key=lambda c: -c.mtime)
    return out


class Predictor:
    """An ultralytics checkpoint, loaded once and reused across images."""

    def __init__(self, weights: Path, device: str = ""):
        self.weights = Path(weights)
        self.device = device
        self.model = None
        self.names: Dict[int, str] = {}
        self.task = "segment"

    def load(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError(
                "ultralytics 가 필요합니다.\n    uv pip install ultralytics") from e
        if not self.weights.exists():
            raise RuntimeError(f"체크포인트가 없습니다: {self.weights}")
        self.model = YOLO(str(self.weights))
        raw = getattr(self.model, "names", None) or {}
        self.names = {int(k): str(v) for k, v in dict(raw).items()}
        # A detection checkpoint has no mask head; its predictions become
        # rectangles rather than being silently dropped.
        self.task = "detect" if getattr(self.model, "task", "") == "detect" \
            else "segment"

    def unload(self) -> None:
        self.model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def predict(self, rgb: np.ndarray, conf: float = 0.35,
                epsilon: float = 0.002, min_area: float = 24.0,
                smooth: int = 2, max_det: int = 300
                ) -> List[Tuple[int, Shape]]:
        """(class index, Shape) for one image, in original pixel coordinates."""
        if self.model is None:
            raise RuntimeError("모델이 로드되지 않았습니다.")
        import cv2
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        kw = dict(conf=float(conf), verbose=False, max_det=int(max_det))
        if self.device:
            kw["device"] = self.device
        res = self.model(bgr, **kw)
        if not res:
            return []
        r = res[0]
        H, W = rgb.shape[:2]
        out: List[Tuple[int, Shape]] = []

        boxes = getattr(r, "boxes", None)
        cls = ([] if boxes is None or boxes.cls is None
               else boxes.cls.cpu().numpy().astype(int).tolist())
        scores = ([] if boxes is None or boxes.conf is None
                  else boxes.conf.cpu().numpy().astype(float).tolist())

        masks = getattr(r, "masks", None)
        if masks is not None and masks.data is not None and self.task != "detect":
            data = masks.data.cpu().numpy()
            for i, m in enumerate(data):
                if m.shape[:2] != (H, W):
                    m = cv2.resize(m.astype(np.uint8), (W, H),
                                   interpolation=cv2.INTER_NEAREST)
                polys = mask_to_polygons(m > 0, epsilon, min_area,
                                         keep_largest_only=True, smooth=smooth)
                if not polys:
                    continue
                outer, holes = polys[0]
                out.append((cls[i] if i < len(cls) else 0,
                            Shape(class_id=-1, points=outer, holes=holes,
                                  score=scores[i] if i < len(scores) else 1.0)))
            return out

        # detection checkpoint (or a seg model that returned no masks)
        if boxes is None or boxes.xyxy is None:
            return []
        for i, xyxy in enumerate(boxes.xyxy.cpu().numpy().tolist()):
            x1, y1, x2, y2 = (float(v) for v in xyxy)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            out.append((cls[i] if i < len(cls) else 0,
                        Shape(class_id=-1,
                              points=[(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
                              shape_type="rectangle",
                              score=scores[i] if i < len(scores) else 1.0)))
        return out
