"""Common interface every SAM version is wrapped in.

The UI only ever talks to this interface, so adding SAM 4 later means adding
one class plus one catalog entry -- no UI changes (DEVELOPMENT.md section 4).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class SamResult:
    """masks: (N, H, W) bool, scores: (N,) float, sorted best-first."""
    masks: np.ndarray
    scores: np.ndarray
    labels: List[str] = field(default_factory=list)   # SAM 3 text prompts

    def best(self) -> Optional[np.ndarray]:
        if self.masks is None or len(self.masks) == 0:
            return None
        return self.masks[int(np.argmax(self.scores))]

    def __len__(self) -> int:
        return 0 if self.masks is None else len(self.masks)


EMPTY = SamResult(masks=np.zeros((0, 1, 1), dtype=bool),
                  scores=np.zeros((0,), dtype=np.float32))


class SamBackendError(RuntimeError):
    pass


class SamBackend(ABC):
    """One loaded checkpoint. Holds at most ONE image embedding at a time."""

    supports_text = False
    supports_auto = False        # can segment everything, unprompted
    supports_box = True
    supports_points = True

    def __init__(self, spec, checkpoint: Path, device: str = "cuda",
                 precision: str = "fp16"):
        self.spec = spec
        self.checkpoint = Path(checkpoint)
        self.device = device
        self.precision = precision
        self.model = None
        self.predictor = None
        self._image_key: Optional[str] = None
        self._image_shape: Tuple[int, int] = (0, 0)

    # ---- lifecycle --------------------------------------------------------
    @abstractmethod
    def load(self) -> None:
        """Build the model. Raise SamBackendError with an actionable message."""

    @abstractmethod
    def _encode(self, rgb: np.ndarray) -> None:
        """Compute and cache the embedding for one image."""

    @abstractmethod
    def predict(self, points: Sequence[Tuple[float, float, int]] = (),
                box: Optional[Sequence[float]] = None,
                mask_input: Optional[np.ndarray] = None,
                multimask: bool = True) -> SamResult:
        """points: [(x, y, 1|0)] with 1=foreground, 0=background."""

    def auto_masks(self, rgb, max_masks: int = 200):
        """Every object the model can find, with no prompt at all.

        Returns masks only: an unprompted run has no idea what the things
        are, so the caller has to supply the class. Backends that cannot do
        this raise, rather than quietly returning nothing.
        """
        raise SamBackendError(
            f"{self.spec.display} 은 자동 분할을 지원하지 않습니다.")

    def predict_text(self, text: str) -> SamResult:
        raise SamBackendError(
            f"{self.spec.display} 은 텍스트 프롬프트를 지원하지 않습니다. "
            "SAM 3 계열 모델을 선택하세요.")

    # ---- image handling ---------------------------------------------------
    def set_image(self, rgb: np.ndarray, key: str = "") -> None:
        """Encode only when the image actually changed."""
        if key and key == self._image_key:
            return
        self.reset_image()
        self._encode(rgb)
        self._image_key = key or None
        self._image_shape = rgb.shape[:2]

    def reset_image(self) -> None:
        """Drop the cached embedding -- the single biggest memory consumer."""
        self._image_key = None
        p = self.predictor
        if p is not None and hasattr(p, "reset_image"):
            try:
                p.reset_image()
            except (AttributeError, RuntimeError):
                pass
        elif p is not None and hasattr(p, "reset_predictor"):
            try:
                p.reset_predictor()
            except (AttributeError, RuntimeError):
                pass
        self.empty_cache()

    def unload(self) -> None:
        self.reset_image()
        self.predictor = None
        self.model = None
        self.empty_cache()

    def empty_cache(self) -> None:
        try:
            import torch
            if self.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # ---- helpers ----------------------------------------------------------
    def autocast(self):
        """bf16/fp16 autocast on CUDA; a no-op elsewhere."""
        try:
            import torch
        except ImportError:
            return nullcontext()
        if not self.device.startswith("cuda"):
            return nullcontext()
        if self.precision == "fp16":
            return torch.autocast("cuda", dtype=torch.float16)
        if self.precision in ("bf16", "bfloat16"):
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    def apply_precision(self, model):
        """Cast the *weights* at load time, on the checkpoint as downloaded.

        `autocast` alone only casts activations -- the weights stay fp32, so it
        saves no VRAM. Casting here is what actually halves the footprint
        (measured on a RTX 4060 with SAM 1 ViT-B: 357.6MB -> 178.8MB weights,
        2770MB -> 1395MB peak, and ~3x faster). Call it *before* `.to(device)`
        so the fp32 copy never reaches the GPU.
        """
        try:
            import torch
        except ImportError:
            return model
        if self.device.startswith("cuda"):
            if self.precision == "fp16":
                return model.half()
            if self.precision in ("bf16", "bfloat16"):
                return model.to(torch.bfloat16)
            return model
        return self.quantize_cpu(model)

    def quantize_cpu(self, model):
        """Dynamic int8 for CPU-only machines (Linear layers)."""
        if self.precision != "int8" or self.device != "cpu":
            return model
        try:
            import torch
            from torch.ao.quantization import quantize_dynamic
            return quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        except (ImportError, RuntimeError):
            return model

    @staticmethod
    def _split_points(points):
        coords, labels = [], []
        for p in points or ():
            coords.append([float(p[0]), float(p[1])])
            labels.append(int(p[2]) if len(p) > 2 else 1)
        if not coords:
            return None, None
        return (np.asarray(coords, dtype=np.float32),
                np.asarray(labels, dtype=np.int32))

    @staticmethod
    def _sorted(masks: np.ndarray, scores: np.ndarray) -> SamResult:
        masks = np.asarray(masks)
        if masks.ndim == 2:
            masks = masks[None]
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if scores.size != len(masks):
            scores = np.ones((len(masks),), dtype=np.float32)
        order = np.argsort(-scores)
        return SamResult(masks=(masks[order] > 0), scores=scores[order])

    @property
    def loaded(self) -> bool:
        return self.model is not None or self.predictor is not None
