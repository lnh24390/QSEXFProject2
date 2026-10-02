"""Bounded image caches.

Rule (DEVELOPMENT.md 3.2): only the current image plus a couple of neighbours may
live in RAM. Everything goes through an LRU with a hard byte budget.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np


def imread_unicode(path: Path, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """cv2.imread that survives non-ASCII Windows paths."""
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    img = cv2.imdecode(buf, flags)
    return img


def imwrite_unicode(path: Path, img: np.ndarray) -> bool:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def read_image_size(path: Path) -> Tuple[int, int]:
    """(width, height) without decoding the whole file when possible."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size
    except (ImportError, OSError, ValueError):
        img = imread_unicode(path)
        if img is None:
            return (0, 0)
        return (img.shape[1], img.shape[0])


class LRUImageCache:
    """Thread-safe LRU keyed by path, bounded by total decoded bytes."""

    def __init__(self, budget_bytes: int = 1024 * 1024 * 1024):
        self.budget = int(budget_bytes)
        self._data: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._bytes = 0
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    # ---- stats ------------------------------------------------------------
    @property
    def nbytes(self) -> int:
        return self._bytes

    def __len__(self) -> int:
        return len(self._data)

    def info(self) -> str:
        return f"{len(self._data)}장 / {self._bytes / 1048576:.0f}MB " \
               f"(예산 {self.budget / 1048576:.0f}MB)"

    # ---- core -------------------------------------------------------------
    def get(self, key: str) -> Optional[np.ndarray]:
        with self._lock:
            img = self._data.get(key)
            if img is None:
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return img

    def put(self, key: str, img: np.ndarray) -> None:
        if img is None:
            return
        with self._lock:
            if key in self._data:
                self._bytes -= self._data[key].nbytes
                del self._data[key]
            self._data[key] = img
            self._bytes += img.nbytes
            self._data.move_to_end(key)
            self._evict()

    def evict(self, key: str) -> None:
        """Explicitly drop one image (called right after a label is saved)."""
        with self._lock:
            img = self._data.pop(key, None)
            if img is not None:
                self._bytes -= img.nbytes

    def keep_only(self, keys) -> None:
        """Drop everything except `keys` -- used when jumping across the list."""
        keep = set(keys)
        with self._lock:
            for k in [k for k in self._data if k not in keep]:
                self._bytes -= self._data[k].nbytes
                del self._data[k]

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._bytes = 0

    def _evict(self) -> None:
        # always keep at least one image, otherwise the current view breaks
        while self._bytes > self.budget and len(self._data) > 1:
            _k, img = self._data.popitem(last=False)
            self._bytes -= img.nbytes

    # ---- loading ----------------------------------------------------------
    def load(self, path: Path, max_side: int = 0) -> Optional[np.ndarray]:
        """Return RGB uint8 array, decoding only on a miss."""
        key = str(path)
        img = self.get(key)
        if img is not None:
            return img
        bgr = imread_unicode(Path(path))
        if bgr is None:
            return None
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        elif bgr.shape[2] == 4:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)
        if max_side and max(bgr.shape[:2]) > max_side:
            scale = max_side / max(bgr.shape[:2])
            bgr = cv2.resize(bgr, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.put(key, rgb)
        return rgb


class LRUCountCache:
    """Tiny count-bounded LRU used for thumbnails (QPixmap or ndarray)."""

    def __init__(self, capacity: int = 512):
        self.capacity = int(capacity)
        self._data: "OrderedDict[str, object]" = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: str):
        with self._lock:
            v = self._data.get(key)
            if v is not None:
                self._data.move_to_end(key)
            return v

    def put(self, key: str, value) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.capacity:
                self._data.popitem(last=False)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._data

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        return len(self._data)
