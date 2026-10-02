"""Conversions between binary masks and polygons."""
from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]


def smooth_ring(cnt: np.ndarray, radius: int) -> np.ndarray:
    """Round off the 1px staircase findContours traces along a mask edge.

    A mask boundary is axis-aligned by construction, so the raw contour is a
    staircase of single-pixel steps. Simplification alone does not fix that:
    Douglas-Peucker drops points but keeps the remaining corners square, so
    the outline still reads as ragged. A moving average over the ring pulls
    the steps onto the diagonal the eye expects.

    The ring is closed, so the window wraps -- filtering it as an open
    sequence would leave a visible kink where the last point meets the first.
    No points are removed here; that is simplification's job, afterwards.
    """
    n = len(cnt)
    k = int(radius)
    if k < 1 or n < 8:
        return cnt
    k = min(k, n // 4)
    win = 2 * k + 1
    pts = cnt.reshape(-1, 2).astype(np.float32)
    pad = np.vstack([pts[-k:], pts, pts[:k]])
    kernel = np.ones(win, np.float32) / win
    xs = np.convolve(pad[:, 0], kernel, mode="valid")
    ys = np.convolve(pad[:, 1], kernel, mode="valid")
    return np.stack([xs, ys], axis=1).reshape(-1, 1, 2)


def mask_to_polygons(mask: np.ndarray,
                     epsilon_ratio: float = 0.002,
                     min_area: float = 24.0,
                     keep_largest_only: bool = False,
                     with_holes: bool = True,
                     smooth: int = 0):
    """Binary mask -> list of (outer_ring, [hole_rings]).

    epsilon_ratio: Douglas-Peucker tolerance as a fraction of the contour
    perimeter. 0 disables simplification (dense, exact contours).
    smooth: moving-average radius applied to each ring before simplification,
    in contour points. 0 keeps the raw pixel staircase.
    """
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if m.max() == 0:
        return []

    mode = cv2.RETR_CCOMP if with_holes else cv2.RETR_EXTERNAL
    contours, hierarchy = cv2.findContours(m, mode, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []

    def simplify(cnt: np.ndarray) -> List[Point]:
        # Smooth first: running Douglas-Peucker on the staircase would lock
        # in the square corners it happens to keep.
        if smooth > 0:
            cnt = smooth_ring(cnt, smooth)
        if epsilon_ratio > 0:
            eps = epsilon_ratio * cv2.arcLength(cnt, True)
            cnt = cv2.approxPolyDP(cnt, eps, True)
        return [(float(p[0][0]), float(p[0][1])) for p in cnt]

    results = []
    if hierarchy is None:
        hierarchy = np.full((1, len(contours), 4), -1, dtype=int)
    hier = hierarchy[0]

    for i, cnt in enumerate(contours):
        if with_holes and hier[i][3] != -1:
            continue                       # a hole; handled by its parent
        if cv2.contourArea(cnt) < min_area:
            continue
        outer = simplify(cnt)
        if len(outer) < 3:
            continue
        holes: List[List[Point]] = []
        if with_holes:
            child = hier[i][2]
            while child != -1:
                if cv2.contourArea(contours[child]) >= min_area:
                    ring = simplify(contours[child])
                    if len(ring) >= 3:
                        holes.append(ring)
                child = hier[child][0]
        results.append((outer, holes))

    if keep_largest_only and results:
        results.sort(key=lambda r: cv2.contourArea(
            np.array(r[0], dtype=np.float32).reshape(-1, 1, 2)), reverse=True)
        results = results[:1]
    return results


def polygons_to_mask(outer: Sequence[Point],
                     holes: Sequence[Sequence[Point]],
                     height: int, width: int) -> np.ndarray:
    """Rasterise one shape into a uint8 {0,1} mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(outer) < 3:
        return mask
    cv2.fillPoly(mask, [np.array(outer, dtype=np.int32)], 1)
    for h in holes or []:
        if len(h) >= 3:
            cv2.fillPoly(mask, [np.array(h, dtype=np.int32)], 0)
    return mask


def mask_to_uncompressed_rle(mask: np.ndarray) -> dict:
    """COCO uncompressed RLE: {"size": [h, w], "counts": [...]}.

    Column-major run lengths starting with a run of zeros, which is what
    `pycocotools.mask.frPyObjects` expects. Used for shapes with holes, since
    a COCO polygon list is a *union* of parts and cannot express a hole.
    """
    h, w = mask.shape[:2]
    flat = np.asarray(mask, dtype=bool).ravel(order="F")
    # run boundaries, with an implicit leading zero-run
    changes = np.flatnonzero(np.diff(flat)) + 1
    bounds = np.concatenate(([0], changes, [flat.size]))
    counts = np.diff(bounds).tolist()
    if flat.size and flat[0]:
        counts.insert(0, 0)          # RLE always starts with a zero-run
    return {"size": [int(h), int(w)], "counts": [int(c) for c in counts]}


def shapes_to_label_map(shapes, height: int, width: int,
                        class_to_index=None, background: int = 0) -> np.ndarray:
    """Semantic label map for SegFormer / ConvNeXt-UPerNet.

    Later shapes paint over earlier ones, matching canvas z-order.
    """
    n_cls = len(class_to_index) if class_to_index else 255
    dtype = np.uint8 if n_cls < 255 else np.uint16
    out = np.full((height, width), background, dtype=dtype)
    for s in shapes:
        if class_to_index:
            idx = class_to_index.get(s.class_id, 0)
        else:
            idx = s.class_id + 1
        m = polygons_to_mask(s.points, s.holes, height, width)
        out[m > 0] = idx
    return out


def bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def clip_points(points, width: int, height: int):
    return [(min(max(0.0, x), width - 1.0), min(max(0.0, y), height - 1.0))
            for x, y in points]


def dilate_erode(mask: np.ndarray, px: int) -> np.ndarray:
    """Grow (px>0) or shrink (px<0) a mask; used by the +/- mask refine keys."""
    if px == 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (abs(px) * 2 + 1,) * 2)
    m = (mask > 0).astype(np.uint8)
    return cv2.dilate(m, k) if px > 0 else cv2.erode(m, k)
