"""Dataset exporters: one label JSON -> whatever each trainer needs.

Formats
  yolo   : YOLO segmentation txt + data.yaml   (YOLOv8/9/11/12/26, RT-DETR)
  coco   : COCO instances json                 (Mask R-CNN, RT-DETR)
  masks  : semantic label PNG + palette PNG    (SegFormer, ConvNeXt-UPerNet)

Images are processed strictly one at a time (DEVELOPMENT.md 3.2).
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .annotation import ImageAnnotation
from .image_cache import imread_unicode, imwrite_unicode, read_image_size
from .mask_utils import (mask_to_uncompressed_rle, polygons_to_mask,
                         shapes_to_label_map)
from .project import Project

Progress = Optional[Callable[[int, int, str], None]]


SPLITS = ("train", "val", "test")

#: Folder names that mean a split, normalised to our three names. A dataset
#: downloaded from anywhere uses one of these spellings.
SPLIT_ALIASES = {
    "train": "train", "training": "train", "trainval": "train",
    "val": "val", "valid": "val", "validation": "val", "eval": "val",
    "test": "test", "testing": "test",
}


def deterministic_split(rel: str, val_ratio: float = 0.2,
                        test_ratio: float = 0.0, seed: int = 0) -> str:
    """Stable per-file split: re-exporting never reshuffles the dataset."""
    h = hashlib.md5(f"{seed}:{rel}".encode("utf-8")).hexdigest()
    v = int(h[:8], 16) / 0xFFFFFFFF
    if v < test_ratio:
        return "test"
    if v < test_ratio + val_ratio:
        return "val"
    return "train"


def folder_split(rel: str) -> Optional[str]:
    """The split a path already declares, or None if it declares none.

    Reads the folder segments from the file outwards, so both of the common
    layouts land on the same answer:

        images/val/x.jpg   -> "val"
        val/images/x.jpg   -> "val"

    Nearest-first matters for the second one: `images` is not a split name,
    but a dataset nested as `train/images/val_crops/x.jpg` would otherwise
    pick the wrong folder.
    """
    for seg in reversed(Path(rel).parent.as_posix().split("/")):
        s = SPLIT_ALIASES.get(seg.strip().lower())
        if s:
            return s
    return None


def make_splitter(split_mode: str = "ratio", val_ratio: float = 0.2,
                  test_ratio: float = 0.0, seed: int = 0,
                  fallback: str = "train") -> Callable[[str], Tuple[str, bool]]:
    """rel -> (split, matched).

    `matched` is False only in folder mode, for a path that names no split;
    those files go to `fallback` and the exporters count them so the caller
    can say how many could not be placed instead of silently burying them
    in train.
    """
    if split_mode == "folder":
        def by_folder(rel: str) -> Tuple[str, bool]:
            s = folder_split(rel)
            return (s, True) if s else (fallback, False)
        return by_folder

    def by_ratio(rel: str) -> Tuple[str, bool]:
        return deterministic_split(rel, val_ratio, test_ratio, seed), True
    return by_ratio


def _place_image(src: Path, dst: Path, mode: str = "copy",
                 max_side: int = 0) -> Tuple[float, float]:
    """Put one image in the dataset. Returns the (sx, sy) actually applied.

    `max_side` shrinks the long edge to at most that many pixels; 0 keeps the
    original. Images already smaller are left alone -- upscaling invents
    detail the model would then learn from.

    The ratio is derived from the rounded output size rather than the
    requested one, so label coordinates land exactly on the pixels that were
    written. Resizing needs a real decode/encode, so symlinking is only
    possible at the original size.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return 1.0, 1.0
    if max_side and max_side > 0:
        img = imread_unicode(src)
        if img is not None:
            h, w = img.shape[:2]
            longest = max(w, h)
            if longest > max_side:
                scale = max_side / float(longest)
                nw = max(1, int(round(w * scale)))
                nh = max(1, int(round(h * scale)))
                img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
                if imwrite_unicode(dst, img):
                    return nw / float(w), nh / float(h)
    if mode == "symlink":
        try:
            dst.symlink_to(src.resolve())
            return 1.0, 1.0
        except (OSError, NotImplementedError):
            pass          # Windows without developer mode -> fall back to copy
    shutil.copy2(src, dst)
    return 1.0, 1.0


def _flat_name(rel: str) -> str:
    """a/b/c.jpg -> a__b__c.jpg so sub-folders can share one flat split dir."""
    return rel.replace("/", "__")


def _iter_labeled(project: Project, only_verified: bool = False):
    for rel in project.images:
        if not project.is_labeled(rel):
            continue
        if only_verified and not project.is_verified(rel):
            continue
        yield rel


# ---------------------------------------------------------------------------
# YOLO segmentation
# ---------------------------------------------------------------------------
def export_yolo(project: Project, out_dir: Path,
                val_ratio: float = 0.2, test_ratio: float = 0.0,
                image_mode: str = "copy", only_verified: bool = False,
                seed: int = 0, progress: Progress = None,
                task: str = "segment", max_side: int = 0,
                split_mode: str = "ratio") -> Dict:
    """YOLO dataset. `task` picks the label line shape:

        segment : "cls x1 y1 x2 y2 ..."   (polygon, normalised)
        detect  : "cls cx cy w h"         (box, normalised)

    Both come from the same polygons -- a detection label is the polygon's
    bounding box, computed here rather than stored, so editing a vertex can
    never leave a stale box behind (DEVELOPMENT.md 6).
    """
    out_dir = Path(out_dir)
    cmap = project.contiguous_index()
    names = [c.name for c in project.classes]
    counts = {"train": 0, "val": 0, "test": 0}
    n_shapes = 0

    splitter = make_splitter(split_mode, val_ratio, test_ratio, seed)
    unsplit = 0

    rels = list(_iter_labeled(project, only_verified))
    for i, rel in enumerate(rels):
        split, matched = splitter(rel)
        unsplit += 0 if matched else 1
        ann = project.load_annotation(rel)
        W = ann.width or 0
        H = ann.height or 0
        if not W or not H:
            W, H = read_image_size(project.abs_image(rel))
        if not W or not H:
            continue

        flat = _flat_name(rel)
        # YOLO coordinates are normalised, so a resize leaves every label
        # line byte-for-byte identical -- only the pixels change.
        _place_image(project.abs_image(rel),
                     out_dir / "images" / split / flat, image_mode, max_side)

        lines: List[str] = []
        for s in ann.shapes:
            if s.class_id not in cmap or len(s.points) < 3:
                continue
            if task == "detect":
                x1, y1, x2, y2 = s.bbox()
                x1, x2 = max(x1, 0.0), min(x2, float(W))
                y1, y2 = max(y1, 0.0), min(y2, float(H))
                bw, bh = (x2 - x1) / W, (y2 - y1) / H
                if bw <= 0 or bh <= 0:
                    continue            # degenerate after clipping
                cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
                lines.append(f"{cmap[s.class_id]} {cx:.6f} {cy:.6f} "
                             f"{bw:.6f} {bh:.6f}")
            else:
                coords = []
                for x, y in s.points:
                    coords.append(f"{min(max(x / W, 0.0), 1.0):.6f}")
                    coords.append(f"{min(max(y / H, 0.0), 1.0):.6f}")
                lines.append(str(cmap[s.class_id]) + " " + " ".join(coords))
            n_shapes += 1

        lp = out_dir / "labels" / split / (Path(flat).stem + ".txt")
        lp.parent.mkdir(parents=True, exist_ok=True)
        lp.write_text("\n".join(lines), encoding="utf-8")
        counts[split] += 1
        if progress:
            progress(i + 1, len(rels), rel)

    # ultralytics fails to start if `val:` names a folder that was never
    # created, which folder mode can produce for a dataset with no val split.
    yaml_lines = [
        f"path: {out_dir.resolve().as_posix()}",
        "train: images/train",
        f"val: images/{'val' if counts['val'] else 'train'}",
    ]
    if counts["test"]:
        yaml_lines.append("test: images/test")
    yaml_lines.append("names:")
    yaml_lines += [f"  {i}: {n}" for i, n in enumerate(names)]
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    return {"format": "yolo-det" if task == "detect" else "yolo",
            "task": task, "root": str(out_dir), "data_yaml": str(data_yaml),
            "counts": counts, "shapes": n_shapes, "classes": names,
            "split_mode": split_mode, "unsplit": unsplit}


# ---------------------------------------------------------------------------
# COCO instances
# ---------------------------------------------------------------------------
def export_coco(project: Project, out_dir: Path,
                val_ratio: float = 0.2, test_ratio: float = 0.0,
                image_mode: str = "copy", only_verified: bool = False,
                seed: int = 0, progress: Progress = None,
                max_side: int = 0, split_mode: str = "ratio") -> Dict:
    out_dir = Path(out_dir)
    cmap = project.contiguous_index()
    categories = [{"id": cmap[c.id] + 1, "name": c.name, "supercategory": "object"}
                  for c in project.classes]

    docs = {s: {"info": {"description": project.name}, "licenses": [],
                "images": [], "annotations": [], "categories": categories}
            for s in ("train", "val", "test")}
    ids = {"img": 1, "ann": 1}
    counts = {"train": 0, "val": 0, "test": 0}

    splitter = make_splitter(split_mode, val_ratio, test_ratio, seed)
    unsplit = 0

    rels = list(_iter_labeled(project, only_verified))
    for i, rel in enumerate(rels):
        split, matched = splitter(rel)
        unsplit += 0 if matched else 1
        ann = project.load_annotation(rel)
        W, H = ann.width, ann.height
        if not W or not H:
            W, H = read_image_size(project.abs_image(rel))
        if not W or not H:
            continue

        flat = _flat_name(rel)
        sx, sy = _place_image(project.abs_image(rel),
                              out_dir / "images" / split / flat,
                              image_mode, max_side)
        # COCO stores absolute pixels, so a resize has to be carried through
        # every coordinate *and* the recorded image size.
        if (sx, sy) != (1.0, 1.0):
            ann = ann.scaled(sx, sy)
            W, H = max(1, int(round(W * sx))), max(1, int(round(H * sy)))

        img_id = ids["img"]; ids["img"] += 1
        docs[split]["images"].append(
            {"id": img_id, "file_name": flat, "width": W, "height": H})

        for s in ann.shapes:
            if s.class_id not in cmap or len(s.points) < 3:
                continue
            holes = [h for h in s.holes if len(h) >= 3]
            if holes:
                # A COCO polygon list is a union of parts, so a hole written as
                # another polygon comes back *filled*. Holed shapes go out as
                # RLE, which is the only polygon-free way COCO can express them.
                m = polygons_to_mask(s.points, holes, H, W)
                seg = mask_to_uncompressed_rle(m)
                area = float(np.count_nonzero(m))
            else:
                seg = [[c for p in s.points
                        for c in (round(p[0], 2), round(p[1], 2))]]
                area = s.area()
            x1, y1, x2, y2 = s.bbox()
            docs[split]["annotations"].append({
                "id": ids["ann"], "image_id": img_id,
                "category_id": cmap[s.class_id] + 1,
                "segmentation": seg,
                "bbox": [round(x1, 2), round(y1, 2),
                         round(x2 - x1, 2), round(y2 - y1, 2)],
                "area": round(area, 2), "iscrowd": 0,
            })
            ids["ann"] += 1
        counts[split] += 1
        if progress:
            progress(i + 1, len(rels), rel)

    ann_dir = out_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for split, doc in docs.items():
        if not doc["images"]:
            continue
        p = ann_dir / f"instances_{split}.json"
        p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        written[split] = str(p)

    return {"format": "coco", "root": str(out_dir), "counts": counts,
            "annotations": written, "split_mode": split_mode,
            "unsplit": unsplit,
            "classes": [c["name"] for c in categories]}


# ---------------------------------------------------------------------------
# Semantic masks
# ---------------------------------------------------------------------------
def export_masks(project: Project, out_dir: Path,
                 val_ratio: float = 0.2, test_ratio: float = 0.0,
                 image_mode: str = "copy", only_verified: bool = False,
                 seed: int = 0, progress: Progress = None,
                 max_side: int = 0, split_mode: str = "ratio") -> Dict:
    """index 0 = background, class k -> index k+1."""
    out_dir = Path(out_dir)
    cmap = {c.id: i + 1 for i, c in enumerate(project.classes)}
    names = ["background"] + [c.name for c in project.classes]
    counts = {"train": 0, "val": 0, "test": 0}

    splitter = make_splitter(split_mode, val_ratio, test_ratio, seed)
    unsplit = 0

    rels = list(_iter_labeled(project, only_verified))
    for i, rel in enumerate(rels):
        split, matched = splitter(rel)
        unsplit += 0 if matched else 1
        ann = project.load_annotation(rel)
        W, H = ann.width, ann.height
        if not W or not H:
            W, H = read_image_size(project.abs_image(rel))
        if not W or not H:
            continue

        flat = _flat_name(rel)
        sx, sy = _place_image(project.abs_image(rel),
                              out_dir / "images" / split / flat,
                              image_mode, max_side)
        # Rasterise at full size, then shrink with nearest-neighbour: scaling
        # the polygons first would round each vertex independently and leave
        # gaps between neighbouring regions.
        lab = shapes_to_label_map(ann.shapes, H, W, cmap)
        if (sx, sy) != (1.0, 1.0):
            lab = cv2.resize(lab,
                             (max(1, int(round(W * sx))), max(1, int(round(H * sy)))),
                             interpolation=cv2.INTER_NEAREST)
        imwrite_unicode(out_dir / "masks" / split / (Path(flat).stem + ".png"), lab)
        del lab                                    # free before the next image
        counts[split] += 1
        if progress:
            progress(i + 1, len(rels), rel)

    meta = {"classes": names, "ignore_index": 255, "counts": counts,
            "split_mode": split_mode}
    (out_dir / "dataset.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"format": "masks", "root": str(out_dir), "counts": counts,
            "classes": names, "split_mode": split_mode, "unsplit": unsplit}


def export_yolo_det(project: Project, out_dir: Path, **kw) -> Dict:
    """YOLO detection labels derived from the segment polygons."""
    kw.pop("task", None)
    return export_yolo(project, out_dir, task="detect", **kw)


EXPORTERS = {"yolo": export_yolo, "yolo-det": export_yolo_det,
             "coco": export_coco, "masks": export_masks}

EXPORT_LABELS = {
    "yolo": "YOLO segmentation (YOLOv8/9/11/26-seg)",
    "yolo-det": "YOLO detection / bbox (YOLO12, RT-DETR)",
    "coco": "COCO instances (Mask R-CNN)",
    "masks": "Semantic mask PNG (SegFormer, ConvNeXt-UPerNet)",
}


def export_dataset(project: Project, fmt: str, out_dir: Path, **kw) -> Dict:
    fn = EXPORTERS.get(fmt)
    if fn is None:
        raise ValueError(f"알 수 없는 내보내기 포맷: {fmt}")
    return fn(project, Path(out_dir), **kw)
