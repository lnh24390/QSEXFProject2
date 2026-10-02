"""Import existing detection / segmentation labels into the project.

Boxes come in as `shape_type="rectangle"` with `source="imported"` so the canvas
can draw them dashed and the "box -> segment" tool knows what to convert.
Original box corners are kept in `meta["src_bbox"]` so a conversion is reversible.

Everything is streamed one file at a time -- no dataset-wide buffering.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import (Callable, Dict, Iterable, List, Optional, Sequence,
                    Tuple)

from .annotation import ImageAnnotation, Shape
from .image_cache import read_image_size
from .project import IMAGE_EXTS, Project

Progress = Optional[Callable[[int, int, str], None]]


def rect_points(x1: float, y1: float, x2: float, y2: float):
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def _stem_map(project: Project) -> Dict[str, str]:
    """file stem (and full relative path) -> relative image path."""
    m: Dict[str, str] = {}
    for rel in project.images:
        p = Path(rel)
        m.setdefault(p.stem, rel)
        m.setdefault(p.name, rel)
        m.setdefault(rel, rel)
    return m


def _ensure_class(project: Project, name: str,
                  cache: Dict[str, int]) -> int:
    if name in cache:
        return cache[name]
    c = project.class_by_name(name) or project.add_class(name)
    cache[name] = c.id
    return c.id


def _signature(shape: Shape) -> tuple:
    """Identity of a shape for duplicate detection.

    Class + type + geometry rounded to whole pixels. Rounding absorbs the
    float noise of a normalise/denormalise round-trip, so re-importing the
    same YOLO file twice recognises its own shapes instead of doubling them.
    """
    pts = tuple((round(float(x)), round(float(y))) for x, y in shape.points)
    return (int(shape.class_id), shape.shape_type, pts)


def _add_shape(ann: ImageAnnotation, shape: Shape, seen: set,
               stats: Dict) -> bool:
    """Append unless an identical shape is already on this image."""
    sig = _signature(shape)
    if sig in seen:
        stats["duplicates"] = stats.get("duplicates", 0) + 1
        return False
    seen.add(sig)
    ann.shapes.append(shape)
    return True


def _seen_of(ann: ImageAnnotation) -> set:
    return {_signature(s) for s in ann.shapes}


def _ring(flat: List[float]) -> List[Tuple[float, float]]:
    return [(float(flat[k]), float(flat[k + 1]))
            for k in range(0, len(flat) - 1, 2)]


def _ring_area(pts: Sequence[Tuple[float, float]]) -> float:
    """Absolute shoelace area."""
    n = len(pts)
    if n < 3:
        return 0.0
    s = sum(pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1]
            for i in range(n))
    return abs(s) / 2.0


def _split_rings(rings: List[List[Tuple[float, float]]]):
    """Group COCO polygon rings into (outer, holes) pairs.

    A COCO `segmentation` list is a flat bag of rings: it may hold several
    disjoint parts of one occluded object *and* holes, with nothing marking
    which is which. Ordering by vertex count (the previous rule) turned a
    dense small part into the outer ring; ordering by area and testing
    containment gets both cases right -- a ring inside a bigger one is a hole,
    a ring beside it is its own shape.
    """
    ordered = sorted(rings, key=_ring_area, reverse=True)
    groups: List[Tuple[List, List]] = []
    for ring in ordered:
        cx = sum(p[0] for p in ring) / len(ring)
        cy = sum(p[1] for p in ring) / len(ring)
        for outer, holes in groups:
            if _point_in_ring(cx, cy, outer):
                holes.append(ring)
                break
        else:
            groups.append((ring, []))
    return groups


def _point_in_ring(x: float, y: float,
                   ring: Sequence[Tuple[float, float]]) -> bool:
    """Ray casting; no cv2 needed for a single point."""
    inside = False
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xin = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
            if x < xin:
                inside = not inside
    return inside


def _size_of(project: Project, rel: str, ann: ImageAnnotation) -> Tuple[int, int]:
    if ann.width and ann.height:
        return ann.width, ann.height
    w, h = read_image_size(project.abs_image(rel))
    ann.width, ann.height = w, h
    return w, h


# ---------------------------------------------------------------------------
# format detection
# ---------------------------------------------------------------------------
def detect_format(path: Path) -> str:
    """Guess the label format of a file or directory."""
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() == ".json":
            try:
                d = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return "unknown"
            if isinstance(d, dict) and "annotations" in d and "images" in d:
                return "coco"
            if isinstance(d, dict) and "shapes" in d:
                return "labelme"
        if path.suffix.lower() == ".xml":
            return "voc"
        if path.suffix.lower() == ".txt":
            return "yolo"
        return "unknown"

    if path.is_dir():
        if any(path.glob("*.xml")):
            return "voc"
        for j in path.glob("*.json"):
            return detect_format(j)
        if any(path.glob("*.txt")):
            return "yolo"
        # Split datasets keep labels one level down (labels/train/*.txt).
        # The importers already rglob, so detection must look as deep or it
        # reports "unknown" for a folder it could actually read.
        if any(path.rglob("*.xml")):
            return "voc"
        for j in sorted(path.rglob("*.json")):
            fmt = detect_format(j)
            if fmt != "unknown":
                return fmt
        for t in path.rglob("*.txt"):
            if t.name.lower() not in ("classes.txt", "obj.names", "train.txt",
                                      "val.txt", "test.txt"):
                return "yolo"
    return "unknown"


def _yolo_class_names(labels_dir: Path) -> List[str]:
    """Look for classes.txt / data.yaml next to the label folder."""
    for cand in (labels_dir / "classes.txt",
                 labels_dir.parent / "classes.txt",
                 labels_dir / "obj.names",
                 labels_dir.parent / "obj.names"):
        if cand.exists():
            return [ln.strip() for ln in
                    cand.read_text(encoding="utf-8").splitlines() if ln.strip()]
    for cand in (labels_dir.parent / "data.yaml", labels_dir.parent.parent / "data.yaml"):
        if cand.exists():
            names: List[str] = []
            in_names = False
            for ln in cand.read_text(encoding="utf-8").splitlines():
                st = ln.strip()
                if st.startswith("names:"):
                    in_names = True
                    rest = st.split(":", 1)[1].strip()
                    if rest.startswith("["):
                        return [n.strip().strip("'\"") for n in
                                rest.strip("[]").split(",") if n.strip()]
                    continue
                if in_names:
                    if st.startswith("-"):
                        names.append(st.lstrip("- ").strip().strip("'\""))
                    elif ":" in st and st.split(":")[0].strip().isdigit():
                        names.append(st.split(":", 1)[1].strip().strip("'\""))
                    elif st and not st.startswith("#"):
                        break
            if names:
                return names
    return []


# ---------------------------------------------------------------------------
# YOLO  (detection: cls cx cy w h   |   segmentation: cls x1 y1 x2 y2 ...)
# ---------------------------------------------------------------------------
def import_yolo(project: Project, labels_dir: Path,
                merge: bool = True, progress: Progress = None) -> Dict:
    labels_dir = Path(labels_dir)
    files = sorted(labels_dir.rglob("*.txt"))
    files = [f for f in files if f.name.lower() not in
             ("classes.txt", "obj.names", "train.txt", "val.txt", "test.txt")]
    names = _yolo_class_names(labels_dir)
    stems = _stem_map(project)
    cache: Dict[str, int] = {}
    stats = {"files": 0, "shapes": 0, "boxes": 0, "polys": 0, "skipped": 0,
             "duplicates": 0}

    for i, f in enumerate(files):
        rel = stems.get(f.stem)
        if rel is None:
            stats["skipped"] += 1
            continue
        ann = project.load_annotation(rel) if merge else ImageAnnotation(image_path=rel)
        seen = _seen_of(ann)
        W, H = _size_of(project, rel, ann)
        if not W or not H:
            stats["skipped"] += 1
            continue
        for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                ci = int(float(parts[0]))
                vals = [float(v) for v in parts[1:]]
            except ValueError:
                continue
            cname = names[ci] if ci < len(names) else f"class_{ci}"
            cid = _ensure_class(project, cname, cache)

            if len(vals) == 4:                       # detection box
                cx, cy, bw, bh = vals
                x1, y1 = (cx - bw / 2) * W, (cy - bh / 2) * H
                x2, y2 = (cx + bw / 2) * W, (cy + bh / 2) * H
                _added = _add_shape(ann, seen=seen, stats=stats, shape=Shape(
                    class_id=cid, points=rect_points(x1, y1, x2, y2),
                    shape_type="rectangle", source="imported",
                    meta={"src_bbox": [x1, y1, x2, y2], "format": "yolo"}))
                if _added:
                    stats["boxes"] += 1
            elif len(vals) >= 6 and len(vals) % 2 == 0:   # polygon
                pts = [(vals[k] * W, vals[k + 1] * H)
                       for k in range(0, len(vals), 2)]
                _added = _add_shape(ann, seen=seen, stats=stats, shape=Shape(
                    class_id=cid, points=pts, shape_type="polygon",
                    source="imported", meta={"format": "yolo-seg"}))
                if _added:
                    stats["polys"] += 1
            else:
                continue
            if _added:
                stats["shapes"] += 1
        project.save_annotation(ann)
        stats["files"] += 1
        if progress:
            progress(i + 1, len(files), rel)
    return stats


# ---------------------------------------------------------------------------
# COCO instances json
# ---------------------------------------------------------------------------
def import_coco(project: Project, json_path: Path,
                merge: bool = True, progress: Progress = None) -> Dict:
    d = json.loads(Path(json_path).read_text(encoding="utf-8"))
    cats = {c["id"]: c.get("name", f"class_{c['id']}") for c in d.get("categories", [])}
    imgs = {im["id"]: im for im in d.get("images", [])}
    stems = _stem_map(project)
    cache: Dict[str, int] = {}
    stats = {"files": 0, "shapes": 0, "boxes": 0, "polys": 0, "skipped": 0,
             "duplicates": 0}

    by_image: Dict[int, List[dict]] = {}
    for a in d.get("annotations", []):
        by_image.setdefault(a.get("image_id"), []).append(a)
    d = None                                     # release the big blob early

    total = len(by_image)
    for i, (img_id, anns) in enumerate(by_image.items()):
        im = imgs.get(img_id)
        if not im:
            stats["skipped"] += 1
            continue
        fname = Path(im.get("file_name", "")).name
        rel = stems.get(Path(fname).stem) or stems.get(fname)
        if rel is None:
            stats["skipped"] += 1
            continue
        ann = project.load_annotation(rel) if merge else ImageAnnotation(image_path=rel)
        seen = _seen_of(ann)
        ann.width = int(im.get("width") or ann.width or 0)
        ann.height = int(im.get("height") or ann.height or 0)
        if not ann.width or not ann.height:
            ann.width, ann.height = _size_of(project, rel, ann)

        for a in anns:
            cname = cats.get(a.get("category_id"), f"class_{a.get('category_id')}")
            cid = _ensure_class(project, cname, cache)
            seg = a.get("segmentation")
            made = False
            had_polygon = False        # source had one, added or duplicate
            if isinstance(seg, list) and seg and isinstance(seg[0], list):
                rings = [_ring(s) for s in seg if len(s) >= 6]
                if rings:
                    had_polygon = True
                    for outer, holes in _split_rings(rings):
                        _added = _add_shape(
                            ann, seen=seen, stats=stats,
                            shape=Shape(class_id=cid, points=outer, holes=holes,
                                        source="imported",
                                        meta={"format": "coco"}))
                        if _added:
                            stats["polys"] += 1
                            made = True
            if not had_polygon and a.get("bbox"):
                x, y, w, h = [float(v) for v in a["bbox"]]
                _added = _add_shape(ann, seen=seen, stats=stats, shape=Shape(
                    class_id=cid, points=rect_points(x, y, x + w, y + h),
                    shape_type="rectangle", source="imported",
                    meta={"src_bbox": [x, y, x + w, y + h], "format": "coco"}))
                if _added:
                    stats["boxes"] += 1
                    made = True
            if made:
                stats["shapes"] += 1
        project.save_annotation(ann)
        stats["files"] += 1
        if progress:
            progress(i + 1, total, rel)
    return stats


# ---------------------------------------------------------------------------
# Pascal VOC xml
# ---------------------------------------------------------------------------
def import_voc(project: Project, xml_dir: Path,
               merge: bool = True, progress: Progress = None) -> Dict:
    xml_dir = Path(xml_dir)
    files = [xml_dir] if xml_dir.is_file() else sorted(xml_dir.rglob("*.xml"))
    stems = _stem_map(project)
    cache: Dict[str, int] = {}
    stats = {"files": 0, "shapes": 0, "boxes": 0, "polys": 0, "skipped": 0,
             "duplicates": 0}

    for i, f in enumerate(files):
        try:
            root = ET.parse(f).getroot()
        except ET.ParseError:
            stats["skipped"] += 1
            continue
        fname = (root.findtext("filename") or f.stem)
        rel = stems.get(Path(fname).stem) or stems.get(f.stem)
        if rel is None:
            stats["skipped"] += 1
            continue
        ann = project.load_annotation(rel) if merge else ImageAnnotation(image_path=rel)
        seen = _seen_of(ann)
        size = root.find("size")
        if size is not None:
            ann.width = int(float(size.findtext("width") or 0)) or ann.width
            ann.height = int(float(size.findtext("height") or 0)) or ann.height
        if not ann.width or not ann.height:
            ann.width, ann.height = _size_of(project, rel, ann)

        for obj in root.findall("object"):
            name = (obj.findtext("name") or "object").strip()
            bb = obj.find("bndbox")
            if bb is None:
                continue
            try:
                x1 = float(bb.findtext("xmin")); y1 = float(bb.findtext("ymin"))
                x2 = float(bb.findtext("xmax")); y2 = float(bb.findtext("ymax"))
            except (TypeError, ValueError):
                continue
            cid = _ensure_class(project, name, cache)
            _added = _add_shape(ann, seen=seen, stats=stats, shape=Shape(
                class_id=cid, points=rect_points(x1, y1, x2, y2),
                shape_type="rectangle", source="imported",
                meta={"src_bbox": [x1, y1, x2, y2], "format": "voc"}))
            if _added:
                stats["boxes"] += 1
                stats["shapes"] += 1
        project.save_annotation(ann)
        stats["files"] += 1
        if progress:
            progress(i + 1, len(files), rel)
    return stats


# ---------------------------------------------------------------------------
# LabelMe json
# ---------------------------------------------------------------------------
def import_labelme(project: Project, src: Path,
                   merge: bool = True, progress: Progress = None) -> Dict:
    src = Path(src)
    files = [src] if src.is_file() else sorted(src.rglob("*.json"))
    stems = _stem_map(project)
    cache: Dict[str, int] = {}
    stats = {"files": 0, "shapes": 0, "boxes": 0, "polys": 0, "skipped": 0,
             "duplicates": 0}

    for i, f in enumerate(files):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            stats["skipped"] += 1
            continue
        if "shapes" not in d:
            stats["skipped"] += 1
            continue
        fname = Path(d.get("imagePath") or f.stem).name
        rel = stems.get(Path(fname).stem) or stems.get(f.stem)
        if rel is None:
            stats["skipped"] += 1
            continue
        ann = project.load_annotation(rel) if merge else ImageAnnotation(image_path=rel)
        seen = _seen_of(ann)
        ann.width = int(d.get("imageWidth") or ann.width or 0)
        ann.height = int(d.get("imageHeight") or ann.height or 0)
        if not ann.width or not ann.height:
            ann.width, ann.height = _size_of(project, rel, ann)

        for sh in d.get("shapes", []):
            name = str(sh.get("label", "object"))
            cid = _ensure_class(project, name, cache)
            pts = [(float(p[0]), float(p[1])) for p in sh.get("points", [])]
            st = sh.get("shape_type", "polygon")
            if st == "rectangle" and len(pts) == 2:
                (x1, y1), (x2, y2) = pts
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                _added = _add_shape(ann, seen=seen, stats=stats, shape=Shape(
                    class_id=cid, points=rect_points(x1, y1, x2, y2),
                    shape_type="rectangle", source="imported",
                    meta={"src_bbox": [x1, y1, x2, y2], "format": "labelme"}))
                if _added:
                    stats["boxes"] += 1
            elif len(pts) >= 3:
                _added = _add_shape(ann, seen=seen, stats=stats, shape=Shape(class_id=cid, points=pts,
                                        source="imported",
                                        meta={"format": "labelme"}))
                if _added:
                    stats["polys"] += 1
            else:
                continue
            if _added:
                stats["shapes"] += 1
        project.save_annotation(ann)
        stats["files"] += 1
        if progress:
            progress(i + 1, len(files), rel)
    return stats


IMPORTERS = {
    "yolo": import_yolo,
    "coco": import_coco,
    "voc": import_voc,
    "labelme": import_labelme,
}

FORMAT_LABELS = {
    "yolo": "YOLO txt (detection / segmentation)",
    "coco": "COCO instances json",
    "voc": "Pascal VOC xml",
    "labelme": "LabelMe json",
}


def _count_yolo(f: Path):
    """(boxes, polygons) in one YOLO txt."""
    b = pl = 0
    try:
        for line in f.read_text(encoding="utf-8").splitlines():
            n = len(line.split())
            if n == 5:
                b += 1
            elif n >= 7:
                pl += 1
    except OSError:
        pass
    return b, pl


def _count_labelme(f: Path):
    b = pl = 0
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0, 0
    for sh in d.get("shapes", []):
        if sh.get("shape_type") == "rectangle":
            b += 1
        else:
            pl += 1
    return b, pl


LABEL_EXTS = {".txt", ".xml", ".json"}


IMG_DIR_NAMES = ("images", "image", "imgs", "img", "JPEGImages")
LAB_DIR_NAMES = ("labels", "label", "annotations", "Annotations", "anns", "ann")


def _direct_images(d: Path) -> int:
    try:
        return sum(1 for f in d.iterdir()
                   if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
    except OSError:
        return 0


def _direct_labels(d: Path) -> int:
    skip = {"classes.txt", "obj.names", "train.txt", "val.txt", "test.txt",
            "data.yaml"}
    try:
        return sum(1 for f in d.iterdir()
                   if f.is_file() and f.suffix.lower() in LABEL_EXTS
                   and f.name.lower() not in skip)
    except OSError:
        return 0


def _label_dir_for(img_dir: Path, root: Path) -> Optional[Path]:
    """The label folder that belongs to one image folder.

    Handles the two shapes that actually occur: labels sitting beside the
    images in the same folder, and a mirrored tree where some path segment
    called `images` is spelled `labels` instead -- which is what makes
    `a/images/train` line up with `a/labels/train`.
    """
    if _direct_labels(img_dir):
        return img_dir
    parts = list(img_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() not in {n.lower() for n in IMG_DIR_NAMES}:
            continue
        for lab in LAB_DIR_NAMES:
            cand = Path(*parts[:i], lab, *parts[i + 1:])
            if cand.is_dir() and _direct_labels(cand):
                return cand
    # a sibling label folder for a flat `set/images` + `set/labels` pair
    for lab in LAB_DIR_NAMES:
        cand = img_dir.parent / lab
        if cand.is_dir() and _direct_labels(cand):
            return cand
    return None


def find_dataset_pairs(root: Path) -> List[Dict]:
    """Every image folder under `root`, each with the labels that go with it.

    A dropped folder often holds several subsets side by side
    (`set_a/images`, `set_b/images`, ...). Taking only the first pair would
    silently import a fraction of the data, so collect them all and let the
    caller show the total before anything is written.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    dirs = [root] + [d for d in sorted(root.rglob("*")) if d.is_dir()]
    pairs: List[Dict] = []
    for d in dirs:
        n = _direct_images(d)
        if not n:
            continue
        lab = _label_dir_for(d, root)
        pairs.append({"image_dir": d, "label_src": lab, "images": n,
                      "format": detect_format(lab) if lab else "unknown"})
    if not pairs:
        return []
    # A single COCO json can cover every image folder at once.
    if all(p["label_src"] is None for p in pairs):
        for j in sorted(root.rglob("*.json")):
            if detect_format(j) == "coco":
                for p in pairs:
                    p["label_src"] = j
                    p["format"] = "coco"
                break
    return pairs


def detect_dataset_layout(root: Path) -> Dict:
    """Work out where the images and the labels are inside one dropped folder.

    A dataset arrives in whatever shape its author left it in, so guess
    rather than demand: the common ones are a split `images/` + `labels/`
    pair, everything mixed in one folder, or images with a single COCO json
    beside them. Returns absolute paths plus a human note explaining the
    guess, because a wrong guess must be visible, not silent.
    """
    root = Path(root)
    out: Dict = {"root": root, "image_dir": None, "label_src": None,
                 "format": "unknown", "note": ""}
    if not root.is_dir():
        out["note"] = "폴더가 아닙니다."
        return out

    def imgs_in(d: Path) -> int:
        if not d.is_dir():
            return 0
        return sum(1 for f in d.rglob("*")
                   if f.is_file() and f.suffix.lower() in IMAGE_EXTS)

    def labels_in(d: Path) -> int:
        if not d.is_dir():
            return 0
        n = 0
        for f in d.rglob("*"):
            if not f.is_file() or f.suffix.lower() not in LABEL_EXTS:
                continue
            if f.name.lower() in ("classes.txt", "obj.names", "train.txt",
                                  "val.txt", "test.txt", "data.yaml"):
                continue
            n += 1
        return n

    # 1) the conventional split: images/ next to labels/ (or annotations/)
    img_names = ("images", "image", "imgs", "JPEGImages")
    lab_names = ("labels", "label", "annotations", "Annotations", "anns")
    img_dir = next((root / n for n in img_names if imgs_in(root / n)), None)
    lab_dir = next((root / n for n in lab_names if labels_in(root / n)), None)
    if img_dir is not None:
        out["image_dir"] = img_dir
        if lab_dir is not None:
            out["label_src"] = lab_dir
            out["note"] = f"{img_dir.name}/ + {lab_dir.name}/ 구조로 인식했습니다."
        else:
            out["note"] = f"{img_dir.name}/ 만 찾았습니다 (라벨 폴더 없음)."
    elif imgs_in(root):
        # 2) everything in one folder
        out["image_dir"] = root
        if labels_in(root):
            out["label_src"] = root
            out["note"] = "이미지와 라벨이 같은 폴더에 있습니다."
        else:
            out["note"] = "이미지만 찾았습니다 (라벨 없음)."
    else:
        out["note"] = "이미지를 찾지 못했습니다."
        return out

    # a lone COCO json anywhere under the root wins over a bare folder guess
    if out["label_src"] is None or detect_format(out["label_src"]) == "unknown":
        for j in sorted(root.rglob("*.json")):
            if detect_format(j) == "coco":
                out["label_src"] = j
                out["note"] += f"  COCO 파일을 찾았습니다: {j.name}"
                break

    if out["label_src"] is not None:
        out["format"] = detect_format(out["label_src"])
    return out


def _sample_boxes(fmt: str, f: Path, W: int, H: int, limit: int = 40):
    """Rough (x1, y1, x2, y2) list for a preview -- geometry only, no classes.

    Deliberately separate from the real importers: this runs before the user
    has agreed to anything, so it must not touch the project or care about
    class creation. Approximate is fine; it is drawn at thumbnail size.
    """
    out = []
    try:
        if fmt == "yolo":
            for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                v = [float(x) for x in parts[1:]]
                if len(v) == 4:
                    cx, cy, bw, bh = v
                    out.append(((cx - bw / 2) * W, (cy - bh / 2) * H,
                                (cx + bw / 2) * W, (cy + bh / 2) * H))
                else:
                    xs, ys = v[0::2], v[1::2]
                    if xs and ys:
                        out.append((min(xs) * W, min(ys) * H,
                                    max(xs) * W, max(ys) * H))
                if len(out) >= limit:
                    break
        elif fmt == "voc":
            for obj in ET.parse(f).getroot().findall("object"):
                bb = obj.find("bndbox")
                if bb is None:
                    continue
                out.append((float(bb.findtext("xmin") or 0), float(bb.findtext("ymin") or 0),
                            float(bb.findtext("xmax") or 0), float(bb.findtext("ymax") or 0)))
                if len(out) >= limit:
                    break
        elif fmt == "labelme":
            d = json.loads(f.read_text(encoding="utf-8"))
            for sh in d.get("shapes", []):
                pts = sh.get("points") or []
                xs = [q[0] for q in pts]
                ys = [q[1] for q in pts]
                if xs and ys:
                    out.append((min(xs), min(ys), max(xs), max(ys)))
                if len(out) >= limit:
                    break
    except (OSError, ValueError, ET.ParseError, json.JSONDecodeError):
        return []
    return out


def stem_collisions(project: Project, limit: int = 8) -> List[Tuple[str, List[str]]]:
    """Images whose file names collide once the folder is ignored.

    Labels are paired to images by stem, so two subsets that both contain
    `001.jpg` are indistinguishable to the importer: whichever image was
    scanned first wins, and the other subset's labels land on the wrong
    picture. Silent, and impossible to spot afterwards -- so it is reported
    before importing.
    """
    by_stem: Dict[str, List[str]] = {}
    for rel in project.images:
        by_stem.setdefault(Path(rel).stem, []).append(rel)
    dupes = [(k, v) for k, v in by_stem.items() if len(v) > 1]
    dupes.sort(key=lambda kv: -len(kv[1]))
    return dupes[:limit]


def analyze_many(project: Project, sources: Sequence[Path]) -> Dict:
    """Aggregate `analyze_import` over several label folders."""
    total = {"format": "", "images": len(project.images), "label_files": 0,
             "matched": 0, "boxes": 0, "polygons": 0, "unmatched": [],
             "matched_rels": set(), "problems": [], "sample": None,
             "sources": []}
    seen = set()
    for src in sources:
        key = str(Path(src).resolve())
        if key in seen:
            continue
        seen.add(key)
        a = analyze_import(project, src)
        total["sources"].append({"src": Path(src), "format": a["format"],
                                 "matched": a["matched"], "boxes": a["boxes"]})
        for k in ("label_files", "matched", "boxes", "polygons"):
            total[k] += a[k]
        total["matched_rels"] |= a["matched_rels"]
        total["problems"] += a["problems"]
        for u in a["unmatched"]:
            if len(total["unmatched"]) < 12:
                total["unmatched"].append(u)
        if total["sample"] is None:
            total["sample"] = a["sample"]
        if not total["format"]:
            total["format"] = a["format"]
        elif total["format"] != a["format"]:
            total["format"] = "mixed"
    return total


def analyze_import(project: Project, src: Path, fmt: str = "auto") -> Dict:
    """Dry run: what would `import_labels` actually match, and to what?

    Labels are paired to images by file stem, so the usual failure is a whole
    folder that silently matches nothing (different naming, wrong folder,
    images not scanned yet). Counting that up front turns a mystifying
    "0 shapes imported" into a number the user can act on *before* anything
    is written.
    """
    src = Path(src)
    fmt = detect_format(src) if fmt == "auto" else fmt
    stems = _stem_map(project)
    out: Dict = {"format": fmt, "images": len(project.images),
                 "label_files": 0, "matched": 0, "boxes": 0, "polygons": 0,
                 "unmatched": [], "matched_rels": set(), "problems": [],
                 "sample": None}
    if fmt not in IMPORTERS:
        out["problems"].append(f"라벨 포맷을 알 수 없습니다: {src}")
        return out
    if not project.images:
        out["problems"].append("프로젝트에 이미지가 없습니다. 이미지 폴더를 먼저 지정하세요.")

    if fmt == "coco":
        files = [src] if src.is_file() else sorted(src.glob("*.json"))
        for f in files:
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as e:
                out["problems"].append(f"{f.name}: 읽기 실패 ({e})")
                continue
            by_id = {im["id"]: im.get("file_name", "") for im in d.get("images", [])}
            out["label_files"] += len(by_id)
            hit = {}
            for iid, name in by_id.items():
                rel = stems.get(Path(name).stem) or stems.get(name)
                if rel:
                    hit[iid] = rel
                    out["matched_rels"].add(rel)
                elif len(out["unmatched"]) < 12:
                    out["unmatched"].append(name)
            out["matched"] += len(hit)
            first = next(iter(hit), None)
            for a in d.get("annotations", []):
                if a.get("image_id") not in hit:
                    continue
                if a.get("segmentation"):
                    out["polygons"] += 1
                else:
                    out["boxes"] += 1
            if first is not None and out["sample"] is None:
                boxes = []
                for a in d.get("annotations", []):
                    if a.get("image_id") != first:
                        continue
                    bb = a.get("bbox")
                    if bb and len(bb) == 4:
                        boxes.append((bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]))
                if boxes:
                    out["sample"] = {"rel": hit[first], "boxes": boxes[:40]}
        return out

    patterns = {"yolo": "*.txt", "voc": "*.xml", "labelme": "*.json"}
    files = ([src] if src.is_file()
             else sorted(src.rglob(patterns.get(fmt, "*"))))
    for f in files:
        if fmt == "yolo" and f.name in ("classes.txt", "data.yaml"):
            continue
        out["label_files"] += 1
        rel = stems.get(f.stem)
        if fmt == "voc" and rel is None:
            try:
                name = ET.parse(f).getroot().findtext("filename") or f.stem
                rel = stems.get(Path(name).stem)
            except ET.ParseError as e:
                out["problems"].append(f"{f.name}: XML 오류 ({e})")
                continue
        if rel is None:
            if len(out["unmatched"]) < 12:
                out["unmatched"].append(f.name)
            continue
        out["matched"] += 1
        out["matched_rels"].add(rel)
        if fmt == "yolo":
            b, pl = _count_yolo(f)
        elif fmt == "labelme":
            b, pl = _count_labelme(f)
        else:                                   # voc is boxes only
            try:
                b, pl = len(ET.parse(f).getroot().findall("object")), 0
            except ET.ParseError:
                b, pl = 0, 0
        out["boxes"] += b
        out["polygons"] += pl
        if out["sample"] is None and (b or pl):
            W, H = read_image_size(project.abs_image(rel))
            boxes = _sample_boxes(fmt, f, W or 1, H or 1)
            if boxes:
                out["sample"] = {"rel": rel, "boxes": boxes}
    return out


def describe_analysis(a: Dict) -> str:
    """The report shown before importing anything."""
    fmt = FORMAT_LABELS.get(a["format"], a["format"])
    no_label = a["images"] - len(a["matched_rels"])
    lines = [
        f"포맷: {fmt}",
        f"프로젝트 이미지: {a['images']}장",
        f"라벨 파일: {a['label_files']}개  ·  이미지와 매칭: {a['matched']}개",
        f"라벨 없는 이미지: {max(0, no_label)}장",
        "",
        f"변환 가능한 박스: {a['boxes']}개",
    ]
    if a["polygons"]:
        lines.append(f"이미 폴리곤인 도형: {a['polygons']}개 (변환 불필요)")
    if a["unmatched"]:
        n = len(a["unmatched"])
        lines += ["", f"짝을 못 찾은 라벨 (앞 {n}개):",
                  "  " + ", ".join(a["unmatched"])]
        lines.append("  → 파일 이름(확장자 제외)이 이미지와 같아야 합니다.")
    for src in a.get("sources", []):
        rel = src["src"].name
        lines.append(f"  · {rel}: {src['matched']}개 매칭, 박스 {src['boxes']}개")
    for p in a["problems"]:
        lines += ["", "⚠ " + p]
    if a["matched"] == 0:
        lines += ["", "매칭된 라벨이 없습니다. 가져와도 아무것도 추가되지 않습니다."]
    return "\n".join(lines)


def import_labels(project: Project, src: Path, fmt: str = "auto",
                  merge: bool = True, progress: Progress = None) -> Dict:
    fmt = detect_format(Path(src)) if fmt == "auto" else fmt
    fn = IMPORTERS.get(fmt)
    if fn is None:
        raise ValueError(f"지원하지 않는 라벨 포맷입니다: {fmt}")
    stats = fn(project, Path(src), merge=merge, progress=progress)
    stats["format"] = fmt
    project.save()
    return stats
