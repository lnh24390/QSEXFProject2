"""End-to-end smoke test (no SAM weights required)."""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dsl.core.annotation import ImageAnnotation, Shape
from dsl.core.exporters import export_dataset
from dsl.core.importers import import_labels
from dsl.core.project import Project

tmp = Path(tempfile.mkdtemp(prefix="dsl_smoke_"))
print("temp:", tmp)

# --- fake dataset -----------------------------------------------------------
img_dir = tmp / "proj" / "images"
img_dir.mkdir(parents=True)
for i in range(5):
    img = np.full((480, 640, 3), 40, np.uint8)
    cv2.rectangle(img, (100 + i * 20, 120), (300 + i * 20, 320), (60, 160, 220), -1)
    cv2.circle(img, (480, 240), 70, (200, 80, 90), -1)
    cv2.imwrite(str(img_dir / f"img_{i:02d}.jpg"), img)

proj = Project.create(tmp / "proj", "smoke", classes=["car", "person"])
assert len(proj.images) == 5, proj.images
print("project images:", proj.images)

# --- annotate 3 images ------------------------------------------------------
for i, rel in enumerate(proj.images[:3]):
    ann = proj.load_annotation(rel)
    ann.width, ann.height = 640, 480
    ann.shapes.append(Shape(class_id=0, points=[(100.0 + i * 20, 120.0), (300.0 + i * 20, 120.0),
                                                (300.0 + i * 20, 320.0), (100.0 + i * 20, 320.0)],
                            source="sam2", score=0.93))
    ann.shapes.append(Shape(class_id=1, points=[(410.0, 240.0), (480.0, 170.0),
                                                (550.0, 240.0), (480.0, 310.0)]))
    ann.verified = (i == 0)
    proj.save_annotation(ann)
proj.save()

st = proj.stats()
print("stats:", st)
assert st["labeled"] == 3 and st["shapes"] == 6 and st["verified"] == 1, st

# index survives a reload
proj2 = Project.load(tmp / "proj")
assert proj2.stats() == st, (proj2.stats(), st)
assert proj2.next_unlabeled(proj2.images[0]) == proj2.images[3]
print("index reload + next_unlabeled OK")

# --- exports ----------------------------------------------------------------
for fmt in ("yolo", "yolo-det", "coco", "masks"):
    out = tmp / "ds" / fmt
    r = export_dataset(proj2, fmt, out, val_ratio=0.34, image_mode="copy")
    print(f"export {fmt}:", r["counts"])
    assert sum(r["counts"].values()) == 3, r

y = tmp / "ds" / "yolo"
txts = list((y / "labels").rglob("*.txt"))
assert txts, "no yolo labels"
first = txts[0].read_text().strip().splitlines()
assert first and len(first[0].split()) >= 7, first
print("yolo label sample:", first[0][:70], "...")
assert (y / "data.yaml").exists()

# detection labels are derived from the same polygons: "cls cx cy w h",
# and the box must match the polygon's own bbox (DEVELOPMENT.md 6 -- nothing is
# stored twice, so a box can never go stale against its segment).
yd = tmp / "ds" / "yolo-det"
det_txts = sorted((yd / "labels").rglob("*.txt"))
assert det_txts, "no yolo-det labels"
_checked = 0
for _t in det_txts:
    for _line in _t.read_text().strip().splitlines():
        parts = _line.split()
        assert len(parts) == 5, f"detection line is not 'cls cx cy w h': {parts}"
        cx, cy, bw, bh = (float(v) for v in parts[1:])
        assert 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0, parts
        assert 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0, parts
        _checked += 1
assert _checked, "no detection lines"
# same source, same count of shapes
_seg_lines = sum(len(t.read_text().strip().splitlines()) for t in txts)
assert _checked == _seg_lines, (_checked, _seg_lines)

# compare one box against the annotation it came from
_rel = proj2.images[0]
_ann = proj2.load_annotation(_rel)
if _ann.shapes:
    _s = _ann.shapes[0]
    x1, y1, x2, y2 = _s.bbox()
    _exp = ((x1 + x2) / 2 / _ann.width, (y1 + y2) / 2 / _ann.height,
            (x2 - x1) / _ann.width, (y2 - y1) / _ann.height)
    _stem = _rel.replace("/", "__").rsplit(".", 1)[0]
    _hit = [t for t in det_txts if t.stem == _stem]
    assert _hit, (_stem, [t.stem for t in det_txts])
    _got = [float(v) for v in _hit[0].read_text().splitlines()[0].split()[1:]]
    assert all(abs(g - e) < 1e-4 for g, e in zip(_got, _exp)), (_got, _exp)
    print("yolo-det bbox matches Shape.bbox():",
          [round(v, 4) for v in _got])
print(f"yolo-det: {len(det_txts)} files, {_checked} boxes")

print("data.yaml:\n" + (y / "data.yaml").read_text())

coco = json.loads((tmp / "ds" / "coco" / "annotations" / "instances_train.json").read_text(encoding="utf-8"))
assert coco["annotations"] and coco["categories"], coco.keys()
print("coco anns:", len(coco["annotations"]), "cats:", [c["name"] for c in coco["categories"]])

mask_files = list((tmp / "ds" / "masks" / "masks").rglob("*.png"))
m = cv2.imread(str(mask_files[0]), cv2.IMREAD_UNCHANGED)
print("mask png uniques:", np.unique(m))
assert set(np.unique(m)) <= {0, 1, 2}

# --- import existing YOLO detection labels into a fresh project -------------
det_dir = tmp / "det_labels"
det_dir.mkdir()
(det_dir / "classes.txt").write_text("car\nperson\n", encoding="utf-8")
for i in range(5):
    (det_dir / f"img_{i:02d}.txt").write_text(
        "0 0.31 0.46 0.31 0.42\n1 0.75 0.5 0.22 0.29\n", encoding="utf-8")

proj3_root = tmp / "proj_import"
shutil.copytree(tmp / "proj" / "images", proj3_root / "images")
proj3 = Project.create(proj3_root, "import_test")
stats = import_labels(proj3, det_dir, "auto")
print("import stats:", stats)
assert stats["boxes"] == 10 and stats["files"] == 5, stats
ann = proj3.load_annotation(proj3.images[0])
assert all(s.shape_type == "rectangle" for s in ann.shapes)
assert ann.shapes[0].meta.get("src_bbox"), ann.shapes[0].meta
print("imported box:", [round(v) for v in ann.shapes[0].bbox()],
      "class:", proj3.class_by_id(ann.shapes[0].class_id).name)

# --- COCO round trip --------------------------------------------------------
proj4_root = tmp / "proj_coco"
shutil.copytree(tmp / "proj" / "images", proj4_root / "images")
proj4 = Project.create(proj4_root, "coco_test")
cstats = import_labels(proj4, tmp / "ds" / "coco" / "annotations" / "instances_train.json", "coco")
print("coco import:", cstats)
assert cstats["polys"] > 0, cstats

# --- regressions ------------------------------------------------------------
# 1) same stem, different extension must not share one label file
proj5 = Project.create(tmp / "proj_dup", "dup")
proj5.add_class("car")
blank = np.zeros((40, 40, 3), np.uint8)
for name in ("dup.jpg", "dup.png"):
    cv2.imwrite(str(proj5.image_root / name), blank)
proj5.scan_images()
proj5.save_annotation(ImageAnnotation(image_path="dup.jpg", width=40, height=40,
                                      shapes=[Shape(class_id=0, points=[[1, 1], [9, 1], [9, 9]])]))
proj5.save_annotation(ImageAnnotation(image_path="dup.png", width=40, height=40,
                                      shapes=[Shape(class_id=0, points=[[2, 2], [8, 2], [8, 8]]),
                                              Shape(class_id=0, points=[[3, 3], [7, 3], [7, 7]])]))
assert len(proj5.load_annotation("dup.jpg").shapes) == 1
assert len(proj5.load_annotation("dup.png").shapes) == 2
print("label path collision: none")

# 2) labels written by the pre-extension layout stay readable, then migrate
legacy = proj5.legacy_label_path("dup.jpg")
proj5.label_path("dup.jpg").unlink()
ImageAnnotation(image_path="dup.jpg", width=40, height=40,
                shapes=[Shape(class_id=0, points=[[0, 0], [5, 0], [5, 5]])]).save(legacy)
old = proj5.load_annotation("dup.jpg")
assert len(old.shapes) == 1, "legacy label unreadable"
proj5.save_annotation(old)
assert not legacy.exists() and proj5.label_path("dup.jpg").exists()
print("legacy label: read + migrated")

# 3) a shape with a hole must survive COCO export as RLE, area matching the mask
proj6 = Project.create(tmp / "proj_hole", "hole")
proj6.add_class("donut")
cv2.imwrite(str(proj6.image_root / "d.png"), np.zeros((20, 20, 3), np.uint8))
proj6.scan_images()
proj6.save_annotation(ImageAnnotation(
    image_path="d.png", width=20, height=20,
    shapes=[Shape(class_id=0, points=[[2, 2], [18, 2], [18, 18], [2, 18]],
                  holes=[[[7, 7], [13, 7], [13, 13], [7, 13]]])]))
proj6.save()
export_dataset(proj6, "coco", tmp / "ds_hole", val_ratio=0.0, test_ratio=0.0)
doc = json.loads((tmp / "ds_hole" / "annotations" / "instances_train.json")
                 .read_text(encoding="utf-8"))
seg = doc["annotations"][0]["segmentation"]
assert isinstance(seg, dict) and isinstance(seg["counts"], list), "hole not exported as RLE"
from dsl.train.trainer_maskrcnn import seg_to_mask
m = seg_to_mask(seg, 20, 20)
assert not m[10, 10], "hole got filled"
assert int(m.sum()) == int(doc["annotations"][0]["area"]), "area disagrees with mask"
print("coco holes: preserved as RLE, area matches")

# 4) deleting a class must not leave its shapes behind
proj6.add_class("extra")
proj6.save_annotation(ImageAnnotation(
    image_path="d.png", width=20, height=20,
    shapes=[Shape(class_id=0, points=[[2, 2], [8, 2], [8, 8]]),
            Shape(class_id=1, points=[[9, 9], [15, 9], [15, 15]])]))
assert proj6.remove_class(1) == 1
assert {s.class_id for s in proj6.load_annotation("d.png").shapes} == {0}
print("class delete: shapes removed from disk")

# 5) re-importing the same labels must not double every shape
before = sum(len(proj3.load_annotation(r).shapes) for r in proj3.images)
again = import_labels(proj3, det_dir, "auto")        # the very same source
after = sum(len(proj3.load_annotation(r).shapes) for r in proj3.images)
assert after == before, f"re-import duplicated shapes: {before} -> {after}"
assert again["duplicates"] > 0, again
print(f"re-import: {before} shapes unchanged, {again['duplicates']} duplicates skipped")

# 6) COCO rings: one inside another is a hole, one beside it is its own shape
from dsl.core.importers import _split_rings
_outer = [(0, 0), (100, 0), (100, 100), (0, 100)]
_hole = [(40, 40), (60, 40), (60, 60), (40, 60)]
_part = [(200, 0), (260, 0), (260, 60), (200, 60)]
_groups = _split_rings([_hole, _outer, _part])
assert len(_groups) == 2, _groups
assert len(_groups[0][1]) == 1 and len(_groups[1][1]) == 0, _groups
print("coco rings: hole vs separate part classified by containment")

# dry-run analysis: pairing images to labels *before* importing, so a folder
# that matches nothing is visible up front instead of as "0 shapes imported".
from dsl.core.importers import analyze_import, describe_analysis
_ai = tmp / "an"
_aimg = _ai / "imgs"; _aimg.mkdir(parents=True)
_alab = _ai / "labels"; _alab.mkdir(parents=True)
for _i in range(4):
    cv2.imwrite(str(_aimg / f"a_{_i}.jpg"), np.full((200, 300, 3), 40, np.uint8))
for _i in range(3):
    (_alab / f"a_{_i}.txt").write_text("0 0.5 0.5 0.4 0.4\n0 0.2 0.2 0.1 0.1\n",
                                       encoding="utf-8")
(_alab / "a_2.txt").write_text("0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n", encoding="utf-8")
(_alab / "nobody.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
(_alab / "classes.txt").write_text("car\n", encoding="utf-8")

_ap = Project.create(_ai / "proj", "an", image_dir=str(_aimg))
_ap.scan_images(); _ap.save()
_a = analyze_import(_ap, _alab)
assert _a["format"] == "yolo", _a["format"]
assert _a["images"] == 4, _a["images"]
assert _a["label_files"] == 4, _a["label_files"]      # classes.txt excluded
assert _a["matched"] == 3, _a["matched"]
assert _a["boxes"] == 4, _a["boxes"]                  # a_2 became a polygon
assert _a["polygons"] == 1, _a["polygons"]
assert _a["unmatched"] == ["nobody.txt"], _a["unmatched"]
_txt = describe_analysis(_a)
assert "nobody.txt" in _txt, _txt
print("analyze_import:", _a["matched"], "matched,", _a["boxes"], "boxes,",
      len(_a["unmatched"]), "orphan")

# the dry run must agree with what importing actually writes
_r = import_labels(_ap, _alab, "yolo", True)
assert _r["shapes"] == _a["boxes"] + _a["polygons"], (_r["shapes"], _a)
print("analysis matches the real import:", _r["shapes"], "shapes")

# a label folder that pairs with nothing is reported, not silently empty
_a2 = analyze_import(_ap, tmp / "ds" / "yolo" / "labels")
assert _a2["matched"] == 0 or _a2["unmatched"], _a2
print("mismatched folder flagged before import")

# a dropped dataset folder arrives in whatever shape its author left it in
from dsl.core.importers import detect_dataset_layout
from dsl.core.image_cache import imwrite_unicode as _iw
_ly = tmp / "layouts"

def _mkimg(q):
    q.parent.mkdir(parents=True, exist_ok=True)
    _iw(q, np.full((100, 120, 3), 40, np.uint8))

_a = _ly / "split"
for _i in range(3):
    _mkimg(_a / "images" / f"x{_i}.jpg")
    (_a / "labels").mkdir(parents=True, exist_ok=True)
    (_a / "labels" / f"x{_i}.txt").write_text("0 .5 .5 .2 .2\n", encoding="utf-8")
_b = _ly / "flat"
for _i in range(2):
    _mkimg(_b / f"y{_i}.png")
    (_b / f"y{_i}.txt").write_text("0 .5 .5 .2 .2\n", encoding="utf-8")
_c = _ly / "imgs_only"
_mkimg(_c / "z0.jpg")
_d = _ly / "yolo_split"
for _sp in ("train", "val"):
    _mkimg(_d / "images" / _sp / f"{_sp}0.jpg")
    (_d / "labels" / _sp).mkdir(parents=True, exist_ok=True)
    (_d / "labels" / _sp / f"{_sp}0.txt").write_text("0 .5 .5 .2 .2\n", encoding="utf-8")

_r = detect_dataset_layout(_a)
assert _r["image_dir"] == _a / "images" and _r["label_src"] == _a / "labels", _r
assert _r["format"] == "yolo", _r
_r = detect_dataset_layout(_b)
assert _r["image_dir"] == _b and _r["label_src"] == _b, _r
_r = detect_dataset_layout(_c)
assert _r["image_dir"] == _c and _r["label_src"] is None, _r
# labels one level down must still be recognised, not reported as unknown
_r = detect_dataset_layout(_d)
assert _r["format"] == "yolo", _r
assert detect_dataset_layout(_ly / "nope")["image_dir"] is None
print("dataset layout: split / flat / images-only / nested all detected")

# a dropped folder usually holds several subsets; every pair must be found,
# not just the first, or most of the data is silently left behind.
from dsl.core.importers import (find_dataset_pairs, analyze_many,
                                stem_collisions)
_ms = tmp / "multi"
for _st in ("set_a", "set_b", "set_c"):
    for _i in range(3):
        _mkimg(_ms / _st / "images" / f"{_st}_{_i}.jpg")
        (_ms / _st / "labels").mkdir(parents=True, exist_ok=True)
        (_ms / _st / "labels" / f"{_st}_{_i}.txt").write_text(
            "0 .5 .5 .4 .4\n", encoding="utf-8")
_pairs = find_dataset_pairs(_ms)
assert len(_pairs) == 3, [str(q["image_dir"]) for q in _pairs]
assert sum(q["images"] for q in _pairs) == 9, _pairs
assert all(q["label_src"] is not None and q["format"] == "yolo" for q in _pairs), _pairs
print("multi-subset:", len(_pairs), "pairs,", sum(q["images"] for q in _pairs), "images")

# opening the root recursively must see every subset, and the aggregate
# analysis must equal what importing all of them writes
_mp = Project.create(tmp / "multi_proj", "m", image_dir=str(_ms))
_mp.scan_images(); _mp.save()
assert len(_mp.images) == 9, _mp.images
_srcs = [q["label_src"] for q in _pairs]
_agg = analyze_many(_mp, _srcs)
assert _agg["matched"] == 9 and _agg["boxes"] == 9, _agg
assert len(_agg["sources"]) == 3, _agg["sources"]
_tot = 0
for _q in _srcs:
    _tot += import_labels(_mp, _q, "yolo", True)["shapes"]
assert _tot == _agg["boxes"], (_tot, _agg["boxes"])
print("aggregate analysis matches the multi-source import:", _tot, "shapes")

# file names that repeat across subsets are a real hazard: labels pair by
# stem, so the second subset would land on the first subset's images
_cl = tmp / "collide"
for _st in ("a", "b"):
    for _i in range(2):
        _mkimg(_cl / _st / "images" / f"{_i}.jpg")
_cp = Project.create(tmp / "collide_proj", "c", image_dir=str(_cl))
_cp.scan_images(); _cp.save()
_dups = stem_collisions(_cp)
assert _dups and all(len(v) == 2 for _k, v in _dups), _dups
assert not stem_collisions(_mp), stem_collisions(_mp)
print("stem collisions detected:", [k for k, _v in _dups])

# exporting at a smaller size must move the labels with the pixels
from dsl.core.image_cache import read_image_size as _ris, imread_unicode as _ir
_rz = tmp / "resize"
_rimg = _rz / "p" / "images"; _rimg.mkdir(parents=True)
_W, _H = 800, 600
for _i in range(2):
    _a = np.full((_H, _W, 3), 40, np.uint8)
    cv2.rectangle(_a, (200, 150), (600, 450), (60, 160, 220), -1)
    _iw(_rimg / f"r{_i}.jpg", _a)
_rp = Project.create(_rz / "p", "rz")
_rc = _rp.add_class("box"); _rp.scan_images(); _rp.save()
for _rel in _rp.images:
    _an = ImageAnnotation(image_path=_rel, width=_W, height=_H)
    _an.shapes.append(Shape(class_id=_rc.id,
                            points=[(200, 150), (600, 150), (600, 450), (200, 450)]))
    _rp.save_annotation(_an)
_rp.save_index()

_MS = 320                                  # 800x600 -> 320x240, scale 0.4
for _fmt in ("yolo", "yolo-det", "coco", "masks"):
    _out = _rz / "out" / _fmt
    export_dataset(_rp, _fmt, _out, val_ratio=0.0, test_ratio=0.0,
                   image_mode="copy", max_side=_MS)
    _im = sorted((_out / "images" / "train").glob("*"))[0]
    assert _ris(_im) == (320, 240), (_fmt, _ris(_im))
    if _fmt == "yolo":
        # normalised coordinates are resolution independent: unchanged
        _v = [float(x) for x in
              (_out / "labels" / "train" / (_im.stem + ".txt")).read_text().split()[1:]]
        assert abs(min(_v[0::2]) - 0.25) < 1e-3, _v
        assert abs(max(_v[1::2]) - 0.75) < 1e-3, _v
    if _fmt == "coco":
        _d = json.loads((_out / "annotations" / "instances_train.json")
                        .read_text(encoding="utf-8"))
        assert (_d["images"][0]["width"], _d["images"][0]["height"]) == (320, 240), _d["images"][0]
        _bb = _d["annotations"][0]["bbox"]
        # 200,150,400,300 scaled by 0.4
        assert all(abs(a - b) < 1.5 for a, b in zip(_bb, [80, 60, 160, 120])), _bb
    if _fmt == "masks":
        _m = _ir(_out / "masks" / "train" / (_im.stem + ".png"), cv2.IMREAD_UNCHANGED)
        assert _m.shape[:2] == (240, 320), _m.shape
        _ys, _xs = np.where(_m > 0)
        assert abs(_xs.min() - 80) <= 2 and abs(_ys.min() - 60) <= 2, (_xs.min(), _ys.min())
print("resize export: images 320x240, labels scaled in every format")

# an image already smaller than the target is left alone, not upscaled
_small = _rz / "small" / "images"; _small.mkdir(parents=True)
_iw(_small / "tiny.jpg", np.full((100, 120, 3), 40, np.uint8))
_sp = Project.create(_rz / "small", "sm")
_sc = _sp.add_class("box"); _sp.scan_images(); _sp.save()
_sa = ImageAnnotation(image_path=_sp.images[0], width=120, height=100)
_sa.shapes.append(Shape(class_id=_sc.id, points=[(10, 10), (60, 10), (60, 60), (10, 60)]))
_sp.save_annotation(_sa); _sp.save_index()
export_dataset(_sp, "coco", _rz / "out_small", val_ratio=0.0, image_mode="copy",
               max_side=_MS)
_sim = sorted((_rz / "out_small" / "images" / "train").glob("*"))[0]
assert _ris(_sim) == (120, 100), _ris(_sim)
_sd = json.loads((_rz / "out_small" / "annotations" / "instances_train.json")
                 .read_text(encoding="utf-8"))
assert all(abs(a - b) < 1e-6 for a, b in
           zip(_sd["annotations"][0]["bbox"], [10, 10, 50, 50])), _sd["annotations"][0]["bbox"]
print("small images are not upscaled and their labels are untouched")

# Shape.scaled must not mutate the original
_orig = Shape(class_id=0, points=[(10.0, 20.0)], holes=[[(1.0, 2.0)]])
_cp = _orig.scaled(2.0, 3.0)
assert _orig.points == [(10.0, 20.0)] and _orig.holes == [[(1.0, 2.0)]], _orig
assert _cp.points == [(20.0, 60.0)] and _cp.holes == [[(2.0, 6.0)]], _cp
assert _cp.uid == _orig.uid
print("Shape.scaled copies instead of mutating")

# auto-labelling needs to find the checkpoints training produced
from dsl.train.predictor import find_checkpoints
_rr = tmp / "runs_probe"
for _mk, _run in (("yolo11-seg", "a"), ("yolo11-seg", "b"), ("yolov8-seg", "c")):
    _w = _rr / _mk / _run / "weights"
    _w.mkdir(parents=True)
    (_w / "best.pt").write_bytes(b"x")
    (_w / "last.pt").write_bytes(b"x")          # only best.pt is offered
(_rr / "yolo11-seg" / "d").mkdir(parents=True)  # a run that never finished
_cks = find_checkpoints(_rr)
assert len(_cks) == 3, [c.label for c in _cks]
assert all(c.path.name == "best.pt" for c in _cks), _cks
assert {c.label for c in _cks} == {"yolo11-seg / a", "yolo11-seg / b",
                                   "yolov8-seg / c"}, [c.label for c in _cks]
assert [c.mtime for c in _cks] == sorted((c.mtime for c in _cks), reverse=True)
assert find_checkpoints(tmp / "nope") == []
print("checkpoint discovery:", len(_cks), "runs, newest first")

# --- split mode: ratio vs the dataset's own folders -------------------------
from dsl.core.exporters import folder_split as _fsplit

_fs_cases = {"images/test/a.jpg": "test", "train/images/a.jpg": "train",
             "valid/x.jpg": "val", "VALIDATION/x.jpg": "val",
             "testing/x.jpg": "test", "a/b/c.jpg": None}
for _k, _want in _fs_cases.items():
    assert _fsplit(_k) == _want, (_k, _fsplit(_k), _want)
print("folder_split: reads the split out of", len(_fs_cases), "path shapes")

_sp2 = Project.create(tmp / "proj_split", "split")
_sp2.add_class("car")
_plan = {"train": 6, "val": 3, "test": 2}
for _s, _n in _plan.items():
    for _i in range(_n):
        _rel = f"images/{_s}/{_s}_{_i}.png"
        _f = _sp2.image_root / _rel
        _f.parent.mkdir(parents=True, exist_ok=True)
        _iw(_f, np.full((60, 80, 3), 128, np.uint8))
_sp2.scan_images()
for _rel in _sp2.images:
    _sp2.save_annotation(ImageAnnotation(
        image_path=_rel, width=80, height=60,
        shapes=[Shape(class_id=0, points=[(5, 5), (40, 5), (40, 40), (5, 40)])]))
_sp2.save()

_rf = export_dataset(_sp2, "yolo", tmp / "ds_folder", split_mode="folder")
assert _rf["counts"] == _plan, _rf["counts"]
assert _rf["unsplit"] == 0, _rf["unsplit"]
for _s, _n in _plan.items():
    assert len(list((tmp / "ds_folder" / "images" / _s).glob("*"))) == _n
    assert len(list((tmp / "ds_folder" / "labels" / _s).glob("*.txt"))) == _n
_yaml = (tmp / "ds_folder" / "data.yaml").read_text(encoding="utf-8")
assert "test: images/test" in _yaml and "val: images/val" in _yaml, _yaml
print("folder mode: original train/val/test preserved exactly", _rf["counts"])

# the same project split by ratio must NOT reproduce the folder layout --
# that difference is the whole reason the option exists
_rr = export_dataset(_sp2, "yolo", tmp / "ds_ratio", split_mode="ratio",
                     val_ratio=0.2, test_ratio=0.2)
assert sum(_rr["counts"].values()) == sum(_plan.values())
assert _rr["counts"]["test"] > 0, "test_ratio produced no test split"
print("ratio mode: test split honoured", _rr["counts"])

for _fmt in ("coco", "masks"):
    _r = export_dataset(_sp2, _fmt, tmp / f"ds_folder_{_fmt}", split_mode="folder")
    assert _r["counts"] == _plan, (_fmt, _r["counts"])
print("folder mode applies to coco and masks too")

# a project with no split folders falls back to train and reports it
_np2 = Project.create(tmp / "proj_nosplit", "nosplit")
_np2.add_class("car")
for _i in range(3):
    _f = _np2.image_root / f"plain_{_i}.png"
    _iw(_f, np.full((60, 80, 3), 128, np.uint8))
_np2.scan_images()
for _rel in _np2.images:
    _np2.save_annotation(ImageAnnotation(
        image_path=_rel, width=80, height=60,
        shapes=[Shape(class_id=0, points=[(5, 5), (40, 5), (40, 40), (5, 40)])]))
_np2.save()
_rn = export_dataset(_np2, "yolo", tmp / "ds_nosplit", split_mode="folder")
assert _rn["unsplit"] == 3, _rn["unsplit"]
assert _rn["counts"]["train"] == 3 and _rn["counts"]["val"] == 0
# val is empty, so data.yaml must not point at a folder that was never made
assert "val: images/train" in (tmp / "ds_nosplit" / "data.yaml").read_text(
    encoding="utf-8")
print("no split folders: 3 fall back to train, reported, data.yaml still valid")


print("\nSMOKE OK ->", tmp)
