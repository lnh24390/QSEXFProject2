"""Offscreen GUI smoke test: build the window, drive it, check memory rules."""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import Qt, QTimer
from PySide6.QtCore import QModelIndex
from PySide6.QtWidgets import (QApplication, QScrollArea as _QScrollArea,
                               QToolBar)

from dsl.core.project import Project
from dsl.core.settings import Settings
from dsl.core.annotation import Shape
from dsl.ui.canvas import BOX2SEG, POLYGON, SAM_POINT, SELECT
from dsl.ui.main_window import MainWindow
from dsl.ui.theme import STYLESHEET

tmp = Path(tempfile.mkdtemp(prefix="dsl_gui_"))
img_dir = tmp / "proj" / "images"
img_dir.mkdir(parents=True)
for i in range(12):
    img = np.full((720, 1280, 3), 30, np.uint8)
    cv2.rectangle(img, (200, 150), (600, 500), (70, 150, 230), -1)
    cv2.putText(img, str(i), (60, 90), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cv2.imwrite(str(img_dir / f"s_{i:02d}.png"), img)

proj = Project.create(tmp / "proj", "guitest", classes=["car", "person", "sign"])
print("images:", len(proj.images))

# an imported-looking box, to exercise the box->segment path
ann = proj.load_annotation(proj.images[0])
ann.width, ann.height = 1280, 720
ann.shapes.append(Shape(class_id=0, points=[(200.0, 150.0), (600.0, 150.0),
                                            (600.0, 500.0), (200.0, 500.0)],
                        shape_type="rectangle", source="imported",
                        meta={"src_bbox": [200, 150, 600, 500]}))
proj.save_annotation(ann)
proj.save()

app = QApplication(sys.argv)
app.setStyle("Fusion")
app.setStyleSheet(STYLESHEET)

st = Settings()
st.config_path = tmp / "settings.json"   # never touch the real configs/
st.weights_dir = str(tmp / "weights")
st.image_cache_mb = 64          # tiny budget on purpose
st.prefetch_neighbors = 1
st.recent_projects = []
st.auto_save = True

win = MainWindow(st)
win.resize(1500, 900)
win.show()
win.load_project(tmp / "proj")
app.processEvents()

assert win.project is not None
assert win.current_rel == proj.images[0], win.current_rel
print("opened:", win.current_rel, "shapes:", len(win.annotation.shapes))
assert len(win.canvas.shape_items) == 1
assert win.canvas.shape_items[0].shape_data.shape_type == "rectangle"

# class selection
win.class_panel.select_index(1)
app.processEvents()
assert win.current_class_id == 1, win.current_class_id
print("class ->", win.project.class_by_id(win.current_class_id).name)

# manual polygon
win.set_tool(POLYGON)
win._on_manual_polygon([(700, 100), (900, 120), (880, 300), (690, 280)])
app.processEvents()
assert len(win.annotation.shapes) == 2
assert len(win.canvas.shape_items) == 2
print("after manual polygon:", len(win.annotation.shapes), "shapes")

# undo / redo
win.undo(); app.processEvents()
assert len(win.annotation.shapes) == 1, len(win.annotation.shapes)
win.redo(); app.processEvents()
assert len(win.annotation.shapes) == 2
print("undo/redo OK")

# autosave wrote the label file
lp = win.project.label_path(win.current_rel)
assert lp.exists(), lp
print("label file:", lp.name, lp.stat().st_size, "bytes")

# browse: cache must stay bounded to the prefetch window
for i in range(1, 12):
    win.file_panel.step(1)
    app.processEvents()
print("browsed to:", win.current_rel)
print("cache:", win.cache.info())
assert len(win.cache) <= 3, len(win.cache)          # current + 1 each side
assert win.cache.nbytes <= win.cache.budget + 4 * 1024 * 1024

# thumbnails: only rows the view painted were decoded
QTimer.singleShot(400, app.quit)
app.exec()
print("thumbnail cache items:", len(win.file_panel.model.cache),
      "/ images:", len(win.project.images))
assert len(win.file_panel.model.cache) <= len(win.project.images)

# verify + auto-advance + eviction
st.auto_advance = True
st.hide_completed = False
win.file_panel.select(win.project.images[3])
app.processEvents()
rel = win.current_rel
win.toggle_verified()
app.processEvents()
assert win.project.is_verified(rel), "verified flag not stored"
print("verified:", rel, "-> now at", win.current_rel)

# tools switch cleanly
for t in (SELECT, SAM_POINT, BOX2SEG, POLYGON):
    win.set_tool(t)
    app.processEvents()
print("tool switching OK")

# box->segment without a model must warn, not crash
win.file_panel.select(win.project.images[0])
app.processEvents()
item = [i for i in win.canvas.shape_items
        if i.shape_data.shape_type == "rectangle"]
if item:
    win._on_convert_request(item[0])
    app.processEvents()
    print("convert-without-model handled:", win.statusBar().currentMessage()[:60])

# --- regressions -------------------------------------------------------------
# geometry edits must be undoable (vertex drag / delete / insert used to change
# the shape without ever taking a snapshot, so Ctrl+Z could not get it back)
from PySide6.QtCore import QPointF
if win.canvas.shape_items:
    _it = win.canvas.shape_items[0]
    _orig = [list(map(float, p)) for p in _it.shape_data.points]
    _it.set_editing(True)
    _it.notify_edit_begin()                       # what mousePressEvent does
    _it.handles[0].setPos(QPointF(_orig[0][0] + 7, _orig[0][1] + 7))
    app.processEvents()
    assert [list(map(float, p)) for p in _it.shape_data.points] != _orig
    win.undo()
    app.processEvents()
    _now = [list(map(float, p)) for p in win.annotation.shapes[0].points]
    assert _now == _orig, f"vertex drag not undone: {_now} != {_orig}"

    _it = win.canvas.shape_items[0]
    _n = len(_it.shape_data.points)
    _it.set_editing(True)
    _it.notify_edit_begin()
    _it.remove_vertex(0)
    win.undo()
    app.processEvents()
    assert len(win.annotation.shapes[0].points) == _n, "vertex delete not undone"
    print("vertex edit undo OK (drag + delete)")

    _before = len(win._undo)
    for _ in range(3):
        win._snapshot()
    assert len(win._undo) - _before <= 1, "no-op snapshots piling up"
    print("duplicate snapshots suppressed")

# selecting on the canvas must reach the shape table (canvas emits a ShapeItem,
# the table speaks Shape -- these used to be wired straight together)
if win.canvas.shape_items:
    win.canvas.shape_items[0].setSelected(True)
    app.processEvents()
    rows = sorted({i.row() for i in win.shape_panel.table.selectedIndexes()})
    assert rows == [0], f"canvas selection did not reach the shape table: {rows}"
    print("canvas selection -> shape table OK")

# the SAM load signal must carry precision through to the worker slot
import inspect
from dsl.ui.workers import SamWorker
_sig = MainWindow.samLoad.__str__()
_params = list(inspect.signature(SamWorker.load_model).parameters)[1:]
assert _sig.count("QString") == len(_params), (
    f"samLoad{_sig} does not match load_model{_params}")
print("samLoad arity matches worker slot:", _params)

# training log must never be delivered on the pump thread
import threading
from PySide6.QtCore import QThread
_gui = QThread.currentThread()
_threads = []
win.trainLine.connect(lambda s: _threads.append(QThread.currentThread() is _gui))
_t = threading.Thread(target=lambda: [win.trainLine.emit(f"l{i}") for i in range(3)])
_t.start(); _t.join()
for _ in range(20):
    app.processEvents()
assert _threads and all(_threads), f"training log touched widgets off-thread: {_threads}"
print("training log marshalled to GUI thread:", len(_threads), "lines")

# tools the loaded backend cannot serve must be greyed out (SAM 3 has no
# point prompting on some builds, and a dead tool is worse than a disabled one)
win.set_tool(SAM_POINT)
win._on_sam_capabilities(False, True, True)
assert not win.tool_actions[SAM_POINT].isEnabled(), "point tool stayed enabled"
assert win.canvas.tool == SELECT, "did not leave the unsupported tool"
win._on_sam_capabilities(True, True, True)
assert win.tool_actions[SAM_POINT].isEnabled()
print("prompt tools follow backend capabilities")

# export straight from the window's project
from dsl.core.exporters import export_dataset
r = export_dataset(win.project, "yolo", tmp / "out")
print("export from GUI project:", r["counts"])

# an image folder anywhere on disk, opened without a project (Ctrl+Shift+O).
# The folder itself must stay untouched -- it may be read-only or shared.
import dsl.core.settings as _settings
_ext = tmp / "elsewhere" / "shoot_2026"
_ext.mkdir(parents=True)
for _i in range(4):
    cv2.imwrite(str(_ext / f"ext_{_i}.png"), np.full((120, 160, 3), 90, np.uint8))
(_ext / "notes.txt").write_text("not an image", encoding="utf-8")
_before = sorted(q.name for q in _ext.iterdir())

_app_dir = tmp / "appdir"
_app_dir.mkdir()
_orig_app_dir = _settings.APP_DIR
_settings.APP_DIR = _app_dir            # keep the test out of the real projects/
try:
    assert win.load_image_folder(_ext), "load_image_folder returned False"
    assert win.project is not None
    assert len(win.project.images) == 4, win.project.images
    assert Path(win.project.image_dir).resolve() == _ext.resolve(), win.project.image_dir
    _projects = (_app_dir / "projects").resolve()
    assert Path(win.project.root).resolve().is_relative_to(_projects), win.project.root
    assert sorted(q.name for q in _ext.iterdir()) == _before, "image folder was written to"
    print("external folder:", len(win.project.images), "images, labels ->",
          Path(win.project.root).name)

    _root_first = Path(win.project.root)
    assert win.load_image_folder(_ext)
    assert Path(win.project.root) == _root_first, "reopen created a second project"
    assert len(list((_app_dir / "projects").iterdir())) == 1
    print("reopen reuses the same project")

    _empty = tmp / "elsewhere" / "empty"
    _empty.mkdir()
    assert win.load_image_folder(_empty) is False, "empty folder was accepted"
    print("folder without images rejected")
finally:
    _settings.APP_DIR = _orig_app_dir

# the window must be usable at 1280x720: nothing may be clipped out of
# reach, which is what happens when a tall dock has no scroll area.
from PySide6.QtWidgets import QScrollArea as _QScrollArea
assert (win.minimumWidth(), win.minimumHeight()) == (1280, 720),     (win.minimumWidth(), win.minimumHeight())
_lay_min = win.layout().minimumSize()
assert _lay_min.width() <= 1280 and _lay_min.height() <= 720,     f"layout cannot shrink to 1280x720: {_lay_min.width()}x{_lay_min.height()}"

win.resize(1280, 720)
app.processEvents()
assert (win.width(), win.height()) == (1280, 720), (win.width(), win.height())
win.resize(640, 480)                     # below the minimum -> must clamp
app.processEvents()
assert (win.width(), win.height()) == (1280, 720), (win.width(), win.height())
print("window clamps to 1280x720 minimum")

for _d in (win.dock_files, win.dock_class, win.dock_shape, win.dock_tools):
    _inner = _d.widget()
    assert isinstance(_inner, _QScrollArea), (_d.windowTitle(), type(_inner))
    assert _inner.widgetResizable(), _d.windowTitle()
    assert _inner.widget() is not None, _d.windowTitle()
print("all docks scroll and follow their width")

# the tools dock is the tall one; at 720 its content must overflow *and*
# be reachable through the scroll area rather than clipped away.
win.resize(1280, 720)
app.processEvents()
_sc = win.dock_tools.widget()
assert _sc.widget().sizeHint().height() > _sc.viewport().height(),     "tools panel unexpectedly fits; the scroll test proves nothing"
assert _sc.verticalScrollBarPolicy() != Qt.ScrollBarAlwaysOff
_vbar = _sc.verticalScrollBar()
assert _vbar.maximum() > 0, f"no scroll range: max={_vbar.maximum()}"
print("tall dock scrolls:", _vbar.maximum(), "px of reachable overflow")

# boxes and segments must accumulate together on one image, and the stroke
# has to stay visible however the fill is set.
from dsl.ui.canvas import RECT as _RECT
win.set_tool(_RECT)
app.processEvents()
_W, _H = win.annotation.width, win.annotation.height
# fractions of the open image, whatever size it happens to be
for _fx, _fy in ((0.05, 0.05), (0.40, 0.10), (0.55, 0.55)):
    win.canvas.boxFinished.emit([_fx * _W, _fy * _H,
                                 (_fx + 0.30) * _W, (_fy + 0.30) * _H])
_boxes = [sh for sh in win.annotation.shapes if sh.shape_type == "rectangle"]
assert len(_boxes) >= 3, [sh.shape_type for sh in win.annotation.shapes]
assert len(win.canvas.shape_items) == len(win.annotation.shapes)
print("multiple boxes:", len(_boxes), "rectangles on one image")

# a box dragged past the edge is clipped, never stored outside the image
win.canvas.boxFinished.emit([-80, -80, 9000, 9000])
_x1, _y1, _x2, _y2 = win.annotation.shapes[-1].bbox()
assert _x1 >= 0 and _y1 >= 0, (_x1, _y1)
assert _x2 <= win.annotation.width and _y2 <= win.annotation.height, (_x2, _y2)
print("box clipped to image:", tuple(round(v) for v in (_x1, _y1, _x2, _y2)))

# segments still land alongside them
_m = np.zeros((_H, _W), np.uint8)
_m[int(_H * 0.1):int(_H * 0.5), int(_W * 0.1):int(_W * 0.5)] = 1
win._add_masks_as_shapes(_m[None], np.array([0.9]))
_kinds = {t: [sh.shape_type for sh in win.annotation.shapes].count(t)
          for t in ("rectangle", "polygon")}
assert _kinds["rectangle"] >= 4 and _kinds["polygon"] >= 1, _kinds
print("boxes + segments coexist:", _kinds)

# outline-only must clear the fill but keep the stroke on *both* kinds
win._apply_fill_alpha(0)
app.processEvents()
for _it in win.canvas.shape_items:
    assert _it.brush().color().alpha() == 0, (_it.shape_data.shape_type,
                                              _it.brush().color().alpha())
    assert _it.pen().widthF() > 0, _it.shape_data.shape_type
    assert _it.pen().color().alpha() > 0, _it.shape_data.shape_type
print("outline-only: fill cleared, stroke kept on", len(win.canvas.shape_items), "shapes")

win.canvas.set_outline_width(4.0)
app.processEvents()
assert all(abs(i.pen().widthF() - 4.0) < 0.01 for i in win.canvas.shape_items
           if not i.isSelected()), [i.pen().widthF() for i in win.canvas.shape_items]
win._apply_fill_alpha(70)
win.canvas.set_outline_width(1.8)
print("outline width is adjustable")

win.set_tool(SELECT)
app.processEvents()

# assigning a class must never fail silently -- the two preconditions
# (a selection, a current class) are what made this look like a dead button.
win.shape_panel.table.clearSelection()
app.processEvents()
assert not win.shape_panel.btn_assign.isEnabled(), "assign enabled with no selection"
assert not win.shape_panel.btn_del.isEnabled(), "delete enabled with no selection"

win.assign_class([])
assert win.statusBar().currentMessage(), "empty selection gave no feedback"
print("no selection -> button disabled and explained")

win.shape_panel.table.selectRow(0)
app.processEvents()
assert win.shape_panel.btn_assign.isEnabled(), "assign stayed disabled with a selection"

_saved_cls = win.current_class_id
win.current_class_id = None
win.assign_class(win.shape_panel.selected_shapes())
assert win.statusBar().currentMessage(), "missing class gave no feedback"
win.current_class_id = _saved_cls
print("no current class -> explained instead of ignored")

# and the assignment itself works from the table selection
_target = win.annotation.shapes[0]
_other = [c for c in win.project.classes if c.id != _target.class_id]
if _other:
    win.current_class_id = _other[0].id
    win.shape_panel.table.selectRow(0)
    app.processEvents()
    win.shape_panel.btn_assign.click()
    app.processEvents()
    assert win.annotation.shapes[0].class_id == _other[0].id,         (win.annotation.shapes[0].class_id, _other[0].id)
    print("class applied to selected shape:", _other[0].name)

# segment <-> box must round-trip without loss: the polygon is stashed in
# meta on the way out, so coming back does not need SAM and cannot drift.
_mm = np.zeros((win.annotation.height, win.annotation.width), np.uint8)
cv2.circle(_mm, (win.annotation.width // 2, win.annotation.height // 2),
           min(win.annotation.width, win.annotation.height) // 4, 1, -1)
cv2.circle(_mm, (win.annotation.width // 2, win.annotation.height // 2),
           min(win.annotation.width, win.annotation.height) // 10, 0, -1)
win._add_masks_as_shapes(_mm[None], np.array([0.9]))
_seg = win.annotation.shapes[-1]
assert _seg.shape_type == "polygon", _seg.shape_type
_orig_pts = [tuple(q) for q in _seg.points]
_orig_holes = [[tuple(q) for q in h] for h in _seg.holes]
assert len(_orig_pts) > 4, len(_orig_pts)

for _it in win.canvas.shape_items:
    _it.setSelected(_it.shape_data is _seg)
win.shapes_to_boxes()
assert _seg.shape_type == "rectangle", _seg.shape_type
assert len(_seg.points) == 4, _seg.points
assert "src_polygon" in _seg.meta, _seg.meta.keys()
print("segment -> box:", len(_orig_pts), "points stashed in meta")

for _it in win.canvas.shape_items:
    _it.setSelected(_it.shape_data is _seg)
win.boxes_to_shapes()
assert _seg.shape_type == "polygon", _seg.shape_type
assert [tuple(q) for q in _seg.points] == _orig_pts, "polygon changed on the way back"
assert [[tuple(q) for q in h] for h in _seg.holes] == _orig_holes, "holes lost"
assert "src_polygon" not in _seg.meta, "stash not cleaned up"
print("box -> segment: restored exactly,", len(_seg.holes), "hole(s) intact")

# a box with no stashed polygon needs SAM; without it, say so and change nothing
win.canvas.boxFinished.emit([0.1 * _W, 0.1 * _H, 0.4 * _W, 0.4 * _H])
_plain = win.annotation.shapes[-1]
for _it in win.canvas.shape_items:
    _it.setSelected(_it.shape_data is _plain)
win.boxes_to_shapes()
assert _plain.shape_type == "rectangle", "converted without SAM or a stash"
assert win.statusBar().currentMessage(), "no explanation for the unconvertible box"
print("plain box without SAM: left alone and explained")

# the box-label tool and both conversions are reachable from the toolbar
_tb = win.findChildren(QToolBar)[0]
_labels = [a.text() for a in _tb.actions() if a.text()]
# The toolbar is wider than the window and Qt puts the tail behind its ">>"
# extension button; what matters is that the tools come *before* the long
# file actions, so the ones used constantly stay on screen.
for _need in ("박스 라벨", "세그→박스", "박스→세그"):
    assert any(_need in t for t in _labels), (_need, _labels)
_tools_end = max(i for i, t in enumerate(_labels) if "박스→세그" in t)
_file_start = min([i for i, t in enumerate(_labels)
                   if "가져오기" in t or "내보내기" in t] or [len(_labels)])
assert _tools_end < _file_start, (_labels[_tools_end], _labels[_file_start])
print("toolbar:", len(_labels), "buttons; tools precede the file actions")

# USER_GUIDE and the in-app F1 text both list shortcuts; a shortcut that
# only exists in the docs sends the reader hunting for a key that does
# nothing. Check every documented Ctrl+... against the real actions.
import re as _re
from dsl.ui.main_window import SHORTCUTS_TEXT as _F1

def _norm(t):
    return t.replace(" ", "").lower()

_real = set()
for _a in win.findChildren(type(win.a_save)):
    _ks = _a.shortcut().toString()
    if _ks:
        _real.add(_norm(_ks))
assert _real, "no shortcuts found on the window"

for _doc_name, _text in (("USER_GUIDE.md",
                          (ROOT / "docs" / "USER_GUIDE.md").read_text(encoding="utf-8")),
                         ("SHORTCUTS_TEXT", _F1)):
    _found = set()
    for _m in _re.finditer(r"Ctrl\+(?:Shift\+)?[A-Za-z0-9=]|Ctrl\+-", _text):
        _found.add(_norm(_m.group(0)))
    _missing = sorted(k for k in _found if k not in _real)
    assert not _missing, f"{_doc_name} documents shortcuts the app lacks: {_missing}"
    print(f"{_doc_name}: {len(_found)} shortcuts, all present in the app")

# read-only viewing: labels land in a temp project, never under projects/,
# and are gone once the session ends. The image folder is untouched either way.
from dsl.core.image_cache import imwrite_unicode as _imw
_ro_src = tmp / "readonly_src"
_ro_src.mkdir()
for _i in range(3):
    _imw(_ro_src / f"r_{_i}.jpg", np.full((160, 220, 3), 55, np.uint8))
_ro_before = sorted(q.name for q in _ro_src.iterdir())

_settings_mod = __import__("dsl.core.settings", fromlist=["x"])
_ro_app = tmp / "ro_app"; _ro_app.mkdir()
_keep_app = _settings_mod.APP_DIR
_settings_mod.APP_DIR = _ro_app
try:
    assert win.load_image_folder(_ro_src), "normal open failed"
    _normal_root = Path(win.project.root)
    assert _normal_root.is_relative_to(_ro_app / "projects"), _normal_root

    assert win.load_image_folder(_ro_src, read_only=True), "read-only open failed"
    _ro_root = Path(win.project.root)
    assert not _ro_root.is_relative_to(_ro_app), _ro_root
    assert win._temp_projects, "temp project not tracked for cleanup"
    assert _ro_root.exists()
    assert len(win.project.images) == 3, win.project.images
    print("read-only session:", len(win.project.images), "images, project outside APP_DIR")

    _tmp_parent = win._temp_projects[-1]
finally:
    _settings_mod.APP_DIR = _keep_app

assert sorted(q.name for q in _ro_src.iterdir()) == _ro_before, "image folder written to"

# the import dialog grows with its report and preview; it must scroll
# rather than push its own buttons off the bottom of the screen.
from dsl.ui.dialogs.data_io import ImportLabelsDialog as _ILD
_dl_img = tmp / "dlg_imgs"; _dl_img.mkdir()
_dl_lab = tmp / "dlg_labels"; _dl_lab.mkdir()
for _i in range(4):
    _imw(_dl_img / f"d{_i}.jpg", np.full((200, 260, 3), 45, np.uint8))
    (_dl_lab / f"d{_i}.txt").write_text("0 0.5 0.5 0.4 0.4" + chr(10), encoding="utf-8")
for _i in range(15):                      # many orphans -> a long report
    (_dl_lab / f"orphan{_i}.txt").write_text("0 .5 .5 .1 .1" + chr(10), encoding="utf-8")

_dproj = Project.create(tmp / "dlg_proj", "d", image_dir=str(_dl_img))
_dproj.scan_images(); _dproj.save()
_dlg = _ILD(_dproj, None)
_dlg.show()
app.processEvents()
assert _dlg.height() <= _dlg._max_h, (_dlg.height(), _dlg._max_h)
_dlg.img_dir.setText(str(_dl_img))
_dlg.path.setText(str(_dl_lab))
_dlg.check_data()
app.processEvents()
assert _dlg.height() <= _dlg._max_h, "dialog grew past the screen"
_sc = _dlg.findChild(_QScrollArea)
assert _sc is not None and _sc.widgetResizable(), "form is not scrollable"
assert _sc.widget().sizeHint().height() > _sc.viewport().height(),     "report unexpectedly fits; the scroll test proves nothing"
assert _dlg.btn_run.isVisible() and _dlg.btn_close.isVisible(),     "buttons pushed out of the dialog"
print("import dialog scrolls:", _sc.widget().sizeHint().height(), "px of content in",
      _sc.viewport().height(), "px")
_dlg.close()

# the file panel offers three views of the same rows: thumbnails, plain
# names, and a collapsible folder tree for datasets spread over folders.
from dsl.ui.panels.file_panel import VIEW_THUMB, VIEW_LIST, VIEW_TREE
_fp_img = tmp / "panel_imgs"
for _f in ("set_a", "set_b"):
    (_fp_img / _f).mkdir(parents=True)
    for _i in range(3):
        _imw(_fp_img / _f / f"{_f}_{_i}.jpg", np.full((60, 80, 3), 44, np.uint8))
_imw(_fp_img / "loose.jpg", np.full((60, 80, 3), 44, np.uint8))
_fproj = Project.create(tmp / "panel_proj", "fp", image_dir=str(_fp_img))
_fproj.scan_images(); _fproj.save()
_panel = win.file_panel
_panel.set_project(_fproj)
app.processEvents()

_panel.mode.setCurrentText(VIEW_TREE)
app.processEvents()
_tm = _panel.tree_model
# top level: set_a/, set_b/ and the loose image that sits beside them
assert _tm.rowCount() == 3, [_tm.data(_tm.index(_i, 0), Qt.DisplayRole)
                             for _i in range(_tm.rowCount())]
assert len(_tm.rows) == 7, _tm.rows
_folders = _files = 0
for _f in range(_tm.rowCount()):
    _fi = _tm.index(_f, 0)
    _n = _tm.rowCount(_fi)
    if _n == 0:                                   # a file at the top level
        assert _tm.data(_fi, Qt.UserRole), "leaf row has no rel"
        _files += 1
        continue
    _folders += 1
    for _c in range(_n):
        _ci = _tm.index(_c, 0, _fi)
        assert _tm.parent(_ci).row() == _f, (_f, _tm.parent(_ci).row())
        assert _tm.data(_ci, Qt.UserRole), "child row has no rel"
assert _folders == 2 and _files == 1, (_folders, _files)
print("folder tree:", _folders, "folders,", len(_tm.rows), "images")

# folders nest to any depth rather than being flattened to one row per path
_deep = tmp / "panel_imgs" / "a" / "b" / "c"
_deep.mkdir(parents=True)
_imw(_deep / "deep.jpg", np.full((60, 80, 3), 44, np.uint8))
_fproj.scan_images(); _fproj.save()
_panel.set_project(_fproj)
app.processEvents()
_node = QModelIndex()
for _want in ("a", "b", "c"):
    _hit = [_r for _r in range(_tm.rowCount(_node))
            if str(_tm.data(_tm.index(_r, 0, _node), Qt.DisplayRole)).startswith(_want)]
    assert _hit, (_want, _tm.rowCount(_node))
    _node = _tm.index(_hit[0], 0, _node)
assert _tm.data(_tm.index(0, 0, _node), Qt.UserRole) == "a/b/c/deep.jpg",     _tm.data(_tm.index(0, 0, _node), Qt.UserRole)
print("nested folders resolve to any depth")

_target = _fproj.images[4]
assert _panel.select(_target, emit=False), _target
assert _panel.current_rel() == _target, (_panel.current_rel(), _target)
_panel.step(1)
assert _panel.current_rel() != _target, "step did not move in tree view"
_panel.step(-1)
assert _panel.current_rel() == _target, "step back failed in tree view"
print("tree selection and A/D stepping work")

# name-only mode must not ask for a single thumbnail
_panel.mode.setCurrentText(VIEW_LIST)
app.processEvents()
assert not _panel.model.show_thumbs
assert _panel.model.data(_panel.model.index(0, 0), Qt.DecorationRole) is None,     "list mode still requests icons"
assert _panel.model.data(_panel.model.index(0, 0), Qt.DisplayRole)
assert not _panel.view.isHidden() and _panel.tree.isHidden()
_panel.mode.setCurrentText(VIEW_THUMB)
app.processEvents()
assert _panel.model.show_thumbs and _panel.view.iconSize().width() > 0
print("view modes switch cleanly:", VIEW_THUMB, VIEW_LIST, VIEW_TREE)

# the auto-label dialog: targets, and a clear message when nothing is trained
from dsl.ui.dialogs.data_io import AutoLabelDialog
from dsl.core.annotation import ImageAnnotation
_al_img = tmp / "auto_imgs"; _al_img.mkdir()
for _i in range(5):
    _imw(_al_img / f"n{_i}.jpg", np.full((80, 100, 3), 44, np.uint8))
_alp = Project.create(tmp / "auto_proj", "al", image_dir=str(_al_img))
_alc = _alp.add_class("thing"); _alp.scan_images(); _alp.save()
_a2 = ImageAnnotation(image_path=_alp.images[0], width=100, height=80)
_a2.shapes.append(Shape(class_id=_alc.id, points=[(10, 10), (60, 10), (60, 60)]))
_alp.save_annotation(_a2); _alp.save_index()

_ald = AutoLabelDialog(_alp, st, None)
assert _ald.target.currentData() == "todo"
assert len(_ald._targets()) == 4, _ald._targets()      # one is already labelled
_ald.target.setCurrentIndex(_ald.target.findData("all"))
assert len(_ald._targets()) == 5, _ald._targets()
assert _ald.note.text(), "no target summary shown"
# no trained runs in this temp project -> say so instead of an empty combo
assert _ald.ckpt.count() == 0 or _ald.ckpt.currentData()
if _ald.ckpt.count() == 0:
    assert "학습" in _ald.note.text(), _ald.note.text()
    assert _ald.build_fn() is None, "ran without a checkpoint"
print("auto-label dialog: targets", len(_ald._targets()),
      "· checkpoints", _ald.ckpt.count())
_ald.close()

win.close()
app.processEvents()
assert not _tmp_parent.exists(), "read-only temp project survived close"
print("read-only temp project removed on close")
print("\nGUI SMOKE OK ->", tmp)
