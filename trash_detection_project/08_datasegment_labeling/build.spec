# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the DataSegmentLabeling GUI.

    uv run pyinstaller build.spec --noconfirm

What is *not* bundled, and why
------------------------------
torch / torchvision / ultralytics / transformers / sam2 / segment-anything are
excluded on purpose. DEVELOPMENT.md section 2 requires the app to start and label
manually with none of them installed, and they are all lazy-imported inside
functions. Bundling torch with its CUDA runtime would add roughly 5GB to the
build for code most sessions never touch.

onnxruntime *is* bundled (13MB): it is the one SAM backend that needs no
torch, so the shipped exe can segment out of the box.

The build is onedir, not onefile: onefile unpacks ~200MB to a temp directory
on every launch, and this app is started often.

Runtime layout expected next to the executable:
    DataSegmentLabeling/
      DataSegmentLabeling.exe
      weights/      <- copy or symlink; not bundled (24GB)
      configs/      <- created on first run
"""
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

# The plain .py sources, shipped beside the bundle as `dsl_src/dsl/`.
# Training runs in a separate process on an *external* interpreter (the exe
# rejects `-m`, and torch is not bundled), and that interpreter has to be
# able to import `dsl.train.cli`. See settings.train_source_root().
_SRC = Path("dsl")
source_datas = [
    (str(p), str(Path("dsl_src") / p.parent))
    for p in _SRC.rglob("*.py")
    if "__pycache__" not in p.parts
]

# The window icon. Qt loads it at runtime from the unpacked bundle;
# the exe's own icon is set separately below.
icon_datas = [(str(q), "assets") for q in sorted(Path("assets").glob("*"))
              if q.suffix.lower() in (".svg", ".ico", ".png")]
ICON_FILE = Path("assets/icon.ico")

# dsl.* is imported through several indirection layers (the SAM backend
# registry, the trainer registry), so let PyInstaller take the whole package
# rather than trusting static analysis to find every entry.
hiddenimports = collect_submodules("dsl") + [
    "onnxruntime",
    "onnxruntime.capi._pybind_state",
]

# Heavy optional deps: never traced into the bundle. Each is guarded by a
# try/except ImportError at its use site that tells the user what to install.
excludes = [
    "torch", "torchvision", "torchaudio",
    "ultralytics", "transformers", "huggingface_hub",
    "sam2", "segment_anything", "pycocotools",
    "matplotlib", "scipy", "pandas", "IPython", "notebook",
    "tkinter", "PyQt5", "PyQt6", "PySide2",
    "pytest", "sphinx",
]

a = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=[],
    datas=source_datas + icon_datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DataSegmentLabeling",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,          # GUI app: no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ICON_FILE) if ICON_FILE.exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="DataSegmentLabeling",
)
