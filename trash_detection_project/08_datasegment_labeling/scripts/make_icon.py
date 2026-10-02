"""Render assets/icon.svg into the raster sizes Windows and Qt need.

    uv run --no-project --python .venv/Scripts/python.exe scripts/make_icon.py

The SVG is the source of truth; everything else here is generated, so edit
the SVG and re-run rather than touching the PNGs. Qt loads the SVG directly
at runtime, but PyInstaller needs a real .ico for the executable and the
taskbar, which is what this produces.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SVG = ROOT / "assets" / "icon.svg"
OUT = ROOT / "assets"
SIZES = (16, 24, 32, 48, 64, 128, 256)


def render(size: int):
    from PySide6.QtCore import QSize, Qt
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer

    r = QSvgRenderer(str(SVG))
    if not r.isValid():
        raise SystemExit(f"SVG 를 읽지 못했습니다: {SVG}")
    img = QImage(QSize(size, size), QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.setRenderHint(QPainter.SmoothPixmapTransform, True)
    r.render(p)
    p.end()
    return img


def main() -> int:
    if not SVG.exists():
        print(f"없음: {SVG}", file=sys.stderr)
        return 1
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])   # QImage needs one

    OUT.mkdir(parents=True, exist_ok=True)
    pngs = []
    for s in SIZES:
        img = render(s)
        dst = OUT / f"icon_{s}.png"
        if not img.save(str(dst), "PNG"):
            print(f"저장 실패: {dst}", file=sys.stderr)
            return 1
        pngs.append(dst)
        print(f"  {dst.name}  {img.width()}x{img.height()}")

    try:
        from PIL import Image
    except ImportError:
        print("Pillow 가 없어 .ico 는 건너뜁니다 (uv pip install pillow)")
        return 0
    ico = OUT / "icon.ico"
    base = Image.open(pngs[-1]).convert("RGBA")
    base.save(ico, format="ICO", sizes=[(s, s) for s in SIZES])
    print(f"  {ico.name}  ({', '.join(str(s) for s in SIZES)})")
    del app
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
