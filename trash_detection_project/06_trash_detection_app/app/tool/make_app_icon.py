"""앱 아이콘(SVG 원본 + PNG) 생성.

디자인: 초록 배경 + 재활용 화살표 삼각형 + 카메라 뷰파인더 모서리
(앱이 "카메라로 쓰레기를 보고 분리수거를 안내"하는 것을 한 눈에 보이게 함)

SVG 와 PNG 를 같은 좌표 계산으로 만들어 둘이 어긋나지 않게 합니다.
실행: ml/.venv/Scripts/python.exe app/tool/make_app_icon.py
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

SIZE = 1024
SS = 4  # 안티에일리어싱용 슈퍼샘플링 배율

GREEN_TOP = (46, 125, 50)  # #2E7D32 (앱 시드 컬러)
GREEN_BOTTOM = (20, 83, 27)  # #14531B
WHITE = (255, 255, 255)

OUT = Path(__file__).resolve().parent.parent / "assets" / "icon"


# ---------------------------------------------------------------- 좌표 계산


def rotate(p: tuple[float, float], center: tuple[float, float], deg: float):
    a = math.radians(deg)
    dx, dy = p[0] - center[0], p[1] - center[1]
    return (
        center[0] + dx * math.cos(a) - dy * math.sin(a),
        center[1] + dx * math.sin(a) + dy * math.cos(a),
    )


def recycle_arrows(center: tuple[float, float], radius: float, thickness: float):
    """재활용 삼각형: 화살표 3개(선분 + 화살촉)의 좌표를 만든다."""
    arrows = []
    for k in range(3):
        a0 = -90 + 120 * k
        a1 = a0 + 120
        v0 = (
            center[0] + radius * math.cos(math.radians(a0)),
            center[1] + radius * math.sin(math.radians(a0)),
        )
        v1 = (
            center[0] + radius * math.cos(math.radians(a1)),
            center[1] + radius * math.sin(math.radians(a1)),
        )
        dx, dy = v1[0] - v0[0], v1[1] - v0[1]
        length = math.hypot(dx, dy)
        ux, uy = dx / length, dy / length

        # 모서리에 틈을 두고, 끝에는 화살촉 자리를 비워 둔다
        start = (v0[0] + ux * length * 0.16, v0[1] + uy * length * 0.16)
        shaft_end = (v0[0] + ux * length * 0.68, v0[1] + uy * length * 0.68)
        tip = (v0[0] + ux * length * 0.90, v0[1] + uy * length * 0.90)

        # 화살촉: 진행 방향에 수직으로 벌린 삼각형
        nx, ny = -uy, ux
        half = thickness * 0.95
        head = [
            tip,
            (shaft_end[0] + nx * half, shaft_end[1] + ny * half),
            (shaft_end[0] - nx * half, shaft_end[1] - ny * half),
        ]
        arrows.append({"shaft": (start, shaft_end), "head": head})
    return arrows


def viewfinder_corners(inset: float, arm: float, thickness: float):
    """카메라 뷰파인더 모서리 4개. 각 모서리는 가로·세로 막대 2개."""
    lo, hi = inset, SIZE - inset
    t = thickness
    bars = []
    for cx, cy, sx, sy in (
        (lo, lo, 1, 1),  # 좌상
        (hi, lo, -1, 1),  # 우상
        (lo, hi, 1, -1),  # 좌하
        (hi, hi, -1, -1),  # 우하
    ):
        # 가로 막대
        x0, x1 = sorted([cx, cx + sx * arm])
        y0, y1 = sorted([cy, cy + sy * t])
        bars.append((x0, y0, x1, y1))
        # 세로 막대
        x0, x1 = sorted([cx, cx + sx * t])
        y0, y1 = sorted([cy, cy + sy * arm])
        bars.append((x0, y0, x1, y1))
    return bars


# ---------------------------------------------------------------- 그리기

CENTER = (SIZE / 2, SIZE / 2 + 4)
RADIUS = 224
THICK = 58
BARS = viewfinder_corners(inset=132, arm=176, thickness=46)
ARROWS = recycle_arrows(CENTER, RADIUS, THICK)


def draw_symbol(draw: ImageDraw.ImageDraw, scale: float, offset: tuple[float, float]):
    """재활용 삼각형 + 뷰파인더를 그린다. scale/offset 으로 크기·위치 조정."""

    def p(pt):
        return (pt[0] * scale + offset[0], pt[1] * scale + offset[1])

    for x0, y0, x1, y1 in BARS:
        r = 16 * scale
        draw.rounded_rectangle([p((x0, y0)), p((x1, y1))], radius=r, fill=WHITE)

    for arrow in ARROWS:
        start, end = arrow["shaft"]
        draw.line([p(start), p(end)], fill=WHITE, width=int(THICK * scale), joint="curve")
        # 둥근 끝 처리
        for pt in (start, end):
            r = THICK * scale / 2
            c = p(pt)
            draw.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], fill=WHITE)
        draw.polygon([p(v) for v in arrow["head"]], fill=WHITE)


def render_png(path: Path, with_background: bool, content_scale: float = 1.0):
    img = Image.new("RGBA", (SIZE * SS, SIZE * SS), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    if with_background:
        # 세로 그라데이션 배경 + 둥근 모서리
        grad = Image.new("RGBA", (1, SIZE * SS))
        for y in range(SIZE * SS):
            t = y / (SIZE * SS - 1)
            grad.putpixel(
                (0, y),
                (
                    round(GREEN_TOP[0] + (GREEN_BOTTOM[0] - GREEN_TOP[0]) * t),
                    round(GREEN_TOP[1] + (GREEN_BOTTOM[1] - GREEN_TOP[1]) * t),
                    round(GREEN_TOP[2] + (GREEN_BOTTOM[2] - GREEN_TOP[2]) * t),
                    255,
                ),
            )
        bg = grad.resize((SIZE * SS, SIZE * SS))
        mask = Image.new("L", (SIZE * SS, SIZE * SS), 0)
        ImageDraw.Draw(mask).rounded_rectangle(
            [0, 0, SIZE * SS - 1, SIZE * SS - 1], radius=210 * SS, fill=255
        )
        img.paste(bg, (0, 0), mask)

    # 적응형 아이콘은 가운데 66% 안에 내용이 들어가야 잘리지 않는다
    scale = SS * content_scale
    offset = (
        SIZE * SS * (1 - content_scale) / 2,
        SIZE * SS * (1 - content_scale) / 2,
    )
    draw_symbol(draw, scale, offset)

    img.resize((SIZE, SIZE), Image.LANCZOS).save(path)
    print(f"wrote {path.name} ({SIZE}x{SIZE}, background={with_background})")


# ---------------------------------------------------------------- SVG


def svg_source() -> str:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SIZE}" height="{SIZE}" '
        f'viewBox="0 0 {SIZE} {SIZE}">',
        "  <defs>",
        '    <linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">',
        f'      <stop offset="0" stop-color="rgb{GREEN_TOP}"/>',
        f'      <stop offset="1" stop-color="rgb{GREEN_BOTTOM}"/>',
        "    </linearGradient>",
        "  </defs>",
        f'  <rect width="{SIZE}" height="{SIZE}" rx="210" fill="url(#bg)"/>',
        "  <g fill=\"#fff\">",
    ]
    for x0, y0, x1, y1 in BARS:
        parts.append(
            f'    <rect x="{x0:.1f}" y="{y0:.1f}" width="{x1 - x0:.1f}" '
            f'height="{y1 - y0:.1f}" rx="16"/>'
        )
    parts.append("  </g>")
    parts.append(
        f'  <g stroke="#fff" stroke-width="{THICK}" stroke-linecap="round" fill="#fff">'
    )
    for arrow in ARROWS:
        (sx, sy), (ex, ey) = arrow["shaft"]
        parts.append(
            f'    <line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}"/>'
        )
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in arrow["head"])
        parts.append(f'    <polygon points="{pts}" stroke="none"/>')
    parts.append("  </g>")
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "app_icon.svg").write_text(svg_source(), encoding="utf-8")
    print("wrote app_icon.svg")
    render_png(OUT / "app_icon.png", with_background=True)
    # 적응형 아이콘 전경: 배경 없이 캔버스를 채운다.
    # (flutter_launcher_icons 가 생성 시 16% 여백을 더 넣어 안전 영역을 맞춘다.
    #  여기서 또 줄이면 홈 화면에서 아이콘이 지나치게 작아진다)
    render_png(OUT / "app_icon_foreground.png", with_background=False, content_scale=1.0)


if __name__ == "__main__":
    main()
