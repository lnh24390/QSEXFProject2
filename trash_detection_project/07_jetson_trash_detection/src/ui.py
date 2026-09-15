import time
from collections import deque
from functools import lru_cache

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # Pillow가 없으면 한글은 깨지고 영어만 정상 표시된다.
    Image = ImageDraw = ImageFont = None

FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
YELLOW = (0, 220, 255)
BLACK = (0, 0, 0)

# OpenCV의 내장 폰트(Hershey)는 한글 글리프가 없어 한글을 그리면 물음표로
#깨진다. Jetson(JetPack)에 기본 설치되는 Noto Sans CJK 로 대신 그린다.
_KR_FONT_CANDIDATES = (
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 1),  # 1 = KR variant
    ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", 0),
)


@lru_cache(maxsize=None)
def _kr_font_path():
    if ImageFont is None:
        return None
    for path, index in _KR_FONT_CANDIDATES:
        try:
            ImageFont.truetype(path, 16, index=index)
            return (path, index)
        except OSError:
            continue
    print(">> 한글 폰트를 찾지 못해 화면의 한글이 깨질 수 있습니다 "
          "(예: sudo apt install fonts-noto-cjk).")
    return None


@lru_cache(maxsize=64)
def _kr_font(size):
    found = _kr_font_path()
    if found is None:
        return None
    path, index = found
    return ImageFont.truetype(path, size, index=index)


def _fit_text_kr(msg, font, max_width):
    """max_width(px) 를 넘으면 끝을 "..." 로 잘라 폭에 맞춘다.

    font.getlength() 호출 하나가 ~0.3ms 라, 글자를 하나씩 지우며 매번 재는
    방식은 긴 문장에서 눈에 띄게 느리다(수십 ms). 이진 탐색으로 호출 횟수를
    O(글자수) 에서 O(log 글자수) 로 줄인다.
    """
    if font is None or font.getlength(msg) <= max_width:
        return msg

    lo, hi = 0, len(msg)  # msg[:lo] 는 항상 맞고, msg[:hi] 는 항상 넘친다(hi=len 포함 포괄적으로 시작)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if font.getlength(msg[:mid] + "...") <= max_width:
            lo = mid
        else:
            hi = mid - 1
    return (msg[:lo] + "...") if lo else "..."


def draw_text_kr(frame, lines, x, y_top, line_h):
    """한글이 섞인 여러 줄을 한 번에 그린다.

    lines: [(문구, BGR색, 폰트크기 px), ...]. cv2.putText 는 한글을 지원하지
    않으므로(HERSHEY 폰트에 한글 글리프가 없음) 이 영역만 PIL 로 옮겨 그리고
    다시 합성한다 — 전체 프레임을 매번 변환하지 않아 비용이 작다.
    """
    if not lines:
        return
    if _kr_font_path() is None:
        # 폴백: 한글은 깨지지만 최소한 화면에 뭔가는 나오게 한다.
        for i, (msg, color, size) in enumerate(lines):
            _text(frame, msg, x, y_top + i * line_h + size, color, size / 34.0)
        return

    h_img, w_img = frame.shape[:2]
    x0, y0 = max(x - 4, 0), max(y_top - 4, 0)
    x1 = w_img
    y1 = min(y_top + line_h * len(lines) + 8, h_img)
    if x1 <= x0 or y1 <= y0:
        return

    crop_rgb = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
    img = Image.fromarray(crop_rgb)
    draw = ImageDraw.Draw(img)
    for i, (msg, color, size) in enumerate(lines):
        font = _kr_font(size)
        b, g, r = color
        pos = (x - x0, y_top + i * line_h - y0)
        # 패널을 옅게(alpha 낮게) 쓰다 보니 카메라 화면이 밝을 때 글자가 묻힐 수 있어
        # 그림자를 한 번 깔아 대비를 준다. stroke_width 는 방향별로 여러 번 다시
        # 그려 이 CJK 폰트에서 그림자보다 훨씬 느리다(37ms vs 24ms, 7줄 기준).
        draw.text((pos[0] + 1, pos[1] + 1), msg, font=font, fill=(0, 0, 0))
        draw.text(pos, msg, font=font, fill=(r, g, b))
    frame[y0:y1, x0:x1] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


class FpsMeter:
    def __init__(self, window=30):
        self.timestamps = deque(maxlen=window)

    def tick(self):
        self.timestamps.append(time.time())

    @property
    def fps(self):
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        return (len(self.timestamps) - 1) / elapsed if elapsed > 0 else 0.0


class StageTimer:
    """단계별 소요 시간의 누적 평균. 병목 진단용."""

    def __init__(self):
        self.totals = {"cap": 0.0, "infer": 0.0, "draw": 0.0}
        self.count = 0
        self.last_draw = 0.0

    def add(self, cap=0.0, infer=0.0, draw=0.0):
        self.totals["cap"] += cap
        self.totals["infer"] += infer
        self.totals["draw"] += draw
        self.last_draw = draw
        self.count += 1

    def report(self):
        if self.count == 0:
            return "측정된 프레임이 없습니다."

        avg = {k: v / self.count for k, v in self.totals.items()}
        total = sum(avg.values())
        fps = 1000.0 / total if total > 0 else 0.0

        lines = [f"\n[Benchmark] {self.count} frames — 평균 {total:.1f} ms/frame ({fps:.1f} FPS)"]
        labels = {"cap": "캡처 대기", "infer": "추론    ", "draw": "그리기/표시"}
        for key in ("cap", "infer", "draw"):
            share = (avg[key] / total * 100) if total > 0 else 0
            bar = "#" * int(share / 2)
            lines.append(f"  {labels[key]} {avg[key]:6.1f} ms  {share:5.1f}%  {bar}")
        return "\n".join(lines)


def _panel(frame, x, y, w, h, alpha=0.55):
    """반투명 검은 패널을 그려 글자가 배경에 묻히지 않게 한다."""
    h_img, w_img = frame.shape[:2]
    x2, y2 = min(x + w, w_img), min(y + h, h_img)
    x, y = max(x, 0), max(y, 0)
    if x2 <= x or y2 <= y:
        return

    roi = frame[y:y2, x:x2]
    overlay = roi.copy()
    overlay[:] = BLACK
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)


def _text(frame, msg, x, y, color=WHITE, scale=0.6, thickness=1):
    cv2.putText(frame, msg, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def draw_hud(frame, model_name, task, device, fps, timings, conf, iou, counts,
            recording=False, exposure=None):
    """timings: {'cap': ms, 'infer': ms, 'draw': ms} — 병목이 어디인지 바로 보이게 나눠 찍는다."""
    # 이 HUD 는 cv2.putText(Hershey 폰트)로 그려 한글 글리프가 없다 — 영어로 쓴다.
    exposure_label = "auto" if exposure is None else str(exposure)
    lines = [
        (f"Model : {model_name}  [{task}]", GREEN),
        (f"Device: {device}   FPS: {fps:5.1f}", WHITE),
        (
            f"cap {timings.get('cap', 0):5.1f} | infer {timings.get('infer', 0):5.1f}"
            f" | draw {timings.get('draw', 0):5.1f} ms",
            WHITE,
        ),
        (f"Conf  : {conf:.2f}   IoU: {iou:.2f}   imgsz: {timings.get('imgsz', '-')}"
         f"   EXP: {exposure_label}", WHITE),
    ]

    if counts:
        total = sum(counts.values())
        detail = ", ".join(f"{name} x{n}" for name, n in list(counts.items())[:4])
        lines.append((f"Detect: {total}  ({detail})", YELLOW))
    else:
        lines.append(("Detect: 0", YELLOW))

    _panel(frame, 8, 8, 560, 26 * len(lines) + 14)
    for i, (msg, color) in enumerate(lines):
        _text(frame, msg, 18, 32 + i * 26, color)

    if recording:
        cv2.circle(frame, (frame.shape[1] - 30, 30), 10, (0, 0, 255), -1)
        _text(frame, "REC", frame.shape[1] - 90, 37, (0, 0, 255), 0.7, 2)

    return frame


def draw_guide_panel(frame, items, region_name, notice,
                     empty_message="분리수거할 쓰레기를 화면에 담아주세요", bottom_margin=40):
    """지자체별 분리수거 안내를 화면 하단에 띄운다.

    items: RecyclingGuide.summarize() 가 돌려준 목록. 비어 있으면 촬영 안내 문구를 보여준다
    (trash_sorter 앱의 GuidePanel 과 동일한 설계 — README 참고).
    bottom_margin: draw_help() 단축키 줄이 표시될 때 겹치지 않도록 띄우는 여백.
    """
    h_img, w_img = frame.shape[:2]
    line_h = 26
    max_text_w = w_img - 40

    header = f"[지역: {region_name}]" + (f"  {notice}" if notice else "")
    lines = [(header, YELLOW, 18)]

    if items:
        for item in items:
            color = class_color(hash(item["category"]))
            lines.append((f"{item['name']} -> {item['bin']}", color, 19))
            lines.append((item["instruction"], WHITE, 16))
    else:
        lines.append((empty_message, WHITE, 18))

    panel_h = line_h * len(lines) + 16
    y0 = max(h_img - panel_h - bottom_margin, 0)
    # 패널이 화면 하단 상당 부분을 덮으므로 alpha 를 낮게 유지한다.
    # (0.6 이었을 때 카메라 화면이 전반적으로 어둡게 보인다는 피드백)
    _panel(frame, 8, y0, w_img - 16, panel_h, alpha=0.35)

    fitted = [(_fit_text_kr(msg, _kr_font(size), max_text_w), color, size)
              for msg, color, size in lines]
    draw_text_kr(frame, fitted, 18, y0 + 10, line_h)

    return frame


def draw_help(frame):
    msg = ("[m/n] model  [1-9] select  [c/v] conf  [i/o] iou  [,/.]exp  [g] region  [d] reset  "
           "[s] save  [r] rec  [h] help  [q] quit")
    h_img, w_img = frame.shape[:2]
    scale = 0.5 if cv2.getTextSize(msg, FONT, 0.5, 1)[0][0] <= w_img - 32 else 0.42
    _panel(frame, 8, h_img - 40, w_img - 16, 32)
    _text(frame, msg, 18, h_img - 18, WHITE, scale)
    return frame


def draw_model_menu(frame, model_list, current_idx):
    """화면 위에 모델 목록을 띄운다(영상 루프를 멈추지 않는다)."""
    rows = min(len(model_list), 9)
    width, height = 640, 40 + rows * 28 + 30
    x = max((frame.shape[1] - width) // 2, 0)
    y = max((frame.shape[0] - height) // 2, 0)

    _panel(frame, x, y, width, height, alpha=0.75)
    _text(frame, "SELECT MODEL  (press number, ESC to close)", x + 16, y + 28, GREEN, 0.65)

    for i, info in enumerate(model_list[:9]):
        marker = ">" if i == current_idx else " "
        color = YELLOW if i == current_idx else WHITE
        _text(frame, f"{marker} [{i + 1}] {info['name']}", x + 16, y + 58 + i * 28, color, 0.55)

    if len(model_list) > 9:
        _text(frame, f"... +{len(model_list) - 9} more (use m/n)", x + 16, y + 58 + rows * 28, WHITE, 0.5)

    return frame


# 클래스별 고정 색상. 같은 클래스가 프레임마다 다른 색으로 깜빡이지 않게 한다.
_PALETTE = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
    (147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132),
    (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255),
]


def class_color(idx):
    return _PALETTE[int(idx) % len(_PALETTE)]


def class_name(names, idx):
    """result.names 는 dict 일 수도 list 일 수도 있다."""
    if isinstance(names, dict):
        return names.get(idx, str(idx))
    if isinstance(names, (list, tuple)) and idx < len(names):
        return names[idx]
    return str(idx)


def draw_detections(frame, result, scale=1.0, line_width=2):
    """검출 박스를 frame 위에 직접 그린다. 직접 그릴 수 없으면 False.

    Results.plot() 은 프레임마다 원본 해상도 이미지를 한 장 통째로 복사하고
    선 두께와 폰트 크기를 다시 계산한다. 표시용으로 이미 줄여 둔 화면에 바로
    그리면 그 복사가 사라지고 칠할 픽셀도 줄어든다.

    마스크가 필요한 분할(segmentation) 모델은 직접 그리기가 까다로우므로
    False 를 돌려주고 호출한 쪽이 plot() 으로 넘어가게 한다.
    """
    if result is None:
        return True
    if getattr(result, "masks", None) is not None:
        return False

    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return True

    names = result.names
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int)

    for (x1, y1, x2, y2), conf, cid in zip(xyxy, confs, cls_ids):
        color = class_color(cid)
        p1 = (int(x1 * scale), int(y1 * scale))
        p2 = (int(x2 * scale), int(y2 * scale))
        cv2.rectangle(frame, p1, p2, color, line_width, cv2.LINE_AA)

        label = f"{class_name(names, cid)} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.45, 1)
        ly = max(p1[1], th + 5)
        cv2.rectangle(frame, (p1[0], ly - th - 5), (p1[0] + tw + 5, ly), color, -1)
        cv2.putText(frame, label, (p1[0] + 3, ly - 4), FONT, 0.45, BLACK, 1, cv2.LINE_AA)

    return True
