"""Jetson 쓰레기 검출 - YOLO 모델 실시간 추론 실행기."""

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2

from src.config import DEFAULT_CONFIG, load_config, reset_config, save_config
from src.detector import Detector, diagnose_no_cuda, discover_models
from src.recycling_guide import RecyclingGuide
from src.ui import (FpsMeter, StageTimer, draw_detections, draw_guide_panel,
                    draw_help, draw_hud, draw_model_menu)
from src.webcam import WebCam

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "model"
CONFIG_PATH = PROJECT_ROOT / "config.json"
GUIDE_PATH = PROJECT_ROOT / "config" / "recycling_guide.json"
OUTPUT_DIR = PROJECT_ROOT / "output"

WINDOW_TITLE = "Trash Detection | [m] model  [g] region  [s] save  [h] help  [q] quit"


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO 쓰레기 검출 실시간 추론")
    parser.add_argument("--source", help="카메라 번호 또는 동영상 경로 (기본: 설정값)")
    parser.add_argument("--model", help="사용할 모델 이름 또는 .pt 경로")
    parser.add_argument("--conf", type=float, help="confidence threshold")
    parser.add_argument("--iou", type=float, help="IoU threshold")
    parser.add_argument("--imgsz", type=int, help="추론 입력 크기")
    parser.add_argument("--device", help="auto | cpu | 0")
    parser.add_argument("--quantize", type=int, choices=[8, 16, 32],
                        help="추론 정밀도: 16=FP16(권장), 8=INT8, 32=FP32")
    parser.add_argument("--half", action="store_true",
                        help=argparse.SUPPRESS)  # 구버전 호환: --quantize 16 과 동일
    parser.add_argument("--csi", action="store_true", help="Jetson CSI 카메라 사용")
    parser.add_argument("--flip-method", type=int, default=0, help="CSI 카메라 회전 (0~7)")
    parser.add_argument("--headless", action="store_true", help="창 없이 콘솔 출력만")
    parser.add_argument("--list-models", action="store_true", help="모델 목록만 출력하고 종료")
    parser.add_argument("--reset-config", action="store_true",
                        help="conf/iou/노출/지역 등 저장된 설정을 전부 기본값으로 되돌리고 시작")
    parser.add_argument("--cam-size", help="캡처 해상도 (예: 640x480). 낮추면 캡처 시간이 줄어든다")
    parser.add_argument("--no-mjpg", action="store_true", help="MJPG 강제 지정을 끈다")
    parser.add_argument("--no-thread", action="store_true", help="캡처 스레드를 끈다")
    parser.add_argument("--no-engine", action="store_true",
                        help="TensorRT 엔진 대신 .pt 를 쓴다 (검출 결과 비교용)")
    parser.add_argument("--display-scale", type=float, default=1.0,
                        help="표시 배율 (0.5면 절반 크기). 녹화/스냅샷도 이 배율을 따른다")
    parser.add_argument("--benchmark", type=int, metavar="N",
                        help="N 프레임 처리 후 단계별 평균 시간을 출력하고 종료")
    parser.add_argument("--exposure", type=int,
                        help="수동 노출값 (역광일 때 밝게 하려면 200~500 정도). 생략하면 자동노출")
    parser.add_argument("--region", help="분리수거 안내 지역 (예: default, jeju, seoul)")
    parser.add_argument("--no-guide", action="store_true", help="분리수거 안내 패널을 끈다")
    return parser.parse_args()


def normalize_source(value):
    """'0' 같은 문자열은 카메라 번호로, 나머지는 경로로 취급한다."""
    if isinstance(value, int):
        return value
    text = str(value)
    return int(text) if text.isdigit() else text


def pick_model_index(model_list, wanted):
    if not wanted:
        return 0

    stem = Path(str(wanted)).stem
    for i, info in enumerate(model_list):
        if info["name"] == stem:
            return i

    print(f">> '{wanted}' 모델을 찾지 못했습니다. 첫 번째 모델을 사용합니다.")
    return 0


def apply_args(config, args):
    if args.source is not None:
        config["source"] = normalize_source(args.source)
    if args.conf is not None:
        config["conf_threshold"] = args.conf
    if args.iou is not None:
        config["iou_threshold"] = args.iou
    if args.imgsz is not None:
        config["imgsz"] = args.imgsz
    if args.device is not None:
        config["device"] = args.device
    if args.quantize is not None:
        config["quantize"] = None if args.quantize == 32 else args.quantize
    elif args.half:
        print(">> --half 는 --quantize 16 으로 대체되었습니다.")
        config["quantize"] = 16
    if args.cam_size:
        try:
            width, height = (int(v) for v in args.cam_size.lower().split("x"))
            config["width"], config["height"] = width, height
        except ValueError:
            print(f">> --cam-size 형식이 잘못됐습니다: {args.cam_size} (예: 640x480)")
    if args.region is not None:
        config["region"] = args.region
    if args.exposure is not None:
        config["exposure"] = args.exposure
    return config


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_snapshot(frame):
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / f"snapshot_{timestamp()}.jpg"
    cv2.imwrite(str(path), frame)
    print(f">> 스냅샷 저장: {path}")


def open_writer(frame, fps):
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / f"record_{timestamp()}.mp4"
    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, max(fps, 1.0), (width, height))
    if not writer.isOpened():
        print(">> 녹화를 시작할 수 없습니다 (코덱 확인 필요).")
        return None
    print(f">> 녹화 시작: {path}")
    return writer


def clamp(value, low, high):
    return max(low, min(high, value))


def print_controls():
    print("\n[Controls]")
    print("  m / n      : 다음 / 이전 모델")
    print("  Space      : 모델 선택 메뉴 열기 (숫자키로 선택, ESC 닫기)")
    print("  c / v      : confidence -0.05 / +0.05")
    print("  i / o      : IoU -0.05 / +0.05")
    print("  s          : 현재 화면 저장")
    print("  r          : 녹화 시작 / 중지")
    print("  g          : 분리수거 안내 지역 전환 (GPS가 없어 수동으로 고른다)")
    print("  , / .      : 노출 어둡게 / 밝게 (역광으로 화면이 까맣게 나올 때)")
    print("  0          : 노출을 자동으로 되돌림")
    print("  d          : conf/iou/노출/지역을 기본값으로 되돌림")
    print("  h          : 단축키 표시 토글")
    print("  q / ESC    : 종료\n")


def print_perf_hints(stage_avg, detector):
    """가장 오래 걸린 단계를 짚어 다음에 뭘 손대야 하는지 알려준다."""
    if stage_avg.count == 0:
        return

    avg = {k: v / stage_avg.count for k, v in stage_avg.totals.items()}
    worst = max(avg, key=avg.get)

    print("\n[Hint]")
    if detector.device == "cpu":
        # GPU를 못 쓰는 상태에서는 imgsz나 카메라를 손봐야 몇 FPS 차이다.
        print("  GPU를 쓰지 않고 CPU로 추론하고 있습니다. 이게 가장 큰 병목입니다.")
        for line in diagnose_no_cuda():
            print(f"    {line}")
        return

    if worst == "cap":
        print("  캡처가 병목입니다. 카메라가 FPS를 제한하고 있습니다.")
        print("  - `v4l2-ctl -d /dev/video0 --list-formats-ext` 로 MJPG 지원 해상도/FPS 확인")
        print("  - `--cam-size 640x480` 으로 낮춰 보세요")
    elif worst == "infer":
        print("  추론이 병목입니다.")
        if not str(detector.model_list[detector.index]["path"]).endswith(".engine"):
            print("  - TensorRT로 내보내면 보통 2~4배 빨라집니다:")
            print("      python tools/export_engine.py --all --quantize 16")
        if detector.quantize is None:
            print("  - `--quantize 16` (FP16) 을 켜 보세요")
        print(f"  - 현재 imgsz={detector.imgsz}. `--imgsz 416` 또는 `--imgsz 320` 으로 낮춰 보세요")
        print("  - 전력 모드 확인: `sudo nvpmodel -q` → MAXN 이 아니면 `sudo nvpmodel -m 0 && sudo jetson_clocks`")
    else:
        print("  화면 표시가 병목입니다.")
        print("  - SSH(X11 포워딩)로 띄우고 있다면 그것이 원인입니다. `--headless` 로 확인해 보세요")
        print("  - `--display-scale 0.5` 로 표시 크기를 줄여 보세요")


def main():
    args = parse_args()
    if args.reset_config:
        reset_config(CONFIG_PATH)
        print(">> 설정을 기본값으로 되돌렸습니다.")
    config = apply_args(load_config(CONFIG_PATH), args)

    model_list = discover_models(MODEL_DIR, prefer_engine=not args.no_engine)
    if args.list_models:
        print(f"\n[{len(model_list)} models in {MODEL_DIR}]")
        for i, info in enumerate(model_list):
            print(f"  [{i + 1}] {info['name']}")
        return

    model_index = pick_model_index(model_list, args.model or config.get("model_name"))
    detector = Detector(
        model_list,
        index=model_index,
        device=config["device"],
        imgsz=config["imgsz"],
        quantize=config["quantize"],
    )

    cam = WebCam(
        source=normalize_source(config["source"]),
        width=config["width"],
        height=config["height"],
        csi=args.csi,
        flip_method=args.flip_method,
        mjpg=not args.no_mjpg,
        threaded=not args.no_thread,
        exposure=config["exposure"],
    )
    if not cam.start():
        return

    guide = None
    if not args.no_guide:
        try:
            guide = RecyclingGuide(GUIDE_PATH)
            if config["region"] not in guide.region_ids():
                print(f">> region '{config['region']}' 을 몰라 'default' 를 씁니다.")
                config["region"] = "default"
        except (OSError, ValueError) as exc:
            print(f">> 분리수거 안내를 불러오지 못했습니다 ({exc}). 안내 없이 진행합니다.")

    fps_meter = FpsMeter()
    stage_avg = StageTimer()
    show_help = True
    show_menu = False
    writer = None
    frame_count = 0

    if not args.headless:
        print_controls()
        if guide is not None:
            print(f">> 분리수거 안내 지역: {guide.region_name(config['region'])} "
                  f"('g' 로 전환, 현재: {config['region']})")

    try:
        while True:
            t0 = time.perf_counter()
            frame = cam.get_frame()
            if frame is None:
                break
            cap_ms = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            result = detector.predict(
                frame,
                conf=config["conf_threshold"],
                iou=config["iou_threshold"],
                max_det=config["max_det"],
            )
            infer_ms = (time.perf_counter() - t1) * 1000

            t2 = time.perf_counter()
            counts = detector.summarize(result)
            fps_meter.tick()
            frame_count += 1

            if args.headless:
                stage_avg.add(cap=cap_ms, infer=infer_ms, draw=0.0)
                if frame_count % 30 == 0:
                    summary = counts if counts else "검출 없음"
                    print(f"[{frame_count}] {fps_meter.fps:.1f} FPS | "
                          f"cap {cap_ms:.1f} / infer {infer_ms:.1f} ms | {summary}")
                    if guide is not None and counts:
                        for item in guide.summarize(counts, config["region"], config["guide_max_items"]):
                            print(f"    -> {item['name']}: {item['bin']} | {item['instruction']}")
                if args.benchmark and frame_count >= args.benchmark:
                    break
                continue

            # 표시용 캔버스를 먼저 만든다. 줄일 거라면 그리기 전에 줄여야
            # 원본 해상도에 박스와 HUD를 그렸다가 버리는 낭비가 없다.
            scale = args.display_scale
            if scale != 1.0:
                annotated = cv2.resize(frame, None, fx=scale, fy=scale,
                                       interpolation=cv2.INTER_AREA)
            else:
                annotated = frame.copy()

            if not draw_detections(annotated, result, scale):
                # 마스크가 있는 분할 모델은 plot() 에 맡긴다.
                annotated = result.plot()
                if scale != 1.0:
                    annotated = cv2.resize(annotated, None, fx=scale, fy=scale,
                                           interpolation=cv2.INTER_AREA)

            timings = {"cap": cap_ms, "infer": infer_ms,
                       "draw": stage_avg.last_draw, "imgsz": detector.imgsz}
            draw_hud(
                annotated,
                detector.name,
                detector.task,
                f"{detector.device} {detector.precision_label}",
                fps_meter.fps,
                timings,
                config["conf_threshold"],
                config["iou_threshold"],
                counts,
                recording=writer is not None,
                exposure=config["exposure"],
            )
            if guide is not None:
                items = guide.summarize(counts, config["region"], config["guide_max_items"])
                draw_guide_panel(annotated, items, guide.region_name(config["region"]),
                                 guide.notice(config["region"]), bottom_margin=40 if show_help else 8)
            if show_help:
                draw_help(annotated)
            if show_menu:
                draw_model_menu(annotated, model_list, detector.index)

            if writer is not None:
                writer.write(annotated)

            cv2.imshow(WINDOW_TITLE, annotated)
            key = cv2.waitKey(1) & 0xFF

            draw_ms = (time.perf_counter() - t2) * 1000
            stage_avg.add(cap=cap_ms, infer=infer_ms, draw=draw_ms)

            if args.benchmark and frame_count >= args.benchmark:
                break

            if key == 255:  # 눌린 키 없음
                if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                    break  # 창을 닫으면 종료
                continue

            if key in (ord('q'), 27):  # q / ESC
                if show_menu and key == 27:
                    show_menu = False
                    continue
                break
            elif key == ord(' '):
                show_menu = not show_menu
            elif show_menu and ord('1') <= key <= ord('9'):
                target = key - ord('1')
                if target < len(model_list):
                    detector.load(target)
                show_menu = False
            elif key == ord('m'):
                detector.next_model()
            elif key == ord('n'):
                detector.prev_model()
            elif key == ord('c'):
                config["conf_threshold"] = clamp(round(config["conf_threshold"] - 0.05, 2), 0.05, 0.95)
            elif key == ord('v'):  # 원래 Shift+C 였으나 이 OpenCV/Qt 조합에서 Shift 조합이
                config["conf_threshold"] = clamp(round(config["conf_threshold"] + 0.05, 2), 0.05, 0.95)
            elif key == ord('i'):  # 인식되지 않아(cv2.waitKey 가 항상 소문자로 돌려줌) 대체했다.
                config["iou_threshold"] = clamp(round(config["iou_threshold"] - 0.05, 2), 0.05, 0.95)
            elif key == ord('o'):
                config["iou_threshold"] = clamp(round(config["iou_threshold"] + 0.05, 2), 0.05, 0.95)
            elif key == ord('s'):
                save_snapshot(annotated)
            elif key == ord('h'):
                show_help = not show_help
            elif key == ord('g') and guide is not None:
                config["region"] = guide.next_region(config["region"])
                print(f">> 분리수거 안내 지역: {guide.region_name(config['region'])} "
                      f"({config['region']})")
            elif key == ord(','):
                base = config["exposure"] if config["exposure"] is not None else 300
                config["exposure"] = clamp(base - 50, 1, 5000)
                cam.set_exposure(config["exposure"])
                print(f">> 노출: {config['exposure']} (어둡게)")
            elif key == ord('.'):
                base = config["exposure"] if config["exposure"] is not None else 300
                config["exposure"] = clamp(base + 50, 1, 5000)
                cam.set_exposure(config["exposure"])
                print(f">> 노출: {config['exposure']} (밝게)")
            elif key == ord('0') and not show_menu:
                config["exposure"] = None
                cam.set_exposure(None)
                print(">> 노출: 자동으로 복귀")
            elif key == ord('d'):
                # 실행 중 키로 바꿀 수 있는 값만 되돌린다. 해상도/모델처럼 재시작이
                # 필요한 값은 --reset-config 로 다음 실행 때 반영한다.
                config["conf_threshold"] = DEFAULT_CONFIG["conf_threshold"]
                config["iou_threshold"] = DEFAULT_CONFIG["iou_threshold"]
                config["exposure"] = DEFAULT_CONFIG["exposure"]
                config["region"] = DEFAULT_CONFIG["region"]
                cam.set_exposure(config["exposure"])
                print(">> 설정을 기본값으로 되돌렸습니다 (conf/iou/노출/지역).")
            elif key == ord('r'):
                if writer is None:
                    writer = open_writer(annotated, fps_meter.fps or 20.0)
                else:
                    writer.release()
                    writer = None
                    print(">> 녹화 중지")

    except KeyboardInterrupt:
        print("\n>> 사용자 중단")
    finally:
        if writer is not None:
            writer.release()
        cam.stop()
        config["model_name"] = detector.name
        save_config(CONFIG_PATH, config)
        print(f">> 총 {frame_count} 프레임 처리")
        print(stage_avg.report())
        print_perf_hints(stage_avg, detector)


if __name__ == "__main__":
    main()
