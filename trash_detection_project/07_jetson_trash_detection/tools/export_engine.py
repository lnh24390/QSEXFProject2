"""YOLO .pt 가중치를 TensorRT .engine 으로 내보낸다.

main.py 는 model/ 안에 같은 이름의 .engine 과 .pt 가 함께 있으면 .engine 을
자동으로 고른다(src/detector.py 의 discover_models). 따라서 여기서 변환만 해 두면
실행 쪽은 건드릴 것이 없다.

    python3 tools/export_engine.py --all --quantize 16
    python3 tools/export_engine.py --model ain_yolov11n_epoch100_70000

엔진은 빌드한 장비 / TensorRT 버전에 묶인다. JetPack을 올리거나 다른 젯슨으로
옮기면 다시 내보내야 한다.
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import discover_models, precision_kwargs, resolve_device  # noqa: E402

MODEL_DIR = PROJECT_ROOT / "model"


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO 가중치를 TensorRT 엔진으로 변환")
    parser.add_argument("--model", help="변환할 모델 이름 (생략 시 --all 필요)")
    parser.add_argument("--all", action="store_true", help="model/ 안의 모든 .pt 변환")
    parser.add_argument("--imgsz", type=int, default=640, help="엔진 입력 크기 (기본 640)")
    parser.add_argument("--quantize", type=int, choices=[8, 16, 32], default=16,
                        help="16=FP16(권장), 8=INT8, 32=FP32")
    parser.add_argument("--workspace", type=int, default=4,
                        help="빌드 작업 메모리 상한 (GB). 메모리 부족 시 낮춘다")
    parser.add_argument("--batch", type=int, default=1, help="엔진 배치 크기")
    parser.add_argument("--force", action="store_true", help="이미 있는 엔진도 다시 만든다")
    parser.add_argument("--data", help="INT8 보정용 데이터셋 yaml (--quantize 8 일 때 권장)")
    return parser.parse_args()


def needs_export(pt_path, engine_path, force):
    """이미 최신 엔진이 있으면 건너뛴다. 빌드가 몇 분씩 걸리기 때문이다."""
    if force or not engine_path.exists():
        return True
    if engine_path.stat().st_mtime < pt_path.stat().st_mtime:
        print("   (.pt 가 더 최신이라 다시 만듭니다)")
        return True
    return False


def export_one(pt_path, args):
    from ultralytics import YOLO

    engine_path = pt_path.with_suffix(".engine")
    print(f"\n[{pt_path.name}]")
    if not needs_export(pt_path, engine_path, args.force):
        print(f"   건너뜀 — 이미 있습니다: {engine_path.name}")
        return True

    kwargs = dict(
        format="engine",
        imgsz=args.imgsz,
        batch=args.batch,
        workspace=args.workspace,
        device=0,
        verbose=False,
    )
    kwargs.update(precision_kwargs(args.quantize, for_export=True))
    if args.quantize == 8:
        if args.data:
            kwargs["data"] = args.data
        else:
            print("   주의: --data 없이 INT8로 내보내면 정확도가 크게 떨어질 수 있습니다.")

    print(f"   변환 중 (imgsz={args.imgsz}, {args.quantize}bit) — 몇 분 걸립니다...")
    start = time.time()
    try:
        YOLO(str(pt_path)).export(**kwargs)
    except Exception as exc:
        print(f"   실패: {exc}")
        return False

    if not engine_path.exists():
        print("   실패: 엔진 파일이 생성되지 않았습니다.")
        return False

    size_mb = engine_path.stat().st_size / (1024 * 1024)
    print(f"   완료: {engine_path.name} ({size_mb:.1f} MB, {time.time() - start:.0f}s)")
    return True


def main():
    args = parse_args()

    device = resolve_device("auto")
    if device == "cpu":
        print(">> TensorRT 엔진 변환은 GPU가 필요합니다. 위 진단을 먼저 해결하세요.")
        return 1

    models = discover_models(MODEL_DIR)
    pt_models = [m for m in models if m["path"].endswith(".pt")]

    if args.model:
        stem = Path(args.model).stem
        targets = [Path(m["path"]) for m in pt_models if m["name"] == stem]
        if not targets:
            print(f">> '{args.model}' 에 해당하는 .pt 를 찾지 못했습니다.")
            print(f"   가능한 모델: {', '.join(m['name'] for m in pt_models)}")
            return 1
    elif args.all:
        targets = [Path(m["path"]) for m in pt_models]
    else:
        print(">> --model 또는 --all 중 하나를 지정하세요.")
        return 1

    print(f">> {len(targets)}개 모델을 변환합니다 (device={device})")
    ok = sum(export_one(path, args) for path in targets)

    print(f"\n>> 완료: {ok}/{len(targets)} 성공")
    if ok:
        print("   이제 main.py 를 그냥 실행하면 .engine 이 자동으로 선택됩니다.")
    return 0 if ok == len(targets) else 1


if __name__ == "__main__":
    sys.exit(main())
