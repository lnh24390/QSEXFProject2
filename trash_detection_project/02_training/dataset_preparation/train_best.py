"""
최적 옵션으로 YOLO 학습을 실행하는 스크립트.

실행 방법 (bbox_data 폴더에서):
    uv run python 02_training/dataset_preparation/train_best.py                 # 권장 설정 (YOLO11n, batch=16)
    uv run python 02_training/dataset_preparation/train_best.py --dedicated     # 브라우저 등 모두 끈 전용 학습 (batch=24)
    uv run python 02_training/dataset_preparation/train_best.py --model yolo26n # 정확도 우선 (느리지만 재현율 높음)
    uv run python 02_training/dataset_preparation/train_best.py --quick         # 소량 세트로 3에포크만 동작 확인

설정 근거는 work/BENCH_REPORT.md 를 참고하세요. 모든 값은 RTX 4060 8GB 에서 실측한 결과입니다.
"""
# 현재 작업 디렉토리와 상관없이 공용 작업공간 경로를 찾는다.
from pathlib import Path as _WorkspacePath
import sys as _workspace_sys
_workspace_root = next(p for p in _WorkspacePath(__file__).resolve().parents if (p / 'workspace_paths.py').is_file())
_workspace_sys.path.insert(0, str(_workspace_root))
from workspace_paths import workspace_path


import argparse
import os

# ─────────────────────────────────────────────────────────────────────────────
# 경로 설정 — 여기만 바꾸면 다른 데이터셋에도 그대로 쓸 수 있습니다.
# ─────────────────────────────────────────────────────────────────────────────
BASE = str(workspace_path(''))

# 전체 학습용 데이터셋 (분할을 80/15/5 로 조정한 세트)
FULL_DATA = str(workspace_path('', f'YOLO_7CLASS_10000_PER_CLASS_20260909/data.yaml'))

# 동작 확인용 소량 세트 (클래스별 1,000장)
QUICK_DATA = str(workspace_path('', f'work/bench_1000/data.yaml'))

# 학습 결과 저장 폴더
RUNS = str(workspace_path('', f'work/runs'))


def main():
    p = argparse.ArgumentParser(description="최적 옵션 YOLO 학습")
    p.add_argument("--model", default="yolo11n",
                   help="yolo11n = 빠름(권장) / yolo26n = 정확도 우선(약 21%% 느림)")
    p.add_argument("--dedicated", action="store_true",
                   help="GPU 를 학습에만 쓸 때. batch 를 24(yolo11n)/32(yolo26n)로 올립니다")
    p.add_argument("--quick", action="store_true",
                   help="소량 세트로 3에포크만 실행해 설정이 정상인지 확인")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=None, help="직접 지정하면 자동 선택을 덮어씁니다")
    p.add_argument("--name", default=None, help="결과 폴더 이름")
    a = p.parse_args()

    # 작업 폴더를 고정합니다. 상대 경로로 저장되는 결과물의 위치가 흔들리지 않게 하기 위함입니다.
    os.chdir(str(workspace_path('', f'work')))
    import torch
    from ultralytics import YOLO

    # ─────────────────────────────────────────────────────────────────────────
    # GPU 확인 — 이 검사가 가장 중요합니다.
    #
    # 기존 yolo_test_app 환경은 torch 가 CPU 전용 빌드(+cpu)여서 RTX 4060 을
    # 전혀 쓰지 못했습니다. CPU 로 학습이 시작되면 수십 배 느려지므로,
    # 조용히 진행되지 않도록 여기서 즉시 중단시킵니다.
    # ─────────────────────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        raise SystemExit(
            f"GPU 를 쓸 수 없습니다 (torch {torch.__version__}).\n"
            "torch 가 CPU 전용 빌드이면 아래로 CUDA 빌드를 설치하세요:\n"
            "  uv pip install torch torchvision --index-url "
            "https://download.pytorch.org/whl/cu126"
        )
    print(f"GPU: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB)")

    # ─────────────────────────────────────────────────────────────────────────
    # batch 자동 선택
    #
    # 실측 (소량 세트, workers=4, 1에포크 제외 평균):
    #   YOLO11n  b8 25.22초 → b16 21.36초 → b24 20.95초 → b32 21.03초 → b48 21.44초
    #   YOLO26n  b8 32.04초 → b16 25.82초 → b24 24.37초 → b32 23.82초 → b48 24.19초
    #
    # 즉 8→16 에서 개선의 대부분(-15%)을 얻고, 그 뒤로는 거의 평평해집니다.
    # 32·48 은 더 빠르지 않습니다. GPU 연산이 이미 포화됐기 때문입니다.
    #
    # 기본값을 16 으로 둔 이유: batch 24 는 GPU 메모리를 4.62GB 쓰는데,
    # Windows 데스크톱(브라우저·Docker 등)이 상시 0.5~2GB 를 쓰므로
    # 8.19GB 중 여유가 1.5GB 밖에 남지 않아 OOM 위험이 있습니다.
    # batch 16 은 3.15GB 만 써서 약 3GB 의 여유를 남깁니다.
    # ─────────────────────────────────────────────────────────────────────────
    if a.batch is not None:
        batch = a.batch
    elif a.dedicated:
        batch = 32 if a.model.startswith("yolo26") else 24
    else:
        batch = 16

    data = QUICK_DATA if a.quick else FULL_DATA
    epochs = 3 if a.quick else a.epochs
    name = a.name or f"{a.model}_b{batch}" + ("_quick" if a.quick else "")

    print(f"모델 {a.model} / batch {batch} / epochs {epochs}")
    print(f"데이터 {data}")

    YOLO(str(workspace_path('work', f"{a.model}.pt"))).train(   # .pt = 사전학습 가중치. 처음부터 학습하는 것보다 훨씬 빨리 수렴합니다.
        data=data,
        epochs=epochs,
        imgsz=640,                # 데이터가 640×640 입니다. 낮추면 배터리·비닐 같은
                                  # 작고 얇은 물체의 인식률이 떨어지므로 유지합니다.
        batch=batch,
        workers=4,                # 실측상 4 / 8 / 12 의 차이가 1.4% 이내였습니다.
                                  # 병목이 데이터 공급이 아니라 GPU 연산이므로 올려도 소용없고,
                                  # 워커를 늘리면 RAM 만 더 씁니다. 다만 0 으로 두면 2.6배 느립니다.
        cache=False,              # RAM 캐시는 쓰지 않습니다. Windows 는 프로세스를 spawn 으로
                                  # 만들기 때문에 캐시 배열을 워커로 전달하다 실패합니다
                                  # (pickle data was truncated). workers=0 으로 우회할 수는 있지만
                                  # 그 경우가 오히려 2.3배 느립니다.
        device=0,                 # 0번 GPU. 'cpu' 로 조용히 넘어가지 않도록 명시합니다.
        amp=True,                 # 혼합 정밀도. 메모리와 연산 부담을 줄여줍니다. 항상 켜둡니다.
        pretrained=True,
        val=True,                 # 에포크마다 검증합니다. 조기 종료 판단에 필요합니다.
        patience=30,              # 30에포크 동안 검증 성능이 나아지지 않으면 중단합니다.
                                  # epochs 를 임의로 줄이는 것보다 이 방식이 안전합니다.
        plots=True,               # 학습 곡선·혼동행렬 그림을 저장합니다.
        seed=0,
        project=RUNS,             # 절대 경로로 지정합니다. 상대 경로를 주면 ultralytics 가
                                  # runs/detect/ 아래에 한 번 더 중첩시켜 경로가 헷갈립니다.
        name=name,
        exist_ok=True,
    )

    out = rf"{RUNS}\{name}"
    print(f"\n완료. 결과: {out}")
    print(f"  최고 가중치: {out}\\weights\\best.pt")
    print("\n최종 평가는 아직 쓰지 않은 test 분할로 별도 실행하세요:")
    print(f"  uv run python -c \"from ultralytics import YOLO; "
          f"YOLO(r'{out}\\weights\\best.pt').val(data=r'{data}', split='test')\"")


if __name__ == "__main__":
    # Windows 에서는 이 가드가 반드시 있어야 합니다.
    # 데이터로더 워커가 프로세스를 spawn 으로 새로 만들면서 이 파일을 다시 읽는데,
    # 가드가 없으면 학습이 무한히 중첩 실행되며 오류가 납니다.
    main()
