"""
최적 설정 YOLO 학습 — 검출(bbox)과 세그멘테이션(seg) 둘 다 지원합니다.

    uv run python 02_training/baseline_training/train.py            # 검출 학습
    uv run python 02_training/baseline_training/train.py seg        # 세그멘테이션 학습
    uv run python 02_training/baseline_training/train.py both       # 검출 -> 세그 순서로 연속 학습
    uv run python 02_training/baseline_training/train.py nobg       # 배경 이미지를 뺀 검출 -> 세그 (비교용)
    uv run python 02_training/baseline_training/train.py bbox_nobg  # 배경 제외 검출만 (seg_nobg 도 가능)

설정을 바꾸려면 아래 '설정' 부분만 고치면 됩니다.
값의 근거는 work/BENCH_REPORT.md 에 실측 기록이 있습니다.
"""
# 현재 작업 디렉토리와 상관없이 공용 작업공간 경로를 찾는다.
from pathlib import Path as _WorkspacePath
import sys as _workspace_sys
_workspace_root = next(p for p in _WorkspacePath(__file__).resolve().parents if (p / 'workspace_paths.py').is_file())
_workspace_sys.path.insert(0, str(_workspace_root))
from workspace_paths import workspace_path
from workspace_trainers import trainer_for


import sys
from ultralytics import YOLO
import torch
from metrics_logger import attach

# ═══════════════════════════════════════════════════════════════════════
#  설정
# ═══════════════════════════════════════════════════════════════════════

BASE = str(workspace_path(''))

TASKS = {
    # 검출 — 박스를 찾습니다
    "bbox": dict(
        model="yolo11n.pt",
        data=str(workspace_path('', f'YOLO_7CLASS_10000_PER_CLASS_20260909/data.yaml')),
        name="bbox_yolo11n",
    ),
    # 세그멘테이션 — 외곽선을 찾습니다. 반드시 '-seg' 가중치를 써야 합니다.
    "seg": dict(
        model="yolo11n-seg.pt",
        data=str(workspace_path('', f'YOLO_SEG_7CLASS_10000_PER_CLASS_20260910_yolo_seg/data.yaml')),
        name="seg_yolo11n",
    ),
    # 배경 제외 학습 — 배경 이미지(bg_coco_*)를 train·val·test 에서 모두 뺀 라벨 데이터만 사용.
    # 파일을 옮기지 않고 data_nobg 의 이미지 목록 파일로 고른다.
    # 모델은 요청에 따라 yolo26s(소형)를 쓴다 (2026-09-11 yolo26l → yolo26m → yolo26s 순으로 낮춤, 속도 우선). 위의 yolo11n 학습과 모델이 다르므로
    # 두 결과의 차이에는 배경 효과와 모델 크기 효과가 섞인다.
    #
    # batch=16 은 사용자 요청(2026-09-11)이다. (24 로 했다가 메모리 여유를 위해 16 으로 낮춤)
    # 검증은 ultralytics 가 batch 의 2배(32)로 한다.
    # yolo26s 는 연산량이 yolo11n 의 약 3배라 8GB 에서 메모리가 모자랄 수 있다.
    # ultralytics 는 첫 에포크에서 메모리가 부족하면 batch 를 줄여 최대 3번 다시 시도한다.
    # 실제로 쓴 batch 는 config.txt 에 기록된다. 첫 에포크 이후에 부족하면 학습이 멈추므로
    # 그때는 last.pt 로 이어서 하되 batch 를 낮춰야 한다.
    "bbox_nobg": dict(
        model="yolo26s.pt",
        data=str(workspace_path('', f'best_train/data_nobg/bbox_nobg.yaml')),
        name="bbox_yolo26s_nobg",
        batch=16,
    ),
    "seg_nobg": dict(
        model="yolo26s-seg.pt",
        data=str(workspace_path('', f'best_train/data_nobg/seg_nobg.yaml')),
        name="seg_yolo26s_nobg",
        batch=16,
    ),
}

BATCH = 16        # GPU 를 학습에만 쓸 수 있으면 24. 세그는 메모리를 더 쓰므로 16 을 권장합니다.
EPOCHS = 100      # 상한입니다. 아래 PATIENCE 로 대개 더 일찍 끝납니다.
PATIENCE = 10     # 조기 종료: 10에포크 동안 성능이 나아지지 않으면 중단합니다.
                  # 2026-09-12 사용자 요청으로 30 → 10. 이미 끝난 yolo11n 학습 2개(bbox, seg)는 30 으로 했다.

# ═══════════════════════════════════════════════════════════════════════
#  GPU 확인 — 가장 중요합니다
#
#  torch 가 CPU 전용 빌드면 CPU 로 조용히 학습이 시작되는데, 실측 결과
#  CPU 는 GPU 보다 25.8배 느립니다 (에포크 550.7초 vs 21.4초).
#  100에포크 기준 9일 vs 8.4시간이므로, 여기서 멈추게 합니다.
# ═══════════════════════════════════════════════════════════════════════

if not torch.cuda.is_available():
    raise SystemExit(
        f"GPU 를 쓸 수 없습니다 (torch {torch.__version__}).\n"
        "버전 끝이 '+cpu' 이면 CUDA 빌드로 교체하세요:\n"
        "  uv pip install torch torchvision "
        "--index-url https://download.pytorch.org/whl/cu126"
    )


def run(key):
    t = TASKS[key]
    batch = t.get("batch", BATCH)      # 과제별 batch 가 있으면 그 값을 쓴다 (yolo26s 는 16)
    print(f"\n{'='*62}\n {key} 학습 시작 — {t['model']} / batch={batch}\n{'='*62}")

    model = YOLO(str(workspace_path(t["model"])))
    kind = "배경 이미지 제외" if key.endswith("_nobg") else "배경 이미지 포함"
    attach(model, note=f"{key} 학습 ({t['model']}, {kind})")   # 지표를 텍스트로 기록

    model.train(
        data=t["data"],
        epochs=EPOCHS,
        patience=PATIENCE,  # 조기 종료. epochs 를 임의로 줄이는 것보다 안전합니다.
                            # 성능이 계속 오르면 끝까지 돌고, 멈추면 알아서 끊습니다.

        batch=batch,        # 실측(yolo11n): 8→16 에서 15% 빨라지고 그 뒤로는 거의 평평합니다.
                            # 32·48 은 오히려 느립니다 (24에서 GPU 연산이 포화).
                            # 16 을 기본으로 둔 이유는 메모리입니다. batch 24 는 4.62GB 를
                            # 쓰는데 8.19GB 중 데스크톱이 0.5~2GB 를 상시 쓰므로 여유가
                            # 적습니다. batch 16 은 3.15GB 만 씁니다.

        imgsz=640,          # 데이터가 640×640 입니다. 낮추면 배터리·비닐처럼 작고 얇은
                            # 물체의 인식률이 떨어지므로 그대로 둡니다.

        workers=4,          # 실측: 4 / 8 / 12 차이가 1.4% 이내였습니다. 병목이 데이터 공급이
                            # 아니라 GPU 연산이라 올려도 소용없습니다. 단 0 은 2.6배 느립니다.

        cache=False,        # RAM 캐시는 쓰지 않습니다. Windows 에서는 캐시 배열을 워커
                            # 프로세스로 넘기다 실패하고, workers=0 으로 우회하면
                            # 오히려 2.3배 느려집니다.

        device=0,           # 0번 GPU. 명시하지 않으면 CPU 로 넘어갈 수 있습니다.
        amp=True,           # 혼합 정밀도. 메모리와 연산 부담을 줄입니다. 항상 켭니다.
        plots=True,         # 학습 곡선·혼동행렬을 저장합니다.
        seed=0,
        project=str(workspace_path('', f'best_train/runs')),
        name=t["name"],
        exist_ok=True,
    )

    out = str(workspace_path('', f"best_train/runs/{t['name']}"))
    print(f"\n{key} 완료: {out}")
    print(f"  가중치   : {out}\\weights\\best.pt")
    print(f"  에포크별 : {out}\\metrics.txt")
    print(f"  요약     : {out}\\summary.txt")


def resume(key):
    """중단된 학습을 마지막 저장 지점(last.pt)부터 이어서 한다.

        uv run python 02_training/baseline_training/train.py seg resume

    last.pt 에는 에포크 번호·옵티마이저·학습률 상태가 함께 저장돼 있어,
    설정을 다시 주지 않아도 끊긴 다음 에포크부터 똑같이 이어집니다.
    """
    t = TASKS[key]
    last = str(workspace_path('', f"best_train/runs/{t['name']}/weights/last.pt"))
    print(f"\n{'='*62}\n {key} 이어서 학습 — {last}\n{'='*62}")
    model = YOLO(last)
    attach(model, note=f"{key} 이어서 학습")
    model.train(trainer=trainer_for('seg' if key.startswith('seg') else 'bbox'), resume=True,
                data=t['data'], project=str(workspace_path('best_train/runs')),
                save_dir=str(workspace_path('best_train/runs', t['name'])))


if __name__ == "__main__":
    # Windows 필수. 데이터로더가 프로세스를 새로 만들 때 이 파일을 다시 읽으므로,
    # 이 가드가 없으면 학습이 중첩 실행되며 오류가 납니다.
    arg = sys.argv[1] if len(sys.argv) > 1 else "bbox"
    if len(sys.argv) > 2 and sys.argv[2] == "resume":
        resume(arg)
        sys.exit()
    groups = {"both": ["bbox", "seg"],                  # 배경 포함 검출 -> 세그
              "nobg": ["bbox_nobg", "seg_nobg"]}        # 배경 제외 검출 -> 세그 (비교용)
    for k in groups.get(arg, [arg]):
        run(k)
    print("\n최종 평가는 학습에 쓰지 않은 test 분할로 따로 실행하세요:")
    print("  uv run python 02_training/baseline_training/test.py")
