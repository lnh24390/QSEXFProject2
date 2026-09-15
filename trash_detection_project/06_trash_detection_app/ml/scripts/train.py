"""YOLO 쓰레기 검출 모델 학습(전이학습).

사용 예)
  # TACO 변환 결과로 yolo26n 파인튜닝
  python scripts/train.py --data datasets/taco_yolo/data.yaml --model yolo26n.pt --epochs 100

  # 이어서 학습
  python scripts/train.py --resume runs/trash_yolo26n/weights/last.pt

  # 학습 후 바로 Android 용으로 변환 + 앱에 복사 (Linux/macOS 또는 --docker)
  python scripts/train.py --data ... --export --docker

결과: runs/<name>/weights/best.pt
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import ML_DIR, RUNS_DIR, die, info


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, help="data.yaml 경로")
    p.add_argument("--model", default="yolo26n.pt", help="시작 가중치 (yolo26n/s/m.pt 또는 이전 best.pt)")
    p.add_argument("--name", default=None, help="runs/ 아래 실험 이름 (기본: trash_<model>)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640, help="학습 해상도. 앱 export imgsz 와 맞추세요.")
    p.add_argument("--batch", type=int, default=16, help="-1 이면 자동")
    p.add_argument("--device", default=None, help="'0'(GPU), 'cpu', '0,1' … 기본 자동")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=30, help="early stopping")
    p.add_argument("--freeze", type=int, default=None, help="앞쪽 N개 레이어 동결 (예: 10 = 백본)")
    p.add_argument("--lr0", type=float, default=None)
    p.add_argument("--resume", type=Path, help="last.pt 로 이어서 학습")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--export", action="store_true", help="학습 후 export.py 실행")
    p.add_argument("--docker", action="store_true", help="export 시 Docker 사용(Windows)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    if args.resume:
        if not args.resume.exists():
            die(f"resume 가중치가 없습니다: {args.resume}")
        info(f"이어서 학습: {args.resume}")
        model = YOLO(str(args.resume))
        results = model.train(resume=True)
        best = Path(results.save_dir) / "weights" / "best.pt"
    else:
        if not args.data:
            die("--data 를 지정하세요 (또는 --resume).")
        if not args.data.exists():
            die(f"data.yaml 이 없습니다: {args.data}")
        name = args.name or f"trash_{Path(args.model).stem}"
        info(f"학습 시작: model={args.model} data={args.data} epochs={args.epochs} imgsz={args.imgsz}")
        model = YOLO(args.model)
        train_kwargs = dict(
            data=str(args.data),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            patience=args.patience,
            project=str(RUNS_DIR),
            name=name,
            exist_ok=True,
            seed=args.seed,
            deterministic=True,
            plots=True,
        )
        if args.device is not None:
            train_kwargs["device"] = args.device
        if args.freeze is not None:
            train_kwargs["freeze"] = args.freeze
        if args.lr0 is not None:
            train_kwargs["lr0"] = args.lr0
        results = model.train(**train_kwargs)
        best = Path(results.save_dir) / "weights" / "best.pt"

    if not best.exists():
        die(f"best.pt 가 생성되지 않았습니다: {best}")
    info(f"학습 완료. best.pt: {best}")

    # 검증 지표 출력
    try:
        metrics = YOLO(str(best)).val(data=str(args.data) if args.data else None, imgsz=args.imgsz, plots=False)
        info(f"mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}")
    except Exception as e:  # noqa: BLE001
        info(f"검증 생략: {e}")

    if args.export:
        cmd = [sys.executable, str(ML_DIR / "scripts" / "export.py"), "--weights", str(best),
               "--format", "litert", "--imgsz", str(args.imgsz), "--install-to-app"]
        if args.docker:
            cmd.append("--docker")
        info("변환 실행: " + " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
