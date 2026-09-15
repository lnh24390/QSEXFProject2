"""재학습 없이 추론 방식만 바꿨을 때의 성능 측정.

한 번에 다음을 출력합니다.
  1) 혼동 행렬(실제 클래스별 맞춤 / 배경으로 놓침 / 가장 많이 헷갈린 클래스)
  2) 입력 해상도(--imgsz 목록)와 TTA 별 mAP50 / mAP50-95 / 클래스별 mAP50-95 / 추론 시간
  3) 클래스별 F1 최대 신뢰도 임계값 → app/lib/config/app_config.dart 의 classConfidenceThresholds 에 붙여넣을 수 있는 형식

결과는 RECOGNITION_IMPROVEMENT_REPORT.md 의 측정 방법과 같습니다.
.pt 평가는 Windows 에서도 되지만, .tflite 평가는 Linux(WSL) 에서 실행하세요.

사용 예)
  python scripts/eval_inference.py --weights weights/lnh_10000_best.pt --data <data.yaml>
  python scripts/eval_inference.py --weights weights/lnh_10000_best.pt --data <data.yaml> --imgsz 640 800 960 --tta
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from common import die, info, normalize_class_name


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, required=True, help=".pt 또는 .tflite")
    p.add_argument("--data", type=Path, required=True, help="평가용 data.yaml (val 항목 사용)")
    p.add_argument("--imgsz", type=int, nargs="+", default=[640], help="비교할 입력 해상도 목록")
    p.add_argument("--tta", action="store_true", help="첫 번째 해상도로 TTA(augment=True)도 측정")
    p.add_argument("--app-threshold", type=float, default=0.45, help="비교 기준이 되는 현재 앱 전체 임계값")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def run_val(model, args: argparse.Namespace, imgsz: int, augment: bool = False, plots: bool = False):
    # ultralytics 는 plots=True 일 때만 혼동 행렬을 채운다
    return model.val(data=str(args.data), imgsz=imgsz, batch=args.batch, device=args.device,
                     augment=augment, plots=plots, verbose=False,
                     project=str(Path("runs") / "eval_inference"), name=f"{imgsz}{'_tta' if augment else ''}",
                     exist_ok=True)


def print_confusion(r, names: list[str]) -> None:
    cm = r.confusion_matrix.matrix  # [pred, true], 마지막 = background
    nc = len(names)
    info("== 1) 혼동 행렬: 실제 클래스별 ==")
    for j in range(nc):
        total = cm[:, j].sum()
        if total == 0:
            continue
        others = [(cm[i, j] / total * 100, names[i]) for i in range(nc) if i != j]
        worst = max(others)
        info(f"  {names[j]:>10s}: 맞춤 {cm[j, j] / total * 100:5.1f}%  놓침(배경) {cm[nc, j] / total * 100:5.1f}%"
             f"  혼동 최다 {worst[1]} {worst[0]:.1f}%")
    fp = cm[:nc, nc]
    if fp.sum() > 0:
        info("  배경을 잘못 잡은 비율(오검출): " + ", ".join(
            f"{names[i]} {fp[i] / fp.sum() * 100:.1f}%" for i in np.argsort(-fp) if fp[i] > 0))


def print_class_thresholds(r, names: list[str], app_threshold: float) -> None:
    b = r.box
    px = np.asarray(b.px)
    i_app = int(np.argmin(np.abs(px - app_threshold)))
    info(f"== 3) 클래스별 F1 최대 신뢰도 (현재 {app_threshold:.2f} 과 비교) ==")
    best: dict[str, float] = {}
    for c, name in enumerate(names):
        f1 = np.asarray(b.f1_curve)[c]
        p = np.asarray(b.p_curve)[c]
        rc = np.asarray(b.r_curve)[c]
        k = int(f1.argmax())
        best[normalize_class_name(name)] = round(float(px[k]), 2)
        info(f"  {name:>10s}: best {px[k]:.2f} F1 {f1[k]:.3f} (P {p[k]:.2f} R {rc[k]:.2f})"
             f" | @{app_threshold:.2f} F1 {f1[i_app]:.3f} (P {p[i_app]:.2f} R {rc[i_app]:.2f})")
    info("  app_config.dart classConfidenceThresholds:")
    for k, v in sorted(best.items(), key=lambda kv: kv[1]):
        info(f"    '{k}': {v:.2f},")


def main() -> None:
    args = parse_args()
    if not args.weights.exists():
        die(f"가중치 파일이 없습니다: {args.weights}")
    if not args.data.exists():
        die(f"data.yaml 이 없습니다: {args.data}")

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    names = [model.names[i] for i in range(len(model.names))]

    base = run_val(model, args, args.imgsz[0], plots=True)
    print_confusion(base, names)

    info("== 2) 입력 해상도 / TTA 별 성능 ==")
    info(f"  {'setting':>10s} {'mAP50':>6s} {'mAP50-95':>8s} {'ms/img':>7s} | " + " ".join(f"{n:>8s}" for n in names))
    rows = [(str(args.imgsz[0]), base)]
    for s in args.imgsz[1:]:
        rows.append((str(s), run_val(model, args, s)))
    if args.tta:
        rows.append((f"{args.imgsz[0]}+TTA", run_val(model, args, args.imgsz[0], augment=True)))
    for label, r in rows:
        per = r.box.maps
        info(f"  {label:>10s} {r.box.map50:6.3f} {r.box.map:8.3f} {r.speed['inference']:7.1f} | "
             + " ".join(f"{per[i]:8.3f}" for i in range(len(names))))

    print_class_thresholds(base, names, args.app_threshold)


if __name__ == "__main__":
    sys.exit(main())
