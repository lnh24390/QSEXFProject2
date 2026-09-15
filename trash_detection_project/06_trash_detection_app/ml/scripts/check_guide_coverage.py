"""모델의 클래스가 앱 분리수거 매핑(recycling_guide.json)에 빠짐없이 들어있는지 검사.

사용 예)
  python scripts/check_guide_coverage.py --weights runs/trash_yolo26n/weights/best.pt
  python scripts/check_guide_coverage.py --data configs/trash.yaml
  python scripts/check_guide_coverage.py --tflite ../app/assets/models/yolo26n.tflite

fallback(general) 으로 떨어지는 클래스가 있으면 종료 코드 1 을 반환합니다(CI 용).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from common import die, info, load_guide, model_class_names, resolve_guide_category, warn


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--weights", type=Path, help=".pt / .onnx")
    g.add_argument("--tflite", type=Path)
    g.add_argument("--data", type=Path, help="data.yaml (names 사용)")
    p.add_argument("--strict", action="store_true", help="규칙(rule) 매핑도 실패로 간주")
    args = p.parse_args()

    if args.data:
        names_map = yaml.safe_load(args.data.read_text(encoding="utf-8"))["names"]
        names = [names_map[k] for k in sorted(names_map)] if isinstance(names_map, dict) else list(names_map)
    else:
        path = args.weights or args.tflite
        if not path.exists():
            die(f"파일이 없습니다: {path}")
        names = model_class_names(path)
    if not names:
        die("클래스명을 읽지 못했습니다.")

    guide = load_guide()
    bad = []
    for n in names:
        cat, how = resolve_guide_category(guide, n)
        flag = how == "fallback" or (args.strict and how == "rule")
        info(f"  {'!' if flag else ' '} {n:28s} → {cat:10s} ({how})")
        if flag:
            bad.append(n)
    if bad:
        warn(f"매핑 필요 {len(bad)}개: {bad}")
        warn("app/assets/config/recycling_guide.json 의 classes 에 추가하세요.")
        sys.exit(1)
    ignored = [n for n in names if resolve_guide_category(guide, n)[1] == "ignored"]
    info(f"모든 클래스가 매핑되어 있습니다. (무시 {len(ignored)}개, 안내 대상 {len(names) - len(ignored)}개)")


if __name__ == "__main__":
    main()
