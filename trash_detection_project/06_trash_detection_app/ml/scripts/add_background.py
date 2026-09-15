"""배경(negative) 이미지를 YOLO 데이터셋에 추가한다.

배경 이미지는 라벨 파일이 비어 있는(0바이트) 이미지로, "여기엔 아무것도 없다"를 모델에게
가르친다. 이게 없으면 모델은 화면에 뭔가 보이기만 하면 기존 클래스 중 하나로 밀어붙인다
(모니터를 paper 로 75% 확신하는 식). Ultralytics 는 전체의 ~10% 를 권장한다.

소스는 두 가지를 섞어 쓸 수 있다.
  --from-dir   직접 찍은 사진 폴더. 실제 사용 환경과 도메인이 같아 효과가 가장 크다.
  --from-coco  COCO val2017 에서 쓰레기와 무관한 이미지만 골라 자동으로 내려받는다.
               (annotations 만 받아 대상 이미지를 고른 뒤, 고른 것만 개별 URL 로 받는다)

기존 파일은 건드리지 않는다. 추가되는 파일은 전부 `bg_` 접두어가 붙으므로
되돌리려면 `bg_*` 만 지우면 된다.

사용 예)
  # COCO 에서 300장
  python scripts/add_background.py --dataset C:/.../bench_1000 --from-coco 300

  # 직접 찍은 사진 폴더에서 전부
  python scripts/add_background.py --dataset C:/.../bench_1000 --from-dir C:/Users/USER/Desktop/bg_photos

  # 둘 다
  python scripts/add_background.py --dataset C:/.../bench_1000 --from-coco 300 --from-dir C:/.../bg_photos

  # 무엇이 추가될지만 확인
  python scripts/add_background.py --dataset C:/.../bench_1000 --from-coco 50 --dry-run
"""
from __future__ import annotations

import argparse
import io
import json
import random
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from common import DATASETS_DIR, die, info, warn

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_ANNOTATIONS_MEMBER = "annotations/instances_val2017.json"

# 우리 7개 클래스(paper/plastic/can/glass/battery/vinyl/general)와 헷갈릴 수 있는 COCO 카테고리.
# 이런 게 하나라도 들어 있는 이미지는 배경으로 쓰지 않는다. 배경 이미지에 사실은 쓰레기가
# 찍혀 있으면 "이건 검출하지 마라"를 잘못 가르치게 되기 때문이다.
COCO_TRASHY_CATEGORIES = {
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake",
    "book", "vase", "scissors", "toothbrush", "hair drier",
    "cell phone", "laptop", "mouse", "remote", "keyboard",
    "backpack", "umbrella", "handbag", "suitcase", "tie",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, required=True,
                   help="YOLO 데이터셋 루트 (images/train, labels/train … 구조)")
    p.add_argument("--from-dir", type=Path, default=None, help="직접 찍은 배경 사진 폴더")
    p.add_argument("--from-coco", type=int, default=0, help="COCO val2017 에서 가져올 장수")
    p.add_argument("--val-ratio", type=float, default=0.15,
                   help="추가분 중 val 로 보낼 비율 (기본 0.15)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true", help="실제로 복사하지 않고 계획만 출력")
    return p.parse_args()


def check_dataset(root: Path) -> None:
    if not root.exists():
        die(f"데이터셋 폴더가 없습니다: {root}")
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            d = root / kind / split
            if not d.is_dir():
                die(f"{d} 가 없습니다. YOLO 표준 구조(images/train, labels/train …)가 아닙니다.")


def existing_background_count(root: Path) -> dict[str, int]:
    out = {}
    for split in ("train", "val"):
        label_dir = root / "labels" / split
        out[split] = sum(1 for p in label_dir.glob("*.txt") if p.stat().st_size == 0)
    return out


def collect_from_dir(src: Path) -> list[Path]:
    if not src.is_dir():
        die(f"--from-dir 폴더가 없습니다: {src}")
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        warn(f"{src} 에 이미지가 없습니다.")
    else:
        info(f"직접 촬영 이미지 {len(files)}장 발견: {src}")
    return files


def download_coco_annotations(cache_dir: Path) -> Path:
    """instances_val2017.json 을 캐시에 준비한다."""
    target = cache_dir / "instances_val2017.json"
    if target.exists():
        info(f"COCO annotations 캐시 사용: {target}")
        return target
    cache_dir.mkdir(parents=True, exist_ok=True)
    info(f"COCO annotations 내려받는 중 (약 241MB): {COCO_ANNOTATIONS_URL}")
    with urllib.request.urlopen(COCO_ANNOTATIONS_URL) as resp:
        blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        with zf.open(COCO_ANNOTATIONS_MEMBER) as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)
    info(f"저장: {target}")
    return target


def pick_coco_backgrounds(ann_path: Path, count: int, seed: int) -> list[tuple[str, str]]:
    """쓰레기와 무관한 이미지 (file_name, url) 목록을 고른다."""
    info("COCO annotations 파싱 중…")
    with open(ann_path, encoding="utf-8") as f:
        data = json.load(f)

    trashy_ids = {c["id"] for c in data["categories"] if c["name"] in COCO_TRASHY_CATEGORIES}
    missing = COCO_TRASHY_CATEGORIES - {c["name"] for c in data["categories"]}
    if missing:
        warn(f"COCO 에 없는 카테고리명(무시): {sorted(missing)}")

    dirty: set[int] = set()
    for a in data["annotations"]:
        if a["category_id"] in trashy_ids:
            dirty.add(a["image_id"])

    clean = [img for img in data["images"] if img["id"] not in dirty]
    info(f"쓰레기 관련 물체가 없는 이미지 {len(clean)}장 / 전체 {len(data['images'])}장")
    if len(clean) < count:
        warn(f"요청 {count}장보다 적어 {len(clean)}장만 씁니다.")
        count = len(clean)

    random.Random(seed).shuffle(clean)
    return [(img["file_name"], img["coco_url"]) for img in clean[:count]]


def download_coco_images(picks: list[tuple[str, str]], cache_dir: Path) -> list[Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for i, (name, url) in enumerate(picks, 1):
        dst = cache_dir / name
        if not dst.exists():
            try:
                with urllib.request.urlopen(url) as resp, open(dst, "wb") as f:
                    shutil.copyfileobj(resp, f)
            except Exception as e:  # noqa: BLE001
                warn(f"{name} 내려받기 실패: {e}")
                continue
        out.append(dst)
        if i % 50 == 0 or i == len(picks):
            info(f"  이미지 {i}/{len(picks)}")
    return out


def install(images: list[Path], root: Path, val_ratio: float, seed: int, dry_run: bool) -> None:
    if not images:
        warn("추가할 이미지가 없습니다.")
        return

    shuffled = list(images)
    random.Random(seed).shuffle(shuffled)
    n_val = int(len(shuffled) * val_ratio)
    plan = [("val", p) for p in shuffled[:n_val]] + [("train", p) for p in shuffled[n_val:]]

    info(f"추가 계획: train {len(shuffled) - n_val}장, val {n_val}장")
    if dry_run:
        for split, p in plan[:10]:
            info(f"  [{split}] {p.name}")
        if len(plan) > 10:
            info(f"  … 외 {len(plan) - 10}장")
        info("--dry-run 이므로 파일을 만들지 않았습니다.")
        return

    added = {"train": 0, "val": 0}
    for split, src in plan:
        stem = f"bg_{src.stem}"
        img_dst = root / "images" / split / f"{stem}{src.suffix.lower()}"
        lbl_dst = root / "labels" / split / f"{stem}.txt"
        if img_dst.exists():
            continue
        shutil.copy2(src, img_dst)
        lbl_dst.write_bytes(b"")  # 빈 라벨 = 배경
        added[split] += 1
    info(f"추가 완료: train {added['train']}장, val {added['val']}장")


def main() -> None:
    args = parse_args()
    if not args.from_dir and args.from_coco <= 0:
        die("--from-dir 또는 --from-coco 중 최소 하나를 지정하세요.")

    root = args.dataset.resolve()
    check_dataset(root)

    before = existing_background_count(root)
    total_train = len(list((root / "images" / "train").iterdir()))
    info(f"대상 데이터셋: {root}")
    info(f"현재 train {total_train}장, 그중 배경 {before['train']}장 / val 배경 {before['val']}장")

    images: list[Path] = []
    if args.from_dir:
        images += collect_from_dir(args.from_dir.resolve())
    if args.from_coco > 0:
        cache = DATASETS_DIR / "_background_cache"
        ann = download_coco_annotations(cache)
        picks = pick_coco_backgrounds(ann, args.from_coco, args.seed)
        images += download_coco_images(picks, cache / "coco_val2017")

    install(images, root, args.val_ratio, args.seed, args.dry_run)

    if not args.dry_run:
        after = existing_background_count(root)
        new_total = len(list((root / "images" / "train").iterdir()))
        ratio = after["train"] / new_total * 100 if new_total else 0
        info(f"최종: train {new_total}장, 배경 {after['train']}장 ({ratio:.1f}%)")
        if ratio < 5:
            warn("배경 비율이 5% 미만입니다. 10% 안팎을 권장합니다.")
        info("되돌리려면 images/*/bg_* 와 labels/*/bg_* 를 지우세요.")


if __name__ == "__main__":
    sys.exit(main())
