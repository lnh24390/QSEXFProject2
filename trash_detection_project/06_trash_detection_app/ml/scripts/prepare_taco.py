"""TACO(COCO 형식) 데이터셋 → YOLO 형식 변환.

사용 예)
  python scripts/prepare_taco.py --taco-zip "C:/Users/pc/Desktop/TACO-master.zip" --download
  python scripts/prepare_taco.py --taco-dir datasets/TACO-master

동작:
  1. TACO-master.zip 이 주어지면 datasets/ 에 압축 해제
  2. data/annotations.json 을 읽어 configs/taco_class_map.yaml 로 클래스를 합침
  3. (--download) 이미지가 없으면 flickr URL 에서 내려받음
  4. train/val 로 나눠 datasets/taco_yolo/{images,labels}/{train,val} 에 저장
  5. datasets/taco_yolo/data.yaml 생성 (train.py 에서 그대로 사용)
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from tqdm import tqdm

from common import CONFIGS_DIR, DATASETS_DIR, die, info, warn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--taco-zip", type=Path, help="TACO-master.zip 경로")
    src.add_argument("--taco-dir", type=Path, help="이미 압축 해제된 TACO-master 폴더")
    p.add_argument("--out", type=Path, default=DATASETS_DIR / "taco_yolo", help="출력 폴더")
    p.add_argument("--class-map", type=Path, default=CONFIGS_DIR / "taco_class_map.yaml")
    p.add_argument("--names-yaml", type=Path, default=CONFIGS_DIR / "trash.yaml", help="클래스 순서를 정의한 data.yaml")
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--download", action="store_true", help="이미지가 없으면 flickr URL 에서 다운로드")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--copy", action="store_true", help="이미지를 복사(기본은 하드링크 시도 후 복사)")
    return p.parse_args()


def extract_zip(zip_path: Path) -> Path:
    if not zip_path.exists():
        die(f"zip 파일이 없습니다: {zip_path}")
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    target = DATASETS_DIR / "TACO-master"
    if target.exists():
        info(f"이미 압축 해제됨: {target}")
        return target
    info(f"압축 해제 중: {zip_path} → {DATASETS_DIR}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATASETS_DIR)
    if not target.exists():
        # zip 내부 최상위 폴더명이 다를 수 있음
        candidates = [d for d in DATASETS_DIR.iterdir() if d.is_dir() and (d / "data" / "annotations.json").exists()]
        if not candidates:
            die("압축 해제 후 data/annotations.json 을 찾지 못했습니다.")
        target = candidates[0]
    return target


def download_image(url: str, dest: Path) -> bool:
    import requests

    if dest.exists():
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return True
    except Exception as e:  # noqa: BLE001
        warn(f"다운로드 실패 {url}: {e}")
        return False


def link_or_copy(src: Path, dst: Path, force_copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if not force_copy:
        try:
            import os

            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    taco_dir = extract_zip(args.taco_zip) if args.taco_zip else args.taco_dir
    ann_path = taco_dir / "data" / "annotations.json"
    if not ann_path.exists():
        die(f"annotations.json 이 없습니다: {ann_path}")

    names_cfg = yaml.safe_load(args.names_yaml.read_text(encoding="utf-8"))
    names: dict[int, str] = {int(k): v for k, v in names_cfg["names"].items()}
    name_to_id = {v: k for k, v in names.items()}

    class_map: dict[str, str | None] = yaml.safe_load(args.class_map.read_text(encoding="utf-8"))
    for taco_name, target in class_map.items():
        if target is not None and target not in name_to_id:
            die(f"taco_class_map.yaml 의 '{taco_name}: {target}' 가 {args.names_yaml.name} names 에 없습니다.")

    info(f"annotations 로드: {ann_path}")
    coco = json.loads(ann_path.read_text(encoding="utf-8"))
    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    unmapped = sorted({n for n in cat_id_to_name.values() if n not in class_map})
    if unmapped:
        warn(f"class_map 에 없는 TACO 카테고리 {len(unmapped)}개는 제외됩니다: {unmapped}")

    images = {im["id"]: im for im in coco["images"]}
    anns_by_image: dict[int, list] = {}
    for a in coco["annotations"]:
        anns_by_image.setdefault(a["image_id"], []).append(a)

    # 이미지 준비 (다운로드)
    data_dir = taco_dir / "data"
    if args.download:
        info("이미지 다운로드 확인 중…")
        jobs = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for im in images.values():
                dest = data_dir / im["file_name"]
                if not dest.exists():
                    url = im.get("flickr_url") or im.get("coco_url")
                    if url:
                        jobs.append(ex.submit(download_image, url, dest))
            for _ in tqdm(as_completed(jobs), total=len(jobs), desc="download"):
                pass

    # 분할
    image_ids = sorted(images)
    random.shuffle(image_ids)
    n_val = int(len(image_ids) * args.val_ratio)
    split = {"val": set(image_ids[:n_val]), "train": set(image_ids[n_val:])}

    out = args.out
    for s in ("train", "val"):
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)

    stats = {n: 0 for n in names.values()}
    written = {"train": 0, "val": 0}
    skipped_missing = 0
    for s, ids in split.items():
        for iid in tqdm(sorted(ids), desc=f"convert {s}"):
            im = images[iid]
            src = data_dir / im["file_name"]
            if not src.exists():
                skipped_missing += 1
                continue
            w, h = im["width"], im["height"]
            lines = []
            for a in anns_by_image.get(iid, []):
                target = class_map.get(cat_id_to_name[a["category_id"]])
                if target is None:
                    continue
                x, y, bw, bh = a["bbox"]
                if bw <= 0 or bh <= 0:
                    continue
                cx = (x + bw / 2) / w
                cy = (y + bh / 2) / h
                lines.append(f"{name_to_id[target]} {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}")
                stats[target] += 1
            stem = f"{iid:06d}_{Path(im['file_name']).stem}"
            link_or_copy(src, out / "images" / s / f"{stem}{src.suffix.lower()}", args.copy)
            (out / "labels" / s / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            written[s] += 1

    data_yaml = {
        "path": str(out.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8")

    info(f"완료: train {written['train']}장, val {written['val']}장 (이미지 없음으로 건너뜀 {skipped_missing}장)")
    if skipped_missing and not args.download:
        warn("이미지가 없는 항목이 있습니다. --download 옵션으로 내려받으세요.")
    info("클래스별 박스 수:")
    for n, c in stats.items():
        info(f"  {n:20s} {c}")
    info(f"data.yaml: {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
