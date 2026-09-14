"""COCO 배경 이미지를 두 데이터셋에 동일한 분할로 추가한다.

배경 이미지는 라벨 파일을 '빈 .txt' 로 둔다. YOLO 는 이를 '검출할 물체가 없는 사진'
으로 학습해 오탐(false positive)을 줄인다. 검출·세그멘테이션 모두 같은 방식이다.
"""
# 현재 작업 디렉토리와 상관없이 공용 작업공간 경로를 찾는다.
from pathlib import Path as _WorkspacePath
import sys as _workspace_sys
_workspace_root = next(p for p in _WorkspacePath(__file__).resolve().parents if (p / 'workspace_paths.py').is_file())
_workspace_sys.path.insert(0, str(_workspace_root))
from workspace_paths import workspace_path

import os, csv, sys, random, collections

SRC = str(workspace_path('work/coco/bg_640'))
BB  = str(workspace_path('YOLO_7CLASS_10000_PER_CLASS_20260909'))
SEG = str(workspace_path('YOLO_SEG_7CLASS_10000_PER_CLASS_20260910_yolo_seg'))
RATIO = {"train": 0.80, "val": 0.15, "test": 0.05}
APPLY = "--apply" in sys.argv

files = sorted(os.listdir(SRC))
random.Random(0).shuffle(files)          # 분할에 장면 종류가 치우치지 않도록 섞는다
n = len(files)
n_tr = int(n * RATIO["train"]); n_va = int(n * RATIO["val"])
assign = ([("train", f) for f in files[:n_tr]] +
          [("val",   f) for f in files[n_tr:n_tr+n_va]] +
          [("test",  f) for f in files[n_tr+n_va:]])

c = collections.Counter(s for s, _ in assign)
print(f"배경 이미지 {n:,}장 분할: " +
      " / ".join(f"{s} {c[s]:,} ({c[s]/n*100:.1f}%)" for s in ("train","val","test")))

cur = {}
for name, root in (("bbox", BB), ("seg", SEG)):
    cur[name] = {s: len(os.listdir(f"{root}/images/{s}")) for s in ("train","val","test")}
    tot = sum(cur[name].values())
    after = {s: cur[name][s] + c[s] for s in c}
    ta = sum(after.values())
    print(f"\n[{name}] 현재 {tot:,}장 -> 추가 후 {ta:,}장 (배경 비중 {n/ta*100:.1f}%)")
    for s in ("train","val","test"):
        print(f"   {s}: {cur[name][s]:,} -> {after[s]:,}  ({after[s]/ta*100:.1f}%)")

if not APPLY:
    print("\n(dry-run) 실제로 추가하려면 --apply 를 붙이세요.")
    sys.exit()

with open(str(workspace_path('work/background_log.csv')), "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(["file", "split"])
    added = 0
    for s, f in assign:
        for root in (BB, SEG):
            ip = f"{root}/images/{s}/{f}"
            if not os.path.exists(ip):
                os.link(str(workspace_path('work/coco/bg_640', f'{f}')), ip)          # 하드링크 — 디스크를 추가로 쓰지 않는다
            lp = f"{root}/labels/{s}/{os.path.splitext(f)[0]}.txt"
            if not os.path.exists(lp):
                open(lp, "w").close()              # 빈 라벨 = 배경
        w.writerow([f, s]); added += 1
print(f"\n추가 완료: {added:,}장 x 2개 데이터셋 (로그: work/background_log.csv)")
