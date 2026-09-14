"""seg 데이터셋 분할을 조정 전 상태로 되돌린다.

    uv run python 02_training/dataset_preparation/seg_restore_split.py            # 되돌릴 내역만 표시
    uv run python 02_training/dataset_preparation/seg_restore_split.py --apply    # 실제 복원

파일 내용은 바뀌지 않았고 위치만 옮겼으므로, work/seg_split_before.csv 한 장이
완전한 복원 자료입니다.
"""
# 현재 작업 디렉토리와 상관없이 공용 작업공간 경로를 찾는다.
from pathlib import Path as _WorkspacePath
import sys as _workspace_sys
_workspace_root = next(p for p in _WorkspacePath(__file__).resolve().parents if (p / 'workspace_paths.py').is_file())
_workspace_sys.path.insert(0, str(_workspace_root))
from workspace_paths import workspace_path

import os, csv, sys, shutil, collections
SEG = str(workspace_path('YOLO_SEG_7CLASS_10000_PER_CLASS_20260910_yolo_seg'))
APPLY = "--apply" in sys.argv

want = {r["seg_stem"]: r["split_before"]
        for r in csv.DictReader(open(str(workspace_path('work/seg_split_before.csv')), encoding="utf-8"))}
now = {}
for s in ("train", "val", "test"):
    for f in os.listdir(str(workspace_path('YOLO_SEG_7CLASS_10000_PER_CLASS_20260910_yolo_seg', f'images/{s}'))):
        now[os.path.splitext(f)[0]] = s

todo = [(st, now[st], want[st]) for st in now if st in want and now[st] != want[st]]
print(f"복원 대상 {len(todo):,}쌍:",
      dict(collections.Counter(f"{a}->{b}" for _, a, b in todo)) or "없음")
missing = [st for st in now if st not in want]
if missing: print(f"인벤토리에 없는 파일 {len(missing)}장 — 확인 필요")

if not APPLY:
    print("(dry-run) 실제 복원하려면 --apply 를 붙이세요."); sys.exit()

n = 0
for st, src, dst in todo:
    shutil.move(str(workspace_path('YOLO_SEG_7CLASS_10000_PER_CLASS_20260910_yolo_seg', f'images/{src}/{st}.jpg')), str(workspace_path('YOLO_SEG_7CLASS_10000_PER_CLASS_20260910_yolo_seg', f'images/{dst}/{st}.jpg')))
    shutil.move(str(workspace_path('YOLO_SEG_7CLASS_10000_PER_CLASS_20260910_yolo_seg', f'labels/{src}/{st}.txt')), str(workspace_path('YOLO_SEG_7CLASS_10000_PER_CLASS_20260910_yolo_seg', f'labels/{dst}/{st}.txt')))
    n += 1
print(f"복원 완료: {n:,}쌍")
