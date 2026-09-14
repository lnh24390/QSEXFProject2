"""seg 데이터셋의 분할을 조정된 bbox 데이터셋과 정확히 일치시킨다 (dry-run 우선)."""
# 현재 작업 디렉토리와 상관없이 공용 작업공간 경로를 찾는다.
from pathlib import Path as _WorkspacePath
import sys as _workspace_sys
_workspace_root = next(p for p in _WorkspacePath(__file__).resolve().parents if (p / 'workspace_paths.py').is_file())
_workspace_sys.path.insert(0, str(_workspace_root))
from workspace_paths import workspace_path

import os, re, csv, sys, collections, shutil

SEG = str(workspace_path('YOLO_SEG_7CLASS_10000_PER_CLASS_20260910_yolo_seg'))
BB  = str(workspace_path('YOLO_7CLASS_10000_PER_CLASS_20260909'))
APPLY = "--apply" in sys.argv

# 1) 목표: 조정된 bbox 데이터셋의 stem -> split
target = {}
for s in ("train", "val", "test"):
    for f in os.listdir(str(workspace_path('YOLO_7CLASS_10000_PER_CLASS_20260909', f'images/{s}'))):
        target[os.path.splitext(f)[0]] = s

# 2) 현재 seg 상태를 기록 (되돌리기용 전체 인벤토리)
cur = []   # (파일명, 현재 split, bbox stem)
for s in ("train", "val", "test"):
    for f in os.listdir(str(workspace_path('YOLO_SEG_7CLASS_10000_PER_CLASS_20260910_yolo_seg', f'images/{s}'))):
        st = os.path.splitext(f)[0]
        m = re.match(r"^images__(?:train|val|test)__(.+)$", st)
        cur.append((st, s, m.group(1) if m else None))

if not APPLY:
    os.makedirs("work", exist_ok=True)
    with open(str(workspace_path('work/seg_split_before.csv')), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["seg_stem", "split_before", "bbox_stem"])
        w.writerows(cur)
    print(f"되돌리기 인벤토리 저장: work/seg_split_before.csv ({len(cur):,}행)")

# 3) 이동 계획
moves = []; missing = 0
for st, s, bstem in cur:
    if bstem is None or bstem not in target:
        missing += 1; continue
    if target[bstem] != s:
        moves.append((st, s, target[bstem]))

print(f"\nbbox 에서 찾을 수 없는 seg 파일: {missing}")
print(f"이동 대상: {len(moves):,}쌍 (이미지+라벨)")
c = collections.Counter((a, b) for _, a, b in moves)
for (a, b), n in sorted(c.items()):
    print(f"  {a} -> {b}: {n:,}")

after = collections.Counter()
for st, s, bstem in cur:
    after[target.get(bstem, s)] += 1
tot = sum(after.values())
print(f"\n예상 분할: " + " / ".join(f"{s} {after[s]:,} ({after[s]/tot*100:.1f}%)" for s in ("train","val","test")))

if not APPLY:
    print("\n(dry-run) 실제 이동하려면 --apply 를 붙여 실행하세요.")
    sys.exit()

# 4) 실제 이동
with open(str(workspace_path('work/seg_split_move_log.csv')), "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(["seg_stem", "from", "to"])
    n = 0
    for st, src, dst in moves:
        ip, lp = str(workspace_path('YOLO_SEG_7CLASS_10000_PER_CLASS_20260910_yolo_seg', f'images/{src}/{st}.jpg')), str(workspace_path('YOLO_SEG_7CLASS_10000_PER_CLASS_20260910_yolo_seg', f'labels/{src}/{st}.txt'))
        if not (os.path.exists(ip) and os.path.exists(lp)):
            print("건너뜀(파일 없음):", st); continue
        shutil.move(ip, str(workspace_path('YOLO_SEG_7CLASS_10000_PER_CLASS_20260910_yolo_seg', f'images/{dst}/{st}.jpg')))
        shutil.move(lp, str(workspace_path('YOLO_SEG_7CLASS_10000_PER_CLASS_20260910_yolo_seg', f'labels/{dst}/{st}.txt')))
        w.writerow([st, src, dst]); n += 1
print(f"\n이동 완료: {n:,}쌍  (로그: work/seg_split_move_log.csv)")
