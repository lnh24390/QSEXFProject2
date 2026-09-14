# 현재 작업 디렉토리와 상관없이 공용 작업공간 경로를 찾는다.
from pathlib import Path as _WorkspacePath
import sys as _workspace_sys
_workspace_root = next(p for p in _WorkspacePath(__file__).resolve().parents if (p / 'workspace_paths.py').is_file())
_workspace_sys.path.insert(0, str(_workspace_root))
from workspace_paths import workspace_path

import gzip, json, collections, csv, os, shutil, sys
ROOT = str(workspace_path('YOLO_7CLASS_10000_PER_CLASS_20260909'))
dest_of = {}
for row in csv.DictReader(open(str(workspace_path('work/move_plan.csv')), encoding="utf-8")):
    dest_of[row["split_group"]] = row["dest"]

groups = collections.defaultdict(list)
for line in gzip.open(os.path.join(ROOT,"selection_manifest.jsonl.gz"),'rt',encoding='utf-8'):
    r = json.loads(line)
    if r["split_group"] in dest_of:
        groups[r["split_group"]].append(r)

log = open(str(workspace_path('work/split_move_log.csv')),"w",newline="",encoding="utf-8")
w = csv.writer(log); w.writerow(["split_group","stem","from","to"])
moved = 0; missing = []
for g, rs in groups.items():
    dst = dest_of[g]
    for r in rs:
        stem = os.path.splitext(os.path.basename(r["image"]))[0]
        src_split = r["split"]
        if src_split == dst:
            continue
        ip = os.path.join(ROOT,"images",src_split,stem+".jpg")
        lp = os.path.join(ROOT,"labels",src_split,stem+".txt")
        if not (os.path.exists(ip) and os.path.exists(lp)):
            missing.append(stem); continue
        shutil.move(ip, os.path.join(ROOT,"images",dst,stem+".jpg"))
        shutil.move(lp, os.path.join(ROOT,"labels",dst,stem+".txt"))
        w.writerow([g, stem, src_split, dst]); moved += 1
log.close()
print("moved files (image+label pairs):", moved)
print("missing:", len(missing), missing[:5])
