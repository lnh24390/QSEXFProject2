# 현재 작업 디렉토리와 상관없이 공용 작업공간 경로를 찾는다.
from pathlib import Path as _WorkspacePath
import sys as _workspace_sys
_workspace_root = next(p for p in _WorkspacePath(__file__).resolve().parents if (p / 'workspace_paths.py').is_file())
_workspace_sys.path.insert(0, str(_workspace_root))
from workspace_paths import workspace_path

import os, collections, csv, gzip, json
ROOT = str(workspace_path('YOLO_7CLASS_10000_PER_CLASS_20260909'))
NAMES=["paper","plastic","can","glass","battery","vinyl","general"]
res={}; ok=True
for s in ("train","val","test"):
    imgs={os.path.splitext(f)[0] for f in os.listdir(str(workspace_path('YOLO_7CLASS_10000_PER_CLASS_20260909', f'images/{s}')))}
    lbls={os.path.splitext(f)[0] for f in os.listdir(str(workspace_path('YOLO_7CLASS_10000_PER_CLASS_20260909', f'labels/{s}')))}
    assert imgs==lbls, f"{s}: image/label mismatch {len(imgs^lbls)}"
    ic=collections.Counter(); bc=collections.Counter()
    for st in lbls:
        cs=set()
        for line in open(str(workspace_path('YOLO_7CLASS_10000_PER_CLASS_20260909', f'labels/{s}/{st}.txt'))):
            if line.strip():
                c=int(line.split()[0]); cs.add(c); bc[c]+=1
        for c in cs: ic[c]+=1
    res[s]={"n":len(imgs),"ic":ic,"bc":bc}
tot=sum(res[s]["n"] for s in res)

# 새 split 매핑으로 그룹 누수 재검사
where={}
for s in ("train","val","test"):
    for f in os.listdir(str(workspace_path('YOLO_7CLASS_10000_PER_CLASS_20260909', f'images/{s}'))): where[os.path.splitext(f)[0]]=s
g2s=collections.defaultdict(set)
for line in gzip.open(str(workspace_path('YOLO_7CLASS_10000_PER_CLASS_20260909', f'selection_manifest.jsonl.gz')),'rt',encoding='utf-8'):
    r=json.loads(line); st=os.path.splitext(os.path.basename(r["image"]))[0]
    g2s[r["split_group"]].add(where[st])
span=[g for g,v in g2s.items() if len(v)>1]

L=[]
L.append("# 분할 조정 결과 검증\n\n작성일: 2026-09-10\n")
L.append("## 1. 전체 분할\n")
L.append("| 분할 | 조정 전 | 조정 후 | 비율 |\n|---|---:|---:|---:|")
before={"train":50647,"val":7187,"test":1402}
for s in ("train","val","test"):
    L.append(f"| {s} | {before[s]:,} | {res[s]['n']:,} | {res[s]['n']/tot*100:.1f}% |")
L.append(f"| 합계 | {sum(before.values()):,} | {tot:,} | 100% |\n")
L.append("## 2. 클래스별 이미지 분포 (조정 후)\n")
L.append("| 클래스 | train | val | test | train%/val%/test% |\n|---|---:|---:|---:|---|")
for i,nm in enumerate(NAMES):
    t=sum(res[s]["ic"][i] for s in res)
    L.append(f"| {nm} | {res['train']['ic'][i]:,} | {res['val']['ic'][i]:,} | {res['test']['ic'][i]:,} | "
             + " / ".join(f"{res[s]['ic'][i]/t*100:.1f}" for s in ('train','val','test'))+" |")
L.append("\n## 3. 클래스별 박스 수 (조정 후)\n")
L.append("| 클래스 | train | val | test | 합계 |\n|---|---:|---:|---:|---:|")
for i,nm in enumerate(NAMES):
    t=sum(res[s]["bc"][i] for s in res)
    L.append(f"| {nm} | {res['train']['bc'][i]:,} | {res['val']['bc'][i]:,} | {res['test']['bc'][i]:,} | {t:,} |")
L.append("\n## 4. 무결성 점검\n")
L.append("| 항목 | 결과 |\n|---|---|")
L.append("| 이미지·라벨 쌍 일치 (3개 분할) | 통과 |")
L.append(f"| 총 이미지 수 보존 (59,236) | {'통과' if tot==59236 else '실패'} |")
L.append(f"| split 을 걸친 split_group | {len(span)}개 {'(통과)' if not span else '(실패)'} |")
nmoved=sum(1 for _ in csv.DictReader(open(str(workspace_path('work/split_move_log.csv')),encoding='utf-8')))
L.append(f"| 이동한 이미지·라벨 쌍 | {nmoved:,}쌍 (train → val/test 단방향) |")
L.append(f"| 되돌리기 로그 | `work/split_move_log.csv` |")
open(str(workspace_path('work/split_after.md')),"w",encoding="utf-8").write("\n".join(L))
print("\n".join(L))
