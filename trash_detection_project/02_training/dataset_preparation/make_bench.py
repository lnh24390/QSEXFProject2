# 현재 작업 디렉토리와 상관없이 공용 작업공간 경로를 찾는다.
from pathlib import Path as _WorkspacePath
import sys as _workspace_sys
_workspace_root = next(p for p in _WorkspacePath(__file__).resolve().parents if (p / 'workspace_paths.py').is_file())
_workspace_sys.path.insert(0, str(_workspace_root))
from workspace_paths import workspace_path

import gzip, json, collections, os, heapq, yaml
ROOT = str(workspace_path('YOLO_7CLASS_10000_PER_CLASS_20260909'))
OUT  = str(workspace_path('work/bench_1000'))
NAMES=["paper","plastic","can","glass","battery","vinyl","general"]
TARGET={"train":800,"val":150,"test":50}

where={}
for s in ("train","val","test"):
    for f in os.listdir(str(workspace_path('YOLO_7CLASS_10000_PER_CLASS_20260909', f'images/{s}'))): where[os.path.splitext(f)[0]]=s
recs=[]
for line in gzip.open(str(workspace_path('YOLO_7CLASS_10000_PER_CLASS_20260909', f'selection_manifest.jsonl.gz')),'rt',encoding='utf-8'):
    r=json.loads(line); r["stem"]=os.path.splitext(os.path.basename(r["image"]))[0]
    r["split"]=where[r["stem"]]; recs.append(r)
bysplit=collections.defaultdict(lambda: collections.defaultdict(list))
for r in recs: bysplit[r["split"]][r["split_group"]].append(r)

picked=collections.defaultdict(list)
for s in ("train","val","test"):
    need={c:TARGET[s] for c in range(7)}
    gcls={g:collections.Counter(int(c) for r in rs for c in r["objects"]) for g,rs in bysplit[s].items()}
    def score(g):
        gain=pen=0
        for c,n in gcls[g].items():
            t=min(n,need[c]); gain+=t; pen+=(n-t)*1.5
        return gain-pen
    heap=[]
    for g in gcls:
        sc=score(g)
        if sc>0: heapq.heappush(heap,(-sc/len(bysplit[s][g]),g))
    while heap and any(v>0 for v in need.values()):
        _,g=heapq.heappop(heap); sc=score(g)
        if sc<=0: continue
        if heap and -sc/len(bysplit[s][g])>heap[0][0]:
            heapq.heappush(heap,(-sc/len(bysplit[s][g]),g)); continue
        picked[s].append(g)
        for c,n in gcls[g].items(): need[c]=max(0,need[c]-n)

for s in ("train","val","test"):
    os.makedirs(str(workspace_path('work/bench_1000', f'images/{s}')),exist_ok=True); os.makedirs(str(workspace_path('work/bench_1000', f'labels/{s}')),exist_ok=True)
stats={s:collections.Counter() for s in ("train","val","test")}; counts={}
for s in ("train","val","test"):
    n=0
    for g in picked[s]:
        for r in bysplit[s][g]:
            st=r["stem"]
            for sub,ext in (("images",".jpg"),("labels",".txt")):
                dst=str(workspace_path('work/bench_1000', f'{sub}/{s}/{st}{ext}'))
                if not os.path.exists(dst): os.link(str(workspace_path('YOLO_7CLASS_10000_PER_CLASS_20260909', f'{sub}/{s}/{st}{ext}')),dst)
            for c in r["objects"]: stats[s][int(c)]+=1
            n+=1
    counts[s]=n
yaml.safe_dump({"path":OUT.replace("\\","/"),"train":"images/train","val":"images/val","test":"images/test",
                "nc":7,"names":{i:n for i,n in enumerate(NAMES)}},
               open(str(workspace_path('work/bench_1000', f'data.yaml')),"w",encoding="utf-8"),allow_unicode=True,sort_keys=False)
print("bench_1000 이미지:", counts, "합계", sum(counts.values()))
print(f"{'클래스':<9}{'train':>7}{'val':>7}{'test':>7}{'합계':>7}")
for i,nm in enumerate(NAMES):
    print(f"{nm:<9}{stats['train'][i]:>7}{stats['val'][i]:>7}{stats['test'][i]:>7}{sum(stats[s][i] for s in stats):>7}")
