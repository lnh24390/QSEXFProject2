# 현재 작업 디렉토리와 상관없이 공용 작업공간 경로를 찾는다.
from pathlib import Path as _WorkspacePath
import sys as _workspace_sys
_workspace_root = next(p for p in _WorkspacePath(__file__).resolve().parents if (p / 'workspace_paths.py').is_file())
_workspace_sys.path.insert(0, str(_workspace_root))
from workspace_paths import workspace_path

import gzip, json, collections, heapq, csv, os
ROOT = str(workspace_path('YOLO_7CLASS_10000_PER_CLASS_20260909'))
NAMES = ["paper","plastic","can","glass","battery","vinyl","general"]
TARGET = {"val": 1500, "test": 500}
TOL = 60          # 목표 초과 허용치
PEN = 3.0         # 초과 1장당 벌점

recs = []
for line in gzip.open(os.path.join(ROOT,"selection_manifest.jsonl.gz"),'rt',encoding='utf-8'):
    recs.append(json.loads(line))

groups = collections.defaultdict(list)
for r in recs:
    groups[r['split_group']].append(r)

# 현재 split 별 클래스 이미지 수
cur = {s: collections.Counter() for s in ("train","val","test")}
for r in recs:
    for c in r['objects']:
        cur[r['split']][int(c)] += 1

# train 전용 그룹의 클래스별 이미지 수
gcls, gsplit = {}, {}
for g, rs in groups.items():
    sp = {r['split'] for r in rs}
    gsplit[g] = sp
    if sp == {"train"}:
        cc = collections.Counter()
        for r in rs:
            for c in r['objects']:
                cc[int(c)] += 1
        gcls[g] = cc

def select(dest, avail):
    """dest split 의 클래스 목표를 채우는 그룹을 탐욕적으로 고른다."""
    need = {c: max(0, TARGET[dest] - cur[dest][c]) for c in range(7)}
    over = {c: 0 for c in range(7)}
    chosen = []
    def score(g):
        cc = gcls[g]; gain = pen = 0
        for c, n in cc.items():
            take = min(n, need[c]); gain += take
            ex = n - take
            room = max(0, TOL - over[c])
            pen += max(0, ex - room) * PEN
        return gain - pen
    heap = []
    for g in avail:
        s = score(g)
        if s > 0:
            heapq.heappush(heap, (-s / len(groups[g]), -s, g))
    while heap and any(v > 0 for v in need.values()):
        _, _, g = heapq.heappop(heap)
        s = score(g)
        if s <= 0:
            continue
        # lazy 재평가: 현재 점수로 다시 넣어 최상단인지 확인
        if heap and -s/len(groups[g]) > heap[0][0]:
            heapq.heappush(heap, (-s/len(groups[g]), -s, g))
            continue
        chosen.append(g)
        for c, n in gcls[g].items():
            take = min(n, need[c]); need[c] -= take
            over[c] += n - take
            cur[dest][c] += n
            cur["train"][c] -= n
        avail.discard(g)
    return chosen, need

avail = set(gcls)
test_groups, test_need = select("test", avail)
val_groups,  val_need  = select("val",  avail)

with open(str(workspace_path('work/move_plan.csv')),"w",newline="",encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["split_group","dest","images"])
    for g in test_groups: w.writerow([g,"test",len(groups[g])])
    for g in val_groups:  w.writerow([g,"val", len(groups[g])])

mv_test = sum(len(groups[g]) for g in test_groups)
mv_val  = sum(len(groups[g]) for g in val_groups)
print(f"이동 그룹: test {len(test_groups)}개 / {mv_test}장,  val {len(val_groups)}개 / {mv_val}장")
tot = len(recs)
tr = tot - (cur['val'].total() and 0) # placeholder
n_tr = sum(1 for r in recs if r['split']=='train') - mv_test - mv_val
n_val = sum(1 for r in recs if r['split']=='val') + mv_val
n_test = sum(1 for r in recs if r['split']=='test') + mv_test
print(f"\n예상 분할: train {n_tr} ({n_tr/tot*100:.1f}%) / val {n_val} ({n_val/tot*100:.1f}%) / test {n_test} ({n_test/tot*100:.1f}%)")
print(f"\n{'클래스':<9}{'train':>8}{'val':>8}{'test':>8}   train%/val%/test%")
for i,nm in enumerate(NAMES):
    t = cur['train'][i]+cur['val'][i]+cur['test'][i]
    print(f"{nm:<9}{cur['train'][i]:>8}{cur['val'][i]:>8}{cur['test'][i]:>8}   "
          + " / ".join(f"{cur[s][i]/t*100:5.1f}" for s in ('train','val','test')))
print("\n미달 잔여 (val):", {NAMES[c]:v for c,v in val_need.items() if v>0} or "없음")
print("미달 잔여 (test):", {NAMES[c]:v for c,v in test_need.items() if v>0} or "없음")
