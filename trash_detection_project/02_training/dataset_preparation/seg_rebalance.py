"""seg 데이터셋 분할을 80/15/5 로 조정한다. 그룹 단위 이동으로 누수를 막는다."""
# 현재 작업 디렉토리와 상관없이 공용 작업공간 경로를 찾는다.
from pathlib import Path as _WorkspacePath
import sys as _workspace_sys
_workspace_root = next(p for p in _WorkspacePath(__file__).resolve().parents if (p / 'workspace_paths.py').is_file())
_workspace_sys.path.insert(0, str(_workspace_root))
from workspace_paths import workspace_path

import os, re, csv, sys, gzip, json, heapq, shutil, collections

SEG = str(workspace_path('YOLO_SEG_7CLASS_10000_PER_CLASS_20260910_yolo_seg'))
MANIFEST = str(workspace_path('YOLO_7CLASS_10000_PER_CLASS_20260909/selection_manifest.jsonl.gz'))
NAMES = ["general","can","plastic","paper","glass","vinyl","battery"]   # seg data.yaml 순서
TARGET = {"val": 1500, "test": 500}
TOL, PEN = 60, 3.0
APPLY = "--apply" in sys.argv

# 1) seg 현재 상태와 라벨에서 클래스 읽기
cur_split = {}; img_cls = {}
for s in ("train","val","test"):
    for f in os.listdir(str(workspace_path('YOLO_SEG_7CLASS_10000_PER_CLASS_20260910_yolo_seg', f'labels/{s}'))):
        st = os.path.splitext(f)[0]
        cur_split[st] = s
        cs = set()
        for line in open(str(workspace_path('YOLO_SEG_7CLASS_10000_PER_CLASS_20260910_yolo_seg', f'labels/{s}/{f}'))):
            p = line.split()
            if p: cs.add(int(p[0]))
        img_cls[st] = cs

# 2) 누수 방지 그룹 — 파일명 접두어를 떼면 bbox 매니페스트의 stem 과 1:1 로 대응한다
stem2group = {}
for line in gzip.open(MANIFEST,'rt',encoding='utf-8'):
    r = json.loads(line)
    stem2group[os.path.splitext(os.path.basename(r["image"]))[0]] = r["split_group"]

groups = collections.defaultdict(list); nogroup = 0
for st in cur_split:
    m = re.match(r"^images__(?:train|val|test)__(.+)$", st)
    g = stem2group.get(m.group(1)) if m else None
    if g is None: nogroup += 1; g = f"__single__{st}"
    groups[g].append(st)

# 그룹이 여러 split 에 걸쳐 있는지 확인
span = [g for g, v in groups.items() if len({cur_split[x] for x in v}) > 1]

cur = {s: collections.Counter() for s in ("train","val","test")}
for st, s in cur_split.items():
    for c in img_cls[st]: cur[s][c] += 1

print(f"그룹 {len(groups):,}개, 그룹 미확인 이미지 {nogroup}장, split 을 걸친 그룹 {len(span)}개")
print(f"\n{'클래스':<9}{'train':>8}{'val':>8}{'test':>8}   조정 전 train%/val%/test%")
for i, nm in enumerate(NAMES):
    t = sum(cur[s][i] for s in cur)
    print(f"{nm:<9}{cur['train'][i]:>8}{cur['val'][i]:>8}{cur['test'][i]:>8}   "
          + " / ".join(f"{cur[s][i]/t*100:5.1f}" for s in ("train","val","test")))

# 3) 탐욕 선택 (train 전용 그룹만, train -> val/test 단방향)
gcls = {}
for g, sts in groups.items():
    if {cur_split[x] for x in sts} == {"train"}:
        cc = collections.Counter()
        for st in sts:
            for c in img_cls[st]: cc[c] += 1
        gcls[g] = cc

def select(dest, avail):
    need = {c: max(0, TARGET[dest] - cur[dest][c]) for c in range(7)}
    over = {c: 0 for c in range(7)}
    chosen = []
    def score(g):
        gain = pen = 0
        for c, n in gcls[g].items():
            take = min(n, need[c]); gain += take
            pen += max(0, (n - take) - max(0, TOL - over[c])) * PEN
        return gain - pen
    heap = []
    for g in avail:
        s = score(g)
        if s > 0: heapq.heappush(heap, (-s/len(groups[g]), g))
    while heap and any(v > 0 for v in need.values()):
        _, g = heapq.heappop(heap); s = score(g)
        if s <= 0: continue
        if heap and -s/len(groups[g]) > heap[0][0]:
            heapq.heappush(heap, (-s/len(groups[g]), g)); continue
        chosen.append(g)
        for c, n in gcls[g].items():
            take = min(n, need[c]); need[c] -= take; over[c] += n - take
            cur[dest][c] += n; cur["train"][c] -= n
        avail.discard(g)
    return chosen, need

avail = set(gcls)
tg, tneed = select("test", avail)
vg, vneed = select("val",  avail)
moves = [(st, "train", "test") for g in tg for st in groups[g]] + \
        [(st, "train", "val")  for g in vg for st in groups[g]]

n_tr = sum(1 for s in cur_split.values() if s=="train") - len(moves)
n_val = sum(1 for s in cur_split.values() if s=="val") + sum(1 for _,_,d in moves if d=="val")
n_te = sum(1 for s in cur_split.values() if s=="test") + sum(1 for _,_,d in moves if d=="test")
tot = len(cur_split)
print(f"\n이동: 그룹 test {len(tg)}개 + val {len(vg)}개 = {len(moves):,}쌍")
print(f"예상 분할: train {n_tr:,} ({n_tr/tot*100:.1f}%) / val {n_val:,} ({n_val/tot*100:.1f}%) / test {n_te:,} ({n_te/tot*100:.1f}%)")
print(f"\n{'클래스':<9}{'train':>8}{'val':>8}{'test':>8}   조정 후 train%/val%/test%")
for i, nm in enumerate(NAMES):
    t = sum(cur[s][i] for s in cur)
    print(f"{nm:<9}{cur['train'][i]:>8}{cur['val'][i]:>8}{cur['test'][i]:>8}   "
          + " / ".join(f"{cur[s][i]/t*100:5.1f}" for s in ("train","val","test")))
print("미달 잔여 val:", {NAMES[c]:v for c,v in vneed.items() if v>0} or "없음",
      " test:", {NAMES[c]:v for c,v in tneed.items() if v>0} or "없음")

# 되돌리기 인벤토리 (파일 내용은 바뀌지 않으므로 이 표가 완전한 복원 자료다)
with open(str(workspace_path('work/seg_split_before.csv')),"w",newline="",encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(["seg_stem","split_before"])
    w.writerows(sorted(cur_split.items()))
print(f"\n되돌리기 인벤토리: work/seg_split_before.csv ({len(cur_split):,}행)")

if not APPLY:
    print("\n(dry-run) 실제 이동하려면 --apply 를 붙이세요.")
    sys.exit()

with open(str(workspace_path('work/seg_split_move_log.csv')),"w",newline="",encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(["seg_stem","from","to"]); n = 0
    for st, src, dst in moves:
        ip, lp = str(workspace_path('YOLO_SEG_7CLASS_10000_PER_CLASS_20260910_yolo_seg', f'images/{src}/{st}.jpg')), str(workspace_path('YOLO_SEG_7CLASS_10000_PER_CLASS_20260910_yolo_seg', f'labels/{src}/{st}.txt'))
        if not (os.path.exists(ip) and os.path.exists(lp)):
            print("건너뜀:", st); continue
        shutil.move(ip, str(workspace_path('YOLO_SEG_7CLASS_10000_PER_CLASS_20260910_yolo_seg', f'images/{dst}/{st}.jpg')))
        shutil.move(lp, str(workspace_path('YOLO_SEG_7CLASS_10000_PER_CLASS_20260910_yolo_seg', f'labels/{dst}/{st}.txt')))
        w.writerow([st, src, dst]); n += 1
print(f"\n이동 완료: {n:,}쌍 (로그: work/seg_split_move_log.csv)")
