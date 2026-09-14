# 현재 작업 디렉토리와 상관없이 공용 작업공간 경로를 찾는다.
from pathlib import Path as _WorkspacePath
import sys as _workspace_sys
_workspace_root = next(p for p in _WorkspacePath(__file__).resolve().parents if (p / 'workspace_paths.py').is_file())
_workspace_sys.path.insert(0, str(_workspace_root))
from workspace_paths import workspace_path

import os, re, csv, gzip, json, collections
SEG = str(workspace_path('YOLO_SEG_7CLASS_10000_PER_CLASS_20260910_yolo_seg'))
MAN = str(workspace_path('YOLO_7CLASS_10000_PER_CLASS_20260909/selection_manifest.jsonl.gz'))
NAMES = ["general","can","plastic","paper","glass","vinyl","battery"]
res = {}
for s in ("train","val","test"):
    imgs = {os.path.splitext(f)[0] for f in os.listdir(str(workspace_path('YOLO_SEG_7CLASS_10000_PER_CLASS_20260910_yolo_seg', f'images/{s}')))}
    lbls = {os.path.splitext(f)[0] for f in os.listdir(str(workspace_path('YOLO_SEG_7CLASS_10000_PER_CLASS_20260910_yolo_seg', f'labels/{s}')))}
    assert imgs == lbls, f"{s}: 쌍 불일치 {len(imgs ^ lbls)}"
    ic = collections.Counter(); pc = collections.Counter(); bad = 0
    for st in lbls:
        cs = set()
        for line in open(str(workspace_path('YOLO_SEG_7CLASS_10000_PER_CLASS_20260910_yolo_seg', f'labels/{s}/{st}.txt'))):
            p = line.split()
            if not p: continue
            c = int(p[0]); cs.add(c); pc[c] += 1
            if not (0 <= c <= 6) or (len(p)-1) % 2 or (len(p)-1) < 6: bad += 1
        for c in cs: ic[c] += 1
    res[s] = {"n": len(imgs), "ic": ic, "pc": pc, "bad": bad}
tot = sum(res[s]["n"] for s in res)

where = {}
for s in ("train","val","test"):
    for f in os.listdir(str(workspace_path('YOLO_SEG_7CLASS_10000_PER_CLASS_20260910_yolo_seg', f'images/{s}'))): where[os.path.splitext(f)[0]] = s
s2g = {}
for line in gzip.open(MAN,'rt',encoding='utf-8'):
    r = json.loads(line); s2g[os.path.splitext(os.path.basename(r["image"]))[0]] = r["split_group"]
g2s = collections.defaultdict(set)
for st, s in where.items():
    m = re.match(r"^images__(?:train|val|test)__(.+)$", st)
    g2s[s2g[m.group(1)]].add(s)
span = [g for g, v in g2s.items() if len(v) > 1]

L = ["# seg 데이터셋 분할 조정 결과\n\n작성일: 2026-09-10 · 대상: `YOLO_SEG_7CLASS_10000_PER_CLASS_20260910_yolo_seg`\n"]
L.append("## 1. 전체 분할\n")
L.append("| 분할 | 조정 전 | 조정 후 | 비율 |\n|---|---:|---:|---:|")
before = {"train":50647,"val":7187,"test":1402}
for s in ("train","val","test"):
    L.append(f"| {s} | {before[s]:,} | {res[s]['n']:,} | {res[s]['n']/tot*100:.1f}% |")
L.append(f"| 합계 | {sum(before.values()):,} | {tot:,} | 100% |\n")
L.append("## 2. 클래스별 이미지 분포\n")
L.append("| ID | 클래스 | train | val | test | 조정 후 % | 조정 전 % |\n|---:|---|---:|---:|---:|---|---|")
bef = {"general":(92.3,7.4,0.3),"can":(80.2,14.8,5.0),"plastic":(80.1,14.9,5.0),"paper":(80.1,14.9,5.0),
       "glass":(82.8,14.8,2.4),"vinyl":(80.4,14.7,4.9),"battery":(96.6,3.0,0.4)}
for i, nm in enumerate(NAMES):
    t = sum(res[s]["ic"][i] for s in res)
    now = " / ".join(f"{res[s]['ic'][i]/t*100:.1f}" for s in ("train","val","test"))
    L.append(f"| {i} | {nm} | {res['train']['ic'][i]:,} | {res['val']['ic'][i]:,} | {res['test']['ic'][i]:,} | {now} | " + " / ".join(f"{v:.1f}" for v in bef[nm]) + " |")
L.append("\n## 3. 클래스별 폴리곤 수\n")
L.append("| 클래스 | train | val | test | 합계 |\n|---|---:|---:|---:|---:|")
for i, nm in enumerate(NAMES):
    t = sum(res[s]["pc"][i] for s in res)
    L.append(f"| {nm} | {res['train']['pc'][i]:,} | {res['val']['pc'][i]:,} | {res['test']['pc'][i]:,} | {t:,} |")
L.append("\n## 4. 무결성 점검\n")
L.append("| 항목 | 결과 |\n|---|---|")
L.append("| 이미지·라벨 쌍 일치 (3개 분할) | 통과 |")
L.append(f"| 총 이미지 수 보존 (59,236) | {'통과' if tot==59236 else '실패'} |")
L.append(f"| split 을 걸친 split_group | {len(span)}개 {'(통과)' if not span else '(실패)'} |")
tb = sum(res[s]["bad"] for s in res)
L.append(f"| 라벨 형식 오류 (클래스 범위·좌표 짝) | {tb}개 {'(통과)' if tb==0 else '(실패)'} |")
nmv = sum(1 for _ in csv.DictReader(open(str(workspace_path('work/seg_split_move_log.csv')),encoding='utf-8')))
L.append(f"| 이동한 이미지·라벨 쌍 | {nmv:,}쌍 (train → val/test 단방향) |")
L.append("| 되돌리기 인벤토리 | `work/seg_split_before.csv` (59,236행) |")
L.append("| 이동 로그 | `work/seg_split_move_log.csv` |")
L.append("| 복원 스크립트 | `uv run python 02_training/dataset_preparation/seg_restore_split.py --apply` |")
L.append("\n## 5. data.yaml 수정\n")
L.append("| 항목 | 조정 전 | 조정 후 |\n|---|---|---|")
L.append("| `path` | `C:/Users/USER/Desktop/DataSegmentLabeling/datasets/YOLO_7CLASS_10000_PER_CLASS_20260909_yolo` (**존재하지 않는 폴더**) | 실제 데이터셋 경로 |")
L.append("| `nc` | 없음 | 7 |")
L.append("\n조정 전 원본은 `work/seg_data.yaml.bak` 에 보관했습니다.\n")
L.append("## 6. 주의사항\n")
L.append("- **클래스 ID 순서가 bbox 데이터셋과 다릅니다.** seg 는 `0=general, 1=can, 2=plastic, 3=paper, 4=glass, 5=vinyl, 6=battery`, bbox 는 `0=paper, 1=plastic, 2=can, 3=glass, 4=battery, 5=vinyl, 6=general` 입니다. vinyl(5) 만 같습니다. 두 데이터셋의 라벨·가중치·평가 결과를 섞지 마십시오.")
L.append("- **파일명의 `images__train__` 접두어는 원래 분할을 뜻합니다.** 이동한 3,174장은 이제 접두어와 실제 분할이 다릅니다. 학습에는 폴더 구조만 쓰이므로 영향은 없지만, 파일명으로 분할을 판단하면 안 됩니다.")
L.append("- 폴리곤 인스턴스의 **7.4% 가 좌표 4점(사각형)** 입니다. 실제 외곽선이 아니라 박스에서 변환된 것으로 보이며, 세그멘테이션 성능에 영향을 줄 수 있습니다.")
L.append("- 조정 전 검증 결과와는 직접 비교할 수 없습니다. val 에 general·battery 가 대폭 추가되어 난이도가 달라졌습니다.")
open(str(workspace_path('work/seg_split_after.md')),"w",encoding="utf-8").write("\n".join(L))
print("\n".join(L))
