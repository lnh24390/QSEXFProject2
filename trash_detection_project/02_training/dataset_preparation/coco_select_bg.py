"""7개 쓰레기 클래스가 없는 COCO 실내/사람 장면을 배경 이미지 후보로 고른다."""
# 현재 작업 디렉토리와 상관없이 공용 작업공간 경로를 찾는다.
from pathlib import Path as _WorkspacePath
import sys as _workspace_sys
_workspace_root = next(p for p in _WorkspacePath(__file__).resolve().parents if (p / 'workspace_paths.py').is_file())
_workspace_sys.path.insert(0, str(_workspace_root))
from workspace_paths import workspace_path

import json, collections, csv, os

ANN = str(workspace_path('work/coco/annotations/instances_train2017.json'))

# 7개 클래스(paper, plastic, can, glass, battery, vinyl, general)와 혼동될 수 있는 COCO 클래스.
# 하나라도 있으면 그 이미지는 배경으로 쓸 수 없다.
BAN = {
    # 용기류 -> plastic / glass / can
    "bottle", "wine glass", "cup", "bowl", "vase",
    # 식기 -> plastic
    "fork", "knife", "spoon",
    # 음식물 -> general(음식물 쓰레기)
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake",
    # 종이류 -> paper
    "book",
    # 봉투·가방류 -> vinyl
    "handbag", "backpack", "suitcase", "umbrella",
    # 소형 전자기기 -> battery
    "cell phone", "remote", "hair drier", "toothbrush", "scissors",
    # 주방·욕실 — 라벨이 안 붙은 병·컵이 함께 찍혀 있을 위험이 큰 장소
    "sink", "refrigerator", "microwave", "oven", "toaster", "toilet",
    # 기타 혼동 위험
    "teddy bear", "potted plant", "tie",
}

# 사용자가 요청한 장면 요소. 최소 하나는 있어야 '일반적인 배경'이 된다.
WANT = {"person", "tv", "laptop", "chair", "couch", "bed",
        "dining table", "keyboard", "mouse", "clock", "bench"}

d = json.load(open(ANN, encoding="utf-8"))
cid2name = {c["id"]: c["name"] for c in d["categories"]}
imgs = {im["id"]: im for im in d["images"]}

per_img = collections.defaultdict(set)
area_of = collections.defaultdict(float)
for a in d["annotations"]:
    n = cid2name[a["category_id"]]
    per_img[a["image_id"]].add(n)
    if n in WANT:
        area_of[a["image_id"]] += a.get("area", 0.0)

rows = []
for iid, names in per_img.items():
    if names & BAN:            # 금지 클래스가 하나라도 있으면 탈락
        continue
    if not (names & WANT):     # 원하는 장면 요소가 없으면 탈락
        continue
    im = imgs[iid]
    w, h = im["width"], im["height"]
    if min(w, h) < 480:        # 640x640 으로 만들 때 과하게 확대되지 않도록
        continue
    frac = area_of[iid] / (w * h)
    if frac < 0.03:            # 원하는 요소가 너무 작으면 사실상 빈 사진
        continue
    rows.append({"id": iid, "file": im["file_name"], "w": w, "h": h,
                 "want": ",".join(sorted(names & WANT)),
                 "n_ann": len(names), "frac": round(frac, 4)})

# 장면이 다양하도록 '원하는 클래스 조합' 별로 고르게 섞는다
rows.sort(key=lambda r: (-r["frac"], r["id"]))
bucket = collections.defaultdict(list)
for r in rows:
    bucket[r["want"]].append(r)
mixed, i = [], 0
while len(mixed) < len(rows):
    added = False
    for k in sorted(bucket):
        if i < len(bucket[k]):
            mixed.append(bucket[k][i]); added = True
    if not added: break
    i += 1

os.makedirs("work", exist_ok=True)
with open(str(workspace_path('work/coco_bg_candidates.csv')), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["id","file","w","h","want","n_ann","frac"])
    w.writeheader(); w.writerows(mixed)

print(f"COCO train2017 전체 이미지: {len(imgs):,}")
print(f"배경 후보(금지 클래스 없음 + 원하는 요소 포함 + 크기·비중 조건): {len(mixed):,}")
print(f"\n{'장면 구성':<38}{'장수':>8}")
for k, v in sorted(collections.Counter(r["want"] for r in mixed).items(),
                   key=lambda x: -x[1])[:15]:
    print(f"{k:<38}{v:>8,}")
print(f"\n저장: work/coco_bg_candidates.csv")
