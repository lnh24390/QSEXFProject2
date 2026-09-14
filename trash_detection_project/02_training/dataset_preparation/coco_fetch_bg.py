"""후보 목록에서 N장을 내려받아 640x640 으로 변환한다 (중앙 정사각 크롭)."""
# 현재 작업 디렉토리와 상관없이 공용 작업공간 경로를 찾는다.
from pathlib import Path as _WorkspacePath
import sys as _workspace_sys
_workspace_root = next(p for p in _WorkspacePath(__file__).resolve().parents if (p / 'workspace_paths.py').is_file())
_workspace_sys.path.insert(0, str(_workspace_root))
from workspace_paths import workspace_path

import csv, os, io, sys, concurrent.futures as cf, urllib.request
from PIL import Image

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
OUT = str(workspace_path('work/coco/bg_640'))
os.makedirs(OUT, exist_ok=True)
rows = list(csv.DictReader(open(str(workspace_path('work/coco_bg_candidates.csv')), encoding="utf-8")))[:N]

def fetch(r):
    dst = os.path.join(OUT, f"bg_coco_{int(r['id']):012d}.jpg")
    if os.path.exists(dst):
        return dst
    url = f"http://images.cocodataset.org/train2017/{r['file']}"
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            im = Image.open(io.BytesIO(resp.read())).convert("RGB")
        w, h = im.size
        s = min(w, h)                                   # 중앙 정사각 크롭 후 640 으로 축소.
        im = im.crop(((w-s)//2, (h-s)//2, (w-s)//2+s, (h-s)//2+s))  # 여백을 넣지 않아
        im = im.resize((640, 640), Image.LANCZOS)       # 인위적인 회색 띠가 생기지 않는다.
        im.save(dst, "JPEG", quality=90)
        return dst
    except Exception as e:
        return f"FAIL {r['id']} {type(e).__name__}"

ok = fail = 0
with cf.ThreadPoolExecutor(max_workers=16) as ex:
    for i, res in enumerate(ex.map(fetch, rows), 1):
        if str(res).startswith("FAIL"): fail += 1
        else: ok += 1
        if i % 500 == 0: print(f"  {i}/{len(rows)}  성공 {ok} 실패 {fail}", flush=True)
print(f"완료: 성공 {ok}, 실패 {fail} -> {OUT}")
