"""Load each installed SAM backend from weights/ and segment a known shape.

Skips anything whose library or checkpoint is absent -- an absent backend is
not a failure (DEVELOPMENT.md 2.2), a *broken* one is.

    uv run --no-project --python .venv/Scripts/python.exe tests/smoke_backends.py
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dsl.core.deps import BY_MODULE
from dsl.core.settings import Settings
from dsl.sam.backends import create_backend
from dsl.sam.catalog import CATALOG

# A bright rectangle on a dark ground; every backend should find its edges.
BOX = (220, 150, 420, 330)          # x0, y0, x1, y1
IMG = np.full((480, 640, 3), 30, np.uint8)
IMG[BOX[1]:BOX[3], BOX[0]:BOX[2]] = (230, 180, 60)
CLICK = [((BOX[0] + BOX[2]) / 2.0, (BOX[1] + BOX[3]) / 2.0, 1)]

# One representative per backend class, cheapest checkpoint of each.
CANDIDATES = ["sam2.1_tiny_onnx", "sam1_vit_b", "sam2.1_hiera_small",
              "sam2.1_t_ultra", "mobile_sam"]

# SAM 3 is 3.5GB and takes minutes to load, so it stays out of the default
# run. `--heavy` adds it; it is the only backend with text prompting.
HEAVY = ["sam3"]
if "--heavy" in sys.argv:
    CANDIDATES += HEAVY

#: Max edge error tolerated, per backend. SAM 3 reasons about concepts rather
#: than edges, so on a flat synthetic rectangle its click/box masks land a
#: couple of dozen pixels out while its *text* mask is exact.
TOLERANCE = {"sam3": 30}
DEFAULT_TOLERANCE = 12

BACKEND_LIB = {"sam1": "segment_anything", "sam2": "sam2",
               "ultralytics": "ultralytics", "sam3": "transformers",
               "onnx": "onnxruntime"}

st = Settings.load()
W = st.weights_path
print(f"weights: {W}\n")

failures, ran, skipped = [], 0, 0

for key in CANDIDATES:
    spec = CATALOG.get(key)
    if spec is None:
        continue
    lib = BACKEND_LIB.get(spec.backend, "")
    dep = BY_MODULE.get(lib)
    if lib and dep and not dep.installed:
        print(f"-- {key}: skip ({lib} 미설치 -> {dep.install})")
        skipped += 1
        continue
    if not spec.is_installed(W):
        print(f"-- {key}: skip (체크포인트 없음: {spec.local_path(W)})")
        skipped += 1
        continue

    # torch backends prefer the GPU; the ONNX one has no torch at all.
    device = "cpu"
    if spec.backend != "onnx":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    precision = "fp16" if device.startswith("cuda") else "fp32"
    if precision not in spec.quant_modes:
        precision = spec.quant_modes[0]

    print(f"-- {key}  ({spec.backend}, {device}, {precision})")
    try:
        t0 = time.time()
        be = create_backend(key, W, device=device, precision=precision)
        be.load()
        t_load = time.time() - t0

        t0 = time.time()
        be.set_image(IMG, key="synthetic")
        t_enc = time.time() - t0

        t0 = time.time()
        res = be.predict(points=CLICK, multimask=True)
        t_pred = time.time() - t0

        m = res.best() if callable(res.best) else res.best
        if m is None or not (m > 0).any():
            raise AssertionError("빈 마스크")
        ys, xs = np.where(m > 0)
        got = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        # Every backend resizes internally, so edges wobble a little.
        tol = TOLERANCE.get(key, DEFAULT_TOLERANCE)
        off = max(abs(a - b) for a, b in zip(got, BOX))
        ok = off <= tol
        print(f"   load {t_load:5.2f}s  encode {t_enc:5.2f}s  predict {t_pred:5.2f}s  "
              f"masks {len(res)}")
        print(f"   bbox {got}  기대 {BOX}  최대오차 {off}px (허용 {tol})  "
              f"-> {'OK' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"{key}: bbox off by {off}px (>{tol})")

        # SAM 3's headline feature; nothing else in the catalog has it.
        if getattr(be, "supports_text", False):
            try:
                tres = be.predict_text("orange rectangle")
                tm = tres.best() if callable(tres.best) else tres.best
                if tm is not None and (tm > 0).any():
                    tys, txs = np.where(tm > 0)
                    tgot = (int(txs.min()), int(tys.min()),
                            int(txs.max()) + 1, int(tys.max()) + 1)
                    toff = max(abs(a - b) for a, b in zip(tgot, BOX))
                    print(f"   text  bbox {tgot}  최대오차 {toff}px  "
                          f"-> {'OK' if toff <= tol else 'FAIL'}")
                    if toff > tol:
                        failures.append(f"{key}: text bbox off by {toff}px")
                else:
                    print("   text  빈 마스크 -> FAIL")
                    failures.append(f"{key}: text prompt returned nothing")
            except Exception as e:
                print(f"   text  FAIL {type(e).__name__}: {e}")
                failures.append(f"{key} text: {type(e).__name__}: {e}")
        ran += 1
        be.unload()
    except Exception as e:
        print(f"   FAIL {type(e).__name__}: {e}")
        failures.append(f"{key}: {type(e).__name__}: {e}")
    finally:
        try:
            import torch
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        except Exception:
            pass
    print()

print(f"실행 {ran} · 건너뜀 {skipped} · 실패 {len(failures)}")
if failures:
    for f in failures:
        print("  FAIL", f)
    raise SystemExit(1)
print("BACKENDS SMOKE OK")
