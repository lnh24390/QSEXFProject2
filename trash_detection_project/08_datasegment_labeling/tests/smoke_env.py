"""Environment smoke test: what is installed, and does CPU-only still work.

DEVELOPMENT.md 2.2 requires the app to run with no CUDA at all, so the checks here
are deliberately non-fatal about GPU: a missing GPU is a pass, not a failure.
What *would* be a failure is the probe raising, or CPU inference not working.

    uv run --no-project --python .venv/Scripts/python.exe tests/smoke_env.py
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dsl.core.settings import APP_DIR, Settings, resolve_under_app
from dsl.sam.downloader import cuda_guidance, detect_nvidia_gpu, probe_hardware

failures = []


def check(label, ok, detail=""):
    print(f"  [{'OK ' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


print("=== paths ===")
print(f"  APP_DIR        {APP_DIR}")
print(f"  frozen         {getattr(sys, 'frozen', False)}")
st = Settings.load()
print(f"  weights_path   {st.weights_path}")
print(f"  runs_path      {st.runs_path}")
check("weights/ is under APP_DIR", st.weights_path.parent == APP_DIR)
check("relative paths ignore CWD",
      resolve_under_app("weights") == APP_DIR / "weights")

print()
print("=== libraries ===")
present = {}
for mod in ("torch", "torchvision", "onnxruntime", "ultralytics",
            "transformers", "sam2", "segment_anything", "pycocotools",
            "psutil", "PySide6", "cv2", "numpy"):
    try:
        present[mod] = importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        present[mod] = False
    print(f"  {mod:<18} {'yes' if present[mod] else 'no'}")

for required in ("PySide6", "cv2", "numpy"):
    check(f"{required} present (required)", present[required])

print()
print("=== hardware probe (must never raise) ===")
try:
    hw = probe_hardware(st.weights_path)
    check("probe_hardware() returned", True)
    print(f"  summary        {hw.summary()}")
    print(f"  device         {hw.device}")
    print(f"  torch_version  {hw.torch_version or '(none)'}")
    print(f"  torch_error    {hw.torch_error or '(none)'}")
    check("device is a known value", hw.device in ("cpu", "cuda", "mps"))
except Exception as e:                      # the whole point of the check
    check("probe_hardware() returned", False, f"{type(e).__name__}: {e}")
    hw = None

print()
print("=== cuda guidance ===")
gpu = detect_nvidia_gpu()
print(f"  nvidia-smi     {gpu or '(no NVIDIA GPU)'}")
g = cuda_guidance(hw) if hw else ""
if g:
    print("  guidance:")
    for line in g.splitlines():
        print(f"    {line}")
else:
    print("  guidance       (none - nothing to fix)")
# No GPU must mean no nagging; a working GPU must also mean no nagging.
if not gpu:
    check("no GPU -> no guidance", g == "")
elif hw and hw.has_cuda:
    check("GPU in use -> no guidance", g == "")
else:
    check("GPU present but unused -> guidance offered", g != "")

print()
print("=== CPU inference (torch-free path) ===")
if present["onnxruntime"]:
    try:
        import numpy as np
        from dsl.sam.backends import create_backend
        be = create_backend("sam2.1_tiny_onnx", st.weights_path,
                            device="cpu", precision="fp32")
        if not be.checkpoint.exists():
            print(f"  skipped: {be.checkpoint} not downloaded")
        else:
            be.load()
            rgb = np.full((480, 640, 3), 30, np.uint8)
            rgb[150:330, 220:420] = (230, 180, 60)
            be.set_image(rgb, key="synthetic")
            res = be.predict(points=[(320.0, 240.0, 1)], multimask=True)
            m = res.best() if callable(res.best) else res.best
            ys, xs = np.where(m > 0)
            box = (int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max()))
            check("ONNX SAM segments on CPU", True, f"bbox x{box[0]}..{box[1]} y{box[2]}..{box[3]}")
            # the synthetic square is 220..419 x 150..329; allow a few px slack
            tight = (abs(box[0] - 220) <= 8 and abs(box[1] - 419) <= 8
                     and abs(box[2] - 150) <= 8 and abs(box[3] - 329) <= 8)
            check("mask matches the synthetic square", tight)
            be.unload()
    except Exception as e:
        check("ONNX SAM segments on CPU", False, f"{type(e).__name__}: {e}")
else:
    print("  skipped: onnxruntime not installed")

print()
if failures:
    print(f"ENV SMOKE FAILED ({len(failures)}): " + ", ".join(failures))
    raise SystemExit(1)
print("ENV SMOKE OK")
