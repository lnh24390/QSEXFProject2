"""Train two epochs of YOLO-seg and check where the files landed.

The point is not the model quality -- it is the layout:
  * the pretrained checkpoint goes to weights/yolo/pretrained/, not the CWD
  * the run goes to runs/<model_key>/<run_name>/, not runs/<run_name>/

    uv run --no-project --python .venv/Scripts/python.exe tests/smoke_train.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dsl.core.deps import BY_MODULE
from dsl.train.jobs import TrainJob
from dsl.train.trainer_ultralytics import _pretrained_dir, _resolve_weights

failures = []


def check(label, ok, detail=""):
    print(f"  [{'OK ' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


print("=== checkpoint path resolution ===")
pre = _pretrained_dir()
print(f"  pretrained dir  {pre}")
check("bare name -> weights/yolo/pretrained/",
      Path(_resolve_weights("yolo11n-seg.pt")).parent == pre)
check("absolute path passes through",
      _resolve_weights(r"C:\models\custom.pt") == r"C:\models\custom.pt")
check("path with separator passes through",
      _resolve_weights("runs/x/weights/best.pt") == "runs/x/weights/best.pt")
check("empty passes through", _resolve_weights("") == "")

if not BY_MODULE["ultralytics"].installed or not BY_MODULE["torch"].installed:
    print("\nultralytics/torch 미설치 -> 학습 단계 건너뜀")
    raise SystemExit(1 if failures else 0)

# --- a two-image YOLO-seg dataset ------------------------------------------
tmp = Path(tempfile.mkdtemp(prefix="dsl_train_"))
ds = tmp / "ds"
for split in ("train", "val"):
    (ds / "images" / split).mkdir(parents=True)
    (ds / "labels" / split).mkdir(parents=True)
    for i in range(2):
        img = np.full((320, 320, 3), 30, np.uint8)
        cv2.rectangle(img, (80, 80), (240, 240), (60, 160, 220), -1)
        cv2.imwrite(str(ds / "images" / split / f"i{i}.jpg"), img)
        # class 0, a square polygon in normalised coords
        poly = "0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75"
        (ds / "labels" / split / f"i{i}.txt").write_text(poly + "\n", encoding="utf-8")
(ds / "data.yaml").write_text(
    f"path: {ds.as_posix()}\ntrain: images/train\nval: images/val\n"
    f"names:\n  0: box\n", encoding="utf-8")

runs_root = tmp / "runs"
job = TrainJob(model_key="yolo11-seg", variant="yolo11n-seg.pt",
               dataset_dir=str(ds), dataset_format="yolo", classes=["box"],
               project_dir=str(runs_root), run_name="smoke",
               device="0", hyper={"epochs": 2, "imgsz": 320, "batch": 2,
                                  "workers": 0, "patience": 5, "plots": False})
job.fill_defaults()
job.save()          # what TrainingProcess.start() does before spawning

root_pt_before = {p.name for p in ROOT.glob("*.pt")}

print("\n=== training (2 epochs) ===")
from dsl.train import trainer_ultralytics
try:
    summary = trainer_ultralytics.train(job)
    print(f"  save_dir  {summary.get('save_dir')}")
    print(f"  best      {summary.get('best')}")
    check("training returned a summary", bool(summary.get("save_dir")))
except Exception as e:
    check("training ran", False, f"{type(e).__name__}: {e}")
    summary = {}

print("\n=== layout ===")
expected_run = runs_root / "yolo11-seg" / "smoke"
check("run dir is runs/<model_key>/<run_name>/", expected_run.is_dir(),
      str(expected_run))
best = expected_run / "weights" / "best.pt"
check("best.pt written", best.is_file())

# job.json must live with the weights it describes: TrainJob.run_dir and the
# trainers' output directory are the same path, and drifting apart once left
# an orphaned job.json in runs/<run_name>/.
check("TrainJob.run_dir matches the trainer output",
      job.run_dir.resolve() == expected_run.resolve(), str(job.run_dir))
check("job.json sits beside the results",
      (expected_run / "job.json").is_file())
stray = runs_root / "smoke"
check("no orphaned run dir without the model key", not stray.exists(),
      str(stray) if stray.exists() else "")

ckpt = _pretrained_dir() / "yolo11n-seg.pt"
check("pretrained checkpoint in weights/yolo/pretrained/", ckpt.is_file(),
      str(ckpt))

root_pt_after = {p.name for p in ROOT.glob("*.pt")}
leaked = root_pt_after - root_pt_before
check("no .pt leaked into the project root", not leaked, str(leaked or ""))

shutil.rmtree(tmp, ignore_errors=True)

print()
if failures:
    print(f"TRAIN SMOKE FAILED ({len(failures)}): " + ", ".join(failures))
    raise SystemExit(1)
print("TRAIN SMOKE OK")
