"""Check the frozen build: bundled sources, paths, and external-python training.

Needs `dist/DataSegmentLabeling/` to exist (scripts/build_exe.py). Proves the
one thing the frozen app cannot do for itself -- run training -- actually
works when pointed at an outside interpreter.

    uv run --no-project --python .venv/Scripts/python.exe tests/smoke_frozen.py
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist" / "DataSegmentLabeling"
EXE = DIST / "DataSegmentLabeling.exe"

failures = []


def check(label, ok, detail=""):
    print(f"  [{'OK ' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


if not EXE.exists():
    print(f"빌드가 없습니다: {EXE}\n  먼저: python scripts/build_exe.py")
    raise SystemExit(0)

print("=== bundle layout ===")
src_root = DIST / "_internal" / "dsl_src"
check("dsl 소스 동봉", (src_root / "dsl" / "train" / "cli.py").is_file(),
      str(src_root))
for d in ("weights", "configs", "datasets", "runs"):
    check(f"{d}/ 생성됨", (DIST / d).is_dir())
check("README.txt 생성됨", (DIST / "README.txt").is_file())

print("\n=== exe --paths ===")
r = subprocess.run([str(EXE), "--paths"], capture_output=True, text=True,
                   encoding="utf-8", errors="replace", timeout=120)
out = r.stdout or ""
print("  " + "\n  ".join(l for l in out.splitlines() if l.strip())[:600])
check("frozen 으로 인식", "frozen        : True" in out)
check("APP_DIR 이 exe 폴더", f"APP_DIR       : {DIST}" in out)
check("weights 가 exe 옆", f"weights       : {DIST / 'weights'}" in out)

print("\n=== exe --deps (번들 제외 패키지 안내) ===")
subprocess.run([str(EXE), "--deps"], capture_output=True, timeout=120)
diag = DIST / "diagnostics.txt"
text = diag.read_text(encoding="utf-8") if diag.exists() else ""
check("torch 를 미설치로 보고", "[없음  ] PyTorch" in text)
check("설치 명령을 안내", "download.pytorch.org/whl/cu126" in text)
check("onnxruntime 은 포함", "[설치됨] ONNX Runtime" in text)

# --- the real question: can an outside python train from these sources? -----
print("\n=== external-python training (frozen 경로 그대로) ===")
try:
    import torch  # noqa: F401
    import ultralytics  # noqa: F401
    have_stack = True
except ImportError:
    have_stack = False

if not have_stack:
    print("  torch/ultralytics 없음 -> 건너뜀")
else:
    tmp = Path(tempfile.mkdtemp(prefix="dsl_frozen_"))
    ds = tmp / "ds"
    for split in ("train", "val"):
        (ds / "images" / split).mkdir(parents=True)
        (ds / "labels" / split).mkdir(parents=True)
        for i in range(2):
            img = np.full((320, 320, 3), 30, np.uint8)
            cv2.rectangle(img, (80, 80), (240, 240), (60, 160, 220), -1)
            cv2.imwrite(str(ds / "images" / split / f"i{i}.jpg"), img)
            (ds / "labels" / split / f"i{i}.txt").write_text(
                "0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n", encoding="utf-8")
    (ds / "data.yaml").write_text(
        f"path: {ds.as_posix()}\ntrain: images/train\nval: images/val\n"
        f"names:\n  0: box\n", encoding="utf-8")

    sys.path.insert(0, str(ROOT))
    from dsl.train.jobs import TrainJob
    runs = tmp / "runs"
    job = TrainJob(model_key="yolo11-seg", variant="yolo11n-seg.pt",
                   dataset_dir=str(ds), dataset_format="yolo", classes=["box"],
                   project_dir=str(runs), run_name="frozen",
                   device="0", hyper={"epochs": 1, "imgsz": 320, "batch": 2,
                                      "workers": 0, "patience": 5})
    job.fill_defaults()
    job_path = job.save()

    # Exactly what runner.TrainingProcess does in a frozen build.
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(src_root)
    cmd = [sys.executable, "-u", "-m", "dsl.train.cli", str(job_path)]
    print(f"  PYTHONPATH={src_root}")
    print(f"  $ {' '.join(cmd)}")
    p = subprocess.run(cmd, cwd=str(src_root), env=env, capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=1800)
    check("학습 프로세스 종료코드 0", p.returncode == 0,
          "" if p.returncode == 0 else (p.stdout or "")[-500:] + (p.stderr or "")[-500:])
    check("메트릭 라인 출력", "@@DSL_METRIC@@" in (p.stdout or ""))
    best = runs / "yolo11-seg" / "frozen" / "weights" / "best.pt"
    check("best.pt 가 runs/<모델키>/<이름>/ 에", best.is_file(), str(best))
    shutil.rmtree(tmp, ignore_errors=True)

print()
if failures:
    print(f"FROZEN SMOKE FAILED ({len(failures)}): " + ", ".join(failures))
    raise SystemExit(1)
print("FROZEN SMOKE OK")
