"""Build the Windows executable and lay out its data folders.

    uv run --no-project --python .venv/Scripts/python.exe scripts/build_exe.py

PyInstaller alone produces the program; this adds the part an installer would
normally do -- creating weights/ configs/ datasets/ runs/ *beside the exe* and
dropping a short note about the packages that are deliberately not bundled
(DEVELOPMENT.md 2.2, 2.3).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist" / "DataSegmentLabeling"
DATA_DIRS = ("weights", "weights/sam", "weights/yolo/pretrained",
             "configs", "datasets", "runs")

NOTE = """DataSegmentLabeling
===================

바로 되는 것
------------
  * 이미지 라벨링 (폴리곤 수동 편집, 가져오기/내보내기)
  * SAM 세그먼트 -- ONNX 백엔드가 포함되어 있어 추가 설치 없이 동작합니다.
    모델은 앱에서 [도구 > SAM 모델 관리자 (Ctrl+M)] 로 내려받으세요.

추가 설치가 필요한 것
---------------------
용량이 큰 패키지는 실행파일에 넣지 않았습니다. torch 하나만 약 4GB라
프로그램(약 300MB)보다 훨씬 커지기 때문입니다. 필요할 때만 설치하세요.

  GPU 가속 + 모든 학습:
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
  YOLO 학습 / RT-DETR / MobileSAM:
    uv pip install ultralytics
  SAM 3, SegFormer, ConvNeXt:
    uv pip install "transformers>=4.57" huggingface_hub

현재 설치 상태는 앱의 [도움말 > 선택 패키지 설치 상태] 또는

    DataSegmentLabeling.exe --deps

폴더 위치
---------
weights/  configs/  datasets/  runs/ 는 모두 이 실행파일과 같은 폴더에
만들어집니다. 폴더째 옮겨도 그대로 동작합니다.

    DataSegmentLabeling.exe --paths    실제 경로 확인

학습
----
실행파일은 학습을 직접 돌리지 못합니다(torch 미포함). torch 가 설치된
파이썬을 [설정 > 학습용 파이썬 경로] 에 지정하면 그대로 학습됩니다.
필요한 소스는 _internal/dsl_src/ 에 함께 들어 있으므로 저장소를 따로
받을 필요는 없습니다.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-pyinstaller", action="store_true",
                    help="이미 빌드된 dist/ 에 폴더/안내만 추가")
    ap.add_argument("--clean", action="store_true", help="build/ dist/ 를 먼저 지움")
    args = ap.parse_args()

    if args.clean:
        for d in (ROOT / "build", ROOT / "dist"):
            if d.exists():
                print(f"제거: {d}")
                shutil.rmtree(d, ignore_errors=True)

    if not args.skip_pyinstaller:
        cmd = [sys.executable, "-m", "PyInstaller", "build.spec", "--noconfirm",
               "--distpath", "dist", "--workpath", "build"]
        print("실행:", " ".join(cmd))
        r = subprocess.run(cmd, cwd=str(ROOT))
        if r.returncode != 0:
            print("PyInstaller 실패", file=sys.stderr)
            return r.returncode

    if not DIST.is_dir():
        print(f"빌드 결과가 없습니다: {DIST}", file=sys.stderr)
        return 1

    print()
    for rel in DATA_DIRS:
        p = DIST / rel
        p.mkdir(parents=True, exist_ok=True)
        print(f"  생성: {rel}/")

    note = DIST / "README.txt"
    note.write_text(NOTE, encoding="utf-8")
    print(f"  생성: {note.name}")

    exe = DIST / "DataSegmentLabeling.exe"
    total = sum(f.stat().st_size for f in DIST.rglob("*") if f.is_file())
    print()
    print(f"완료: {DIST}")
    print(f"  실행파일 {exe.name}  {exe.stat().st_size / 1024**2:,.1f} MB")
    print(f"  전체     {total / 1024**2:,.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
