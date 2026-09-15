"""릴리즈 압축본 만들기.

빌드 산출물(build/, .dart_tool/, .venv/, runs/, datasets/ 등)은 빼고
소스 + 앱 모델(.tflite) + 학습 가중치(.pt) + 문서를 zip 하나로 묶습니다.

실행: ml/.venv/Scripts/python.exe tool/make_release.py
결과: release/trash_sorter_<날짜>.zip
"""
from __future__ import annotations

import zipfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "release"

# 폴더 이름이 이것과 같으면 통째로 제외 (빌드 산출물·캐시·데이터셋)
SKIP_DIRS = {
    "build",
    ".dart_tool",
    ".venv",
    ".gradle",
    ".idea",
    ".git",
    "__pycache__",
    "runs",
    "datasets",
    "release",
    "ephemeral",
    ".cxx",
}

# 파일 단위 제외
SKIP_SUFFIXES = {".log", ".tmp", ".cache", ".pyc"}
SKIP_NAMES = {".flutter-plugins-dependencies", ".DS_Store", "Thumbs.db"}

# 루트에 둔 문서 중 릴리즈에 넣을 것(그 외 루트 .md 는 작업용 메모로 보고 제외)
ROOT_DOCS = {"README.md", "OPTIONS.md", "RECOGNITION_IMPROVEMENT_REPORT.md"}

# 이미 압축된 형식은 다시 압축해도 거의 안 줄어들고 시간만 오래 걸린다
STORE_SUFFIXES = {".tflite", ".pt", ".png", ".jpg", ".jpeg", ".zip", ".apk"}


def should_skip(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if any(part in SKIP_DIRS for part in rel.parts):
        return True
    if path.name in SKIP_NAMES or path.suffix.lower() in SKIP_SUFFIXES:
        return True
    # 루트 문서는 화이트리스트만 넣는다
    if len(rel.parts) == 1 and path.suffix.lower() == ".md" and path.name not in ROOT_DOCS:
        return True
    # ml/exports 는 변환 산출물이라 .tflite 사본을 빼고 설명(.txt)만 넣는다
    if rel.parts[:2] == ("ml", "exports") and path.suffix.lower() != ".txt":
        return True
    return False


def release_info(files: list[Path], total: int) -> str:
    return f"""# 분리수거 도우미 — 릴리즈 {date.today().isoformat()}

Android 에서 YOLO 로 쓰레기를 검출하고 분리수거 방법을 안내하는 Flutter 앱입니다.

## 들어 있는 것
- `app/` — Flutter 앱 소스, 설정(JSON), 앱 아이콘, **모델 {len([f for f in files if f.suffix == '.tflite'])}개(.tflite)**
- `ml/` — 데이터셋 변환·학습·변환·평가 스크립트, **학습 가중치(.pt)**
- 문서 — `README.md`, `OPTIONS.md`(설정 설명), `RECOGNITION_IMPROVEMENT_REPORT.md`(모델 성능 분석)

파일 {len(files)}개 / 약 {total / 1024 / 1024:.0f} MB (압축 전)

## 빠진 것 (다시 만들 수 있는 것들)
`build/`, `.dart_tool/`, `ml/.venv/`, `ml/runs/`, `ml/datasets/`, `ml/exports/` 의 변환 사본,
그리고 완성된 APK. APK 는 아래 순서로 다시 만들 수 있습니다.

## 빌드 방법
```
cd app
flutter pub get
flutter build apk --release      # build/app/outputs/flutter-apk/app-release.apk
```
Flutter 3.47 이상 stable 이 필요합니다. 기기에 설치하려면 `adb install -r <apk>`.

## 모델을 다시 변환하려면
```
cd ml
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
python scripts/export.py --weights weights/<모델>.pt --install-to-app --app-name <이름>
python scripts/check_guide_coverage.py --tflite ../app/assets/models/<이름>.tflite
```
LiteRT 변환은 Windows 에서 안 되므로 WSL(Linux) 또는 Docker 가 필요합니다.

## 주의
- 지역별 분리수거 규칙(`app/assets/config/recycling_guide.json` 의 `regions`)은 2026-09-15 기준으로
  공개 자료를 확인해 넣은 것입니다. 지자체 기준은 수시로 바뀌니 배출 전 거주지 공고를 확인하세요.
- 앱은 위치 권한(지역 판별)과 카메라 권한을 씁니다.
"""


def main() -> None:
    files: list[Path] = []
    total = 0
    for path in sorted(ROOT.rglob("*")):
        if path.is_dir() or should_skip(path):
            continue
        files.append(path)
        total += path.stat().st_size

    OUT_DIR.mkdir(exist_ok=True)
    info = release_info(files, total)
    (OUT_DIR / "RELEASE_INFO.md").write_text(info, encoding="utf-8")

    zip_path = OUT_DIR / f"trash_sorter_{date.today().isoformat()}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("RELEASE_INFO.md", info)
        for path in files:
            rel = path.relative_to(ROOT)
            method = (
                zipfile.ZIP_STORED
                if path.suffix.lower() in STORE_SUFFIXES
                else zipfile.ZIP_DEFLATED
            )
            z.write(path, rel.as_posix(), compress_type=method)

    size = zip_path.stat().st_size
    print(f"파일 {len(files)}개, 압축 전 {total / 1024 / 1024:.1f} MB")
    print(f"→ {zip_path.relative_to(ROOT)} ({size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
