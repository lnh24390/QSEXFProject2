# 분리수거 도우미 — 릴리즈 2026-09-15

Android 에서 YOLO 로 쓰레기를 검출하고 분리수거 방법을 안내하는 Flutter 앱입니다.

## 들어 있는 것
- `app/` — Flutter 앱 소스, 설정(JSON), 앱 아이콘, **모델 10개(.tflite)**
- `ml/` — 데이터셋 변환·학습·변환·평가 스크립트, **학습 가중치(.pt)**
- 문서 — `README.md`, `OPTIONS.md`(설정 설명), `RECOGNITION_IMPROVEMENT_REPORT.md`(모델 성능 분석)

파일 110개 / 약 186 MB (압축 전)

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
