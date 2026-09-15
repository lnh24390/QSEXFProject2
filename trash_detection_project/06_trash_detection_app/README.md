# 분리수거 도우미 (YOLO 실시간 쓰레기 검출 · Android)

카메라 영상에서 쓰레기를 실시간으로 검출하고, 화면 하단에 **어디에 버려야 하는지** 계속 안내하는 Flutter(Android) 앱입니다.
화면에 쓰레기가 없으면 "분리수거할 쓰레기를 화면에 담아주세요" 라고 표시합니다.

```
yolo_trash_detection/
├── app/   Flutter 앱 (Android 주 타깃)
├── ml/    데이터셋 변환 · YOLO 학습 · ONNX/TFLite(LiteRT) 변환 스크립트
└── README.md
```

## 기술 스택 (모두 최신 안정 버전 기준, 2026-09)

| 구성 | 버전 |
|---|---|
| Flutter / Dart | 3.47.x stable / 3.13.x |
| ultralytics_yolo (Flutter 플러그인) | ^0.6.14 (Android: LiteRT 2.x, minSdk 23) |
| permission_handler | ^13.0.2 |
| Ultralytics (Python) | >= 8.4.142 |
| 기본 모델 | YOLO26n (COCO 80클래스) → 자체 학습 모델로 교체 가능 |
| 테스트 기기 | Galaxy S21 Ultra (Android 14, Exynos 2100 / Snapdragon 888) |

## 동작 방식

1. `YOLOView` 가 카메라 프레임을 네이티브(LiteRT)에서 추론하고 박스를 그립니다.
2. 프레임마다 나온 결과를 `DetectionAggregator` 가 클래스별로 합치고, 1.2초 동안 유지해 깜빡임을 막습니다.
3. `RecyclingGuide` 가 클래스명을 `assets/config/recycling_guide.json` 으로 분리수거 카테고리(페트·플라스틱·캔·유리·종이·비닐·음식물·일반…)에 매핑합니다.
4. 하단 `GuidePanel` 이 우선순위 순으로 최대 3개 품목의 **버릴 곳 + 버리는 방법**을 표시하고, 없으면 촬영 안내 문구를 표시합니다.

## 빠른 시작 (앱)

이 PC 는 설치가 끝나 있습니다.

| 도구 | 위치 |
|---|---|
| Flutter 3.47.3 stable (Dart 3.13.3) | `C:\Users\USER\dev\flutter` |
| Android SDK (platform 36 + 37.0, build-tools 36.0.0, adb) | `%LOCALAPPDATA%\Android\Sdk` |
| JDK | Temurin 25 (Gradle 9.3.1 + AGP 9.1.0 과 호환됨) |

PATH 와 `ANDROID_HOME` 은 사용자 환경변수에 등록되어 있습니다. Android 라이선스도 수락 완료라
`flutter doctor` 의 Android toolchain 이 ✓ 입니1다. `app/android/` 는 `setup_android.ps1` 로 생성·패치되어
있습니다(카메라 권한, release ProGuard 규칙, Kotlin 플러그인 적용).

```powershell
cd app
flutter pub get
flutter run --release      # USB 디버깅 켠 기기 연결 후
```

- `app/assets/models/*.tflite` 중 `AppConfig.defaultModelAsset` 을 먼저 쓰고, 없으면 번들된 첫 번째 모델을 씁니다. 하나도 없으면 "사용할 모델이 없습니다" 안내가 나옵니다(자동 다운로드는 하지 않습니다).
- 앱 우측 상단 ⚙(모델 선택)에서 번들된 모델을 즉시 바꿀 수 있습니다.
- 임계값·유지시간·표시 개수는 `app/lib/config/app_config.dart` 에서 조정합니다.

### VS Code 에서 바로 실행하기

1. **확장 설치** — 확장 탭에서 `Flutter` (Dart-Code.flutter) 설치. Dart 확장은 함께 깔립니다.
2. **폴더 열기** — `app` 폴더를 여세요. 리포지토리 루트(`yolo_trash_detection`)를 열면 `pubspec.yaml`
   이 최상위에 없어서 Flutter 확장이 프로젝트를 인식하지 못합니다.
3. **기기 선택** — 하단 상태바 오른쪽의 기기 이름을 클릭 → 목록에서 `SM G998N` 선택.
   (USB 디버깅을 켜고 케이블로 연결한 뒤 폰의 "USB 디버깅 허용" 팝업을 수락해야 목록에 뜹니다)
4. **실행** — `F5` (디버그) 또는 `Ctrl+F5` (디버그 없이). 첫 빌드는 몇 분 걸립니다.

`app/.vscode/launch.json` 에 구성이 준비되어 있습니다. `F5` 를 누르기 전에 실행 및 디버그 탭
(`Ctrl+Shift+D`) 상단 드롭다운에서 고르세요:

| 구성 | 용도 |
|---|---|
| `trash_sorter (debug)` | 기본. hot reload 됨. 추론은 느림 |
| `trash_sorter (profile)` | 성능 측정용. DevTools 로 프레임·FPS 확인 |
| `trash_sorter (release)` | 실제 배포와 동일. 검출 속도 확인은 이걸로 |

`app/.vscode/settings.json` 은 저장 시 `dart format`, Flutter SDK 경로 고정, `build/`·모델 파일
검색 제외를 설정해 둡니다.

- **Hot reload** 는 디버그 모드에서만 됩니다. `lib/` 를 저장하면 즉시 반영되므로 UI 문구·레이아웃
  수정은 디버그로, 검출 속도(FPS) 측정은 릴리스로 하세요. 디버그 빌드는 추론이 눈에 띄게 느립니다.
- `AppConfig` 값은 `const` 라 hot reload 로는 안 바뀝니다. **hot restart**(`Ctrl+Shift+F5`)를 쓰세요.
- 터미널에서 하려면 VS Code 통합 터미널에서 `cd app; flutter run --release` 와 동일합니다.
- 이미 빌드된 APK 를 기기에 다시 올리기만 할 때는 `flutter install --release` 가 가장 빠릅니다.

## 모델 학습 · 변환 (ml/)

Android 플러그인은 **`format=litert nms=None`** 으로 만든 `.tflite` 를 요구합니다.
LiteRT 변환은 Linux x86 / macOS 에서만 되므로 Windows 에서는 **Docker Desktop** 을 켜고 `--docker` 를 씁니다(자동 전환).

```powershell
cd ml
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt          # GPU 학습이면 CUDA torch 를 먼저 설치

# 1) TACO 데이터셋 → YOLO 형식 (Desktop 의 TACO-master.zip 사용)
python scripts/prepare_taco.py --taco-zip "C:/Users/pc/Desktop/TACO-master.zip" --download

# 2) 학습 (yolo26n 전이학습)
python scripts/train.py --data datasets/taco_yolo/data.yaml --model yolo26n.pt --epochs 100 --imgsz 640

# 3) Android 용 변환 + 앱에 복사 (Windows 는 Docker 자동 사용)
python scripts/export.py --weights runs/trash_yolo26n/weights/best.pt --install-to-app --app-name trash_v1

# (선택) ONNX, 검증   ※ 양자화(INT8/FP16) 변환은 사용하지 않음(비활성화) — 모든 모델은 FP32
python scripts/export.py --weights best.pt --format onnx
python scripts/verify.py --model exports/best.onnx --image sample.jpg
```

GPU 가 없으면 `ml/colab_train_export.ipynb` 를 Google Colab 에서 실행해 학습·변환 후 `.tflite` 만 내려받아 `app/assets/models/` 에 넣으면 됩니다.

### 모델을 바꿀 때 해야 할 일

1. `.tflite` 를 `app/assets/models/` 에 넣는다 (`export.py --install-to-app` 이 자동으로 함).
2. `python scripts/check_guide_coverage.py --tflite ../app/assets/models/<이름>.tflite` 로 클래스 매핑 누락을 확인한다.
3. 누락 클래스는 `app/assets/config/recycling_guide.json` 의 `classes` 에 추가한다(앱 코드 수정 불필요).
4. 기본 모델로 쓰려면 `app_config.dart` 의 `defaultModelAsset` 을 바꾼다. `flutter run` 으로 재빌드.

클래스명 매칭 규칙: 소문자화 후 공백/`-`/`&`/`/` → `_`. `classes` 에 없으면 `rules`(부분 문자열) → `fallback`(일반쓰레기) 순으로 결정.

### 준비된 모델 포맷 (`ml/exports/`, yolo26n 기준)

| 파일 | 포맷 | 용도 |
|---|---|---|
| `weights/yolo26n.pt` | PyTorch | 학습 시작점, 재변환 원본 |
| `exports/yolo26n_litert_model/*.tflite` → `app/assets/models/yolo26n.tflite` | LiteRT/TFLite fp32 (`nms=None`) | **Android 앱** (Galaxy S21 Ultra GPU 가속) |
| `exports/yolo26n.onnx` | ONNX (FP32) | onnxruntime, 서버/데스크톱, QNN·기타 툴체인 입력 |
| `exports/yolo26n.torchscript` | TorchScript | PyTorch C++/모바일 배포 |
| `exports/yolo26n_openvino_model/` | OpenVINO | Intel CPU/iGPU |
| `exports/yolo26n_ncnn_model/` | NCNN | ARM CPU 경량 추론 |

재생성: `python scripts/export.py --weights weights/yolo26n.pt --format all --install-to-app`

### Galaxy S21 Ultra 메모

- Android 14 → minSdk 23 조건 충족. 개발자 옵션에서 USB 디버깅 활성화.
- 국내 모델은 Exynos 2100 이라 Snapdragon 전용 QNN(NPU) 경로는 쓰지 않습니다. 플러그인의 기본 GPU→CPU 자동 폴백으로 동작합니다.
- 720p 카메라 + yolo26n fp32 기준 실시간(20~30fps)이 목표. 부족하면 `480p` 로 낮추거나 더 작은 모델을 쓰세요(양자화 변환은 사용하지 않음).

## 분리수거 카테고리

투명 페트병 · 플라스틱 · 비닐 · 캔/고철 · 유리병 · 종이 · 골판지 · 종이팩 · 스티로폼 · 음식물 · 폐가전/건전지 · 의류 · 일반쓰레기.
문구/색상은 `recycling_guide.json` 에서 지자체 기준에 맞게 수정하세요.

## 문제 해결

| 증상 | 조치 |
|---|---|
| 모델 로드 실패 (`onModelError`) | `nms=None` 으로 export 했는지, 파일이 `assets/models/` 이고 pubspec `assets:` 에 폴더가 있는지 확인 |
| release 빌드에서만 크래시 | `android/app/proguard-rules.pro` 가 적용됐는지 확인 (`setup_android.ps1` 이 추가) |
| Windows 에서 `LiteRT export only supported on Linux x86 and macOS` | Docker Desktop 실행 후 `export.py --docker` 또는 Colab 노트북 사용 |
| FPS 가 낮음 | `cameraResolution` 을 `480p` 로, 모델을 `n` 사이즈로 (양자화 변환은 사용하지 않음) |
| 검출이 안 됨 / 너무 많음 | `AppConfig.confidenceThreshold` 조정 (기본 0.45) |
