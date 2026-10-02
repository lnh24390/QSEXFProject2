# DataSegmentLabeling — 개발 지침 (Project Guidelines)

이 문서는 본 저장소에서 작업하는 모든 사람이 지켜야 할 규칙이다.
코드를 작성하기 전에 이 문서를 먼저 읽는다.

---

## 1. 프로젝트 목표

1. **라벨링**: 이미지를 불러와 클릭 몇 번으로 SAM(1/2/3/3.1)이 마스크를 만들고,
   사용자가 클래스를 지정해 인스턴스 세그먼트 라벨을 만든다.
2. **내보내기**: 하나의 라벨 원본(JSON)에서 YOLO-seg / COCO / 시맨틱 마스크 PNG
   포맷으로 변환한다.
3. **학습**: YOLO26, YOLO12(검출), YOLO11, YOLOv9, YOLOv8, RT-DETR(Baidu),
   Mask R-CNN, ConvNeXt(UPerNet), SegFormer 를 GUI에서 설정·학습·모니터링한다.

---

## 2. 기술 스택 결정 사항 (변경 금지, 변경 시 이 문서부터 수정)

| 항목 | 선택 | 이유 |
|---|---|---|
| GUI | **PySide6 (Qt6, LGPLv3)** | `QGraphicsView` 로 줌/팬/정점편집이 기본 제공, `QUndoStack`, 도킹 패널, 시그널 기반 스레딩. CustomTkinter(Tk Canvas)는 정점 수천 개에서 성능이 무너지고 뷰 변환이 없어 부적합 |
| 배열/영상 | numpy, opencv-python | 이미 설치됨 |
| 딥러닝 | torch (cu126 휠) | GPU 학습용. **선택 의존성** — 없으면 CPU로 돈다(2.2절) |
| 라벨 저장 | 이미지 1장 = JSON 1개 | 부분 로딩·부분 저장 가능, git diff 가능 |
| 학습 실행 | **별도 subprocess** | GUI 프리징 방지, CUDA 메모리 격리, 죽어도 GUI는 산다 |
| 파이썬 버전/환경 | **Python 3.13, uv** | `uv`가 인터프리터 설치·가상환경·의존성 설치를 한 도구로 처리. `.python-version` 으로 버전 고정 |

**무거운 의존성(ultralytics, segment-anything, sam2, transformers)은 절대
import 시점에 최상위에서 불러오지 않는다.** 반드시 함수 내부 지연 임포트로 처리하고,
없으면 사용자에게 설치 명령을 안내한다. 앱은 SAM/학습 라이브러리가 하나도 없어도
실행되고 수동 라벨링이 가능해야 한다.

### 2.1 개발 환경 준비 (uv)

인터프리터 설치·가상환경·의존성 관리는 전부 `uv`로 한다. `pip`/`venv`를 직접
쓰지 않는다. 저장소 루트의 `.python-version`(3.13)이 버전을 고정한다.

```bash
# uv 설치 (최초 1회, 이미 있으면 생략)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 프로젝트 파이썬 준비 (.python-version 의 3.13을 자동으로 받아 씀)
uv python install
uv venv                       # .venv 생성
source .venv/bin/activate     # Windows: .venv\Scripts\activate

# 의존성 설치
uv pip install -r requirements.txt
uv pip install -r requirements-optional.txt   # SAM/학습 라이브러리 중 쓸 것만
```

### 2.2 CUDA 없이도 돈다 (필수)

**CUDA 라이브러리가 하나도 없어도 앱은 CPU만으로 실행되어야 한다.**
GPU는 가속 수단이지 실행 조건이 아니다.

- `torch` 가 아예 없어도, 있어도 **CPU 전용 빌드**여도, CUDA DLL이 깨져
  있어도 앱은 뜨고 수동 라벨링이 된다.
- `probe_hardware()` 는 torch 로딩 실패를 **삼키지 않고** `Hardware.torch_error`
  에 담아 하드웨어 패널에 보여준 뒤 `device="cpu"` 로 계속 간다.
  (윈도우에서 흔한 실패: CUDA DLL 누락 `OSError: [WinError 126]`, 드라이버 불일치)
- torch 없이 SAM을 쓰려면 **ONNX 백엔드**를 쓴다 (`onnxruntime`, 13MB).
  `sam2.1_tiny_onnx` / `sam2.1_small_onnx` 가 여기 해당한다.
- 학습(YOLO/Mask R-CNN/SegFormer)은 torch가 필요하다. 없으면 학습 패널에서
  **설치 명령을 안내**하고 기능을 막되, 앱 자체는 계속 동작한다.

**GPU 감지와 안내**: `cuda_guidance()` 가 nvidia-smi 로 GPU를 torch와 무관하게
감지해, "GPU는 있는데 torch가 CPU 전용" 같은 불일치를 잡아 **정확한 설치
명령**을 돌려준다. GPU가 없으면 아무것도 권하지 않는다(CPU 모드가 정답).

### 2.3 실행파일(frozen) 배치 규칙

PyInstaller 번들에서 `__file__` 은 임시 추출 폴더를 가리키므로
`Path(__file__).parents[N]` 로 앱 루트를 구하면 **안 된다.**

- 앱 루트는 `dsl.core.settings.APP_DIR` 하나로만 구한다.
  frozen 이면 `sys.executable` 의 폴더, 아니면 저장소 루트.
- `weights/` `configs/` `datasets/` `runs/` 는 전부 **실행파일과 같은 위치**에
  만든다. 설치 시점에 `weights/` 를 생성한다.
- 설정에 담긴 상대 경로는 `resolve_under_app()` 로 푼다. **CWD 기준 금지** —
  앱을 어디서 실행했는지에 따라 폴더 위치가 달라진다.
  subprocess 로 넘기는 경로(`TrainJob.project_dir` 등)는 **반드시 절대경로**로.
  트레이너는 cwd 가 달라 상대경로가 다른 곳을 가리킨다.
- **학습은 frozen 에서 외부 파이썬으로 돈다.** exe 는 `-m` 을 받지 못하고
  torch 도 없다. `settings.train_python` 이 인터프리터를,
  `settings.train_source_root()` 가 PYTHONPATH 를 정한다. `build.spec` 은
  `dsl` 의 .py 소스를 `_internal/dsl_src/` 에 함께 넣어 외부 파이썬이
  `dsl.train.cli` 를 임포트할 수 있게 한다.

---

## 3. 메모리 규칙 (가장 중요)

데이터셋은 수만 장을 가정한다. **"전부 불러오기"는 어떤 경우에도 금지.**

### 3.1 파일 목록
- `QAbstractListModel` + `QListView` 가상화. 행 위젯을 미리 만들지 않는다.
- 썸네일은 **화면에 보이는 행만** 요청한다 (`data()` 호출 시점 = 가시 행).
- 썸네일은 백그라운드 스레드풀(`QThreadPool`)에서 디코딩하고
  **LRU 캐시(기본 512개, `settings.thumb_cache_items`)** 에만 유지한다.
- 캐시에 없으면 즉시 placeholder를 그리고, 완성되면 그 행만 갱신한다.

### 3.2 원본 이미지
- 메모리에는 **현재 이미지 + 프리페치 1~2장**만 둔다.
- `ImageCache` 는 **바이트 예산(기본 1024MB, `settings.image_cache_mb`)** 기반 LRU.
  예산 초과 시 오래된 것부터 즉시 해제한다.
- 이미지 전환 시 이전 `QPixmap`/`np.ndarray` 참조를 명시적으로 끊는다.

### 3.3 SAM 임베딩
- 임베딩은 이미지당 수십~수백 MB다. **현재 이미지 1장분만 보관**한다.
- 이미지 전환 시 `predictor.reset_image()` 후 `torch.cuda.empty_cache()`.

### 3.4 라벨
- 전체 라벨을 메모리에 올리지 않는다. 현재 이미지 것만 로드한다.
- 통계(클래스별 개수)는 매번 전체 스캔하지 말고 `labels/_index.json` 캐시를
  저장 시점에 증분 갱신한다.

### 3.5 완료 항목 처리
- 저장 성공 시 해당 이미지를 **원본 캐시에서 즉시 방출(evict)** 한다.
- 목록 필터 `미완료만 보기` 를 켜면 완료된 항목은 목록에서 내려간다.
- `자동 다음 이미지` 옵션이 켜져 있으면 저장 후 다음 미완료 이미지로 이동한다.

### 3.6 학습
- 학습은 subprocess. GUI 프로세스는 stdout 라인만 읽어 로그 패널에 흘린다.
- 로그 패널은 최대 5000줄만 유지(그 이상은 앞에서 버린다).

---

## 4. 코드 규칙

- Python 3.13, 타입 힌트 사용, `from __future__ import annotations`.
- 파일 경로는 `pathlib.Path`. 프로젝트 내부 저장은 **상대경로 + POSIX 슬래시**.
- UI 문자열은 한국어, 코드 식별자/주석은 영어 허용(기존 파일 스타일을 따른다).
- Qt 시그널은 `Signal(...)` 명시적 타입. 워커 스레드에서 위젯을 직접 만지지 않는다.
- 예외는 삼키지 말고 상태바/로그에 남긴다. `except Exception: pass` 금지.
- 새 모델/SAM 백엔드는 **레지스트리에 등록만** 하면 UI에 자동 노출되게 만든다
  (`dsl/sam/catalog.py`, `dsl/train/registry.py`). UI 코드를 고치지 않는다.

---

## 5. 디렉터리 구조

### 5.1 소스

```
main.py                     진입점. --paths / --deps 진단 플래그
build.spec                  PyInstaller 스펙 (무거운 패키지 제외)

dsl/
  core/                     GUI·모델에 의존하지 않는 순수 로직
    project.py                프로젝트/클래스 정의, 라벨 인덱스
    annotation.py             ImageAnnotation, Shape (라벨 원본 포맷)
    mask_utils.py             마스크 <-> 폴리곤 변환, 단순화
    image_cache.py            바이트 예산 LRU (3.2절)
    exporters.py              YOLO / COCO / 마스크PNG 내보내기
    importers.py              YOLO·COCO·VOC·LabelMe 가져오기 (8절)
    settings.py               Settings, APP_DIR, resolve_under_app()
    deps.py                   선택 의존성 레지스트리 + 설치 명령 (2.2절)
  sam/
    base.py                   SamBackend 추상 클래스, 정밀도 적용
    catalog.py                ModelSpec 카탈로그, local_path() (5.2절)
    backends.py               SAM 1 / 2 / 3 / Ultralytics / ONNX 구현
    downloader.py             체크포인트 다운로드, probe_hardware(),
                              cuda_guidance() (2.2절)
  train/
    registry.py               TrainerSpec 카탈로그 (모델 추가는 여기만)
    jobs.py                   TrainJob — GUI가 쓰고 CLI가 읽는 JSON
    runner.py                 subprocess 실행, resolve_python() (2.3절)
    cli.py                    subprocess 진입점
    trainer_ultralytics.py    YOLO 계열 + RT-DETR
    trainer_maskrcnn.py       Mask R-CNN (torchvision)
    trainer_hfseg.py          SegFormer / ConvNeXt-UPerNet
  ui/
    main_window.py            메뉴·액션·시그널 배선
    canvas.py                 QGraphicsView (줌/팬/도구)
    items.py                  QGraphicsItem (도형/정점)
    workers.py                SAM 워커 스레드
    theme.py                  스타일시트
    panels/                   file / class / shape / sam / train
    dialogs/                  model_manager, data_io

tests/                      스모크 테스트 (pytest 아님, 직접 실행)
  smoke_pipeline.py           내보내기·가져오기·라벨 포맷
  smoke_gui.py                오프스크린 GUI 조작 + 메모리 규칙
  smoke_env.py                경로 해석·하드웨어 프로브·CPU 추론
  smoke_backends.py           설치된 SAM 백엔드 전부 (--heavy 로 SAM 3)
  smoke_train.py              2 에포크 학습 + 결과 위치
  smoke_frozen.py             빌드 산출물 + 외부 파이썬 학습

scripts/
  build_exe.py                빌드 + 데이터 폴더 생성
  make_icon.py                assets/icon.svg -> PNG/ICO 재생성
  download_sam3.py            게이트 저장소 로그인 다운로드

assets/
  icon.svg                    아이콘 원본 (이것만 수정한다)
  icon.ico                    exe·작업표시줄용 (생성물, 빌드에 필요해 커밋)
  icon_*.png                  중간 산출물 (git 제외)
```

아이콘을 고칠 때는 **`assets/icon.svg` 만 수정하고** `scripts/make_icon.py` 로
나머지를 다시 만든다. PNG/ICO 를 직접 편집하면 다음 재생성 때 사라진다.

### 5.2 데이터 (git 제외, APP_DIR 기준)

`APP_DIR` 은 저장소 루트, frozen 이면 실행파일이 있는 폴더다(2.3절).

```
weights/                    체크포인트
  sam/
    sam1/ sam2/ sam2.1/ sam3/ sam3.1/    ModelSpec.family 그대로
    onnx/                                ONNX 변환본 (torch 불필요)
    quantized/                           양자화본
  yolo/
    pretrained/                          YOLO·RT-DETR .pt
  catalog_overrides.json                 URL/repo_id 사용자 수정본

datasets/                   내보낸 학습용 데이터셋
runs/<모델키>/<실행이름>/    학습 결과 (job.json, weights/best.pt, ...)
projects/<폴더명>_<해시>/    `이미지 폴더 열기` 로 만든 프로젝트
                            (project.json + labels/. 이미지는 외부에 그대로 둔다)
configs/settings.json       앱 설정
```

**규칙**
- SAM 체크포인트 경로는 `ModelSpec.local_path()` **한 곳**에서만 만든다
  (`weights/sam/<family>/<filename>`). UI/다운로더는 그걸 받아쓴다.
- ultralytics 체크포인트는 `trainer_ultralytics._resolve_weights()` 를 거친다.
  맨 이름을 그대로 넘기면 ultralytics 가 **CWD에 받아 루트를 어지럽힌다.**
- 학습 결과는 반드시 `runs/<모델키>/` 아래로. 모델별 비교가 가능해야 한다.
- **사용자가 고른 이미지 폴더에는 아무것도 쓰지 않는다.** 읽기 전용·네트워크·
  공유 폴더일 수 있다. `Project.image_dir` 은 절대경로를 그대로 담고,
  project.json 과 labels/ 는 `projects/<폴더명>_<경로해시>/` 아래에 만든다.
  해시는 이름이 같은 다른 폴더를 구분하기 위한 것이다.

### 5.3 실행파일 빌드

```bash
uv pip install pyinstaller
uv run --no-project --python .venv/Scripts/python.exe scripts/build_exe.py
uv run --no-project --python .venv/Scripts/python.exe scripts/build_exe.py --clean
```

`scripts/build_exe.py` 가 PyInstaller 를 돌린 뒤, 설치 프로그램이 할 일을
대신한다 — `weights/ configs/ datasets/ runs/` 를 **실행파일과 같은 폴더**에
만들고 `README.txt`(외부 설치 안내)를 넣는다.

**무거운 패키지는 `build.spec` 의 `excludes` 로 뺀다.** torch 하나가 약 4GB라
프로그램(약 300MB)보다 커지고, 전부 지연 임포트라 대부분의 세션은 쓰지도
않는다. 대신 `onnxruntime` 은 포함해 **exe 만으로 SAM 세그먼트가 되게** 한다.
빠진 패키지는 `dsl/core/deps.py` 가 설치 명령과 함께 안내한다(2.2절).

`build.spec` 을 고칠 때 주의할 것:
- PyInstaller 6.x 에서 `cipher=`, `a.zipped_data`, `a.zipfiles`,
  `win_no_prefer_redirects` 가 **제거**됐다. 옛 예제를 복사하면 빌드가 깨진다.
- `datas` 로 `dsl` 의 .py 소스를 `_internal/dsl_src/` 에 함께 넣는다.
  외부 파이썬이 학습할 때 `dsl.train.cli` 를 임포트해야 한다(2.3절).
- `console=False` 라 `sys.stdout` 이 없다. `--paths` / `--deps` 는 exe 옆
  `diagnostics.txt` 에도 쓴다.

**검증은 `tests/smoke_frozen.py`** — 번들 구조, `--paths` 의 APP_DIR,
`--deps` 안내, 그리고 **외부 파이썬으로 실제 학습**까지 확인한다.
`dist/` 가 없으면 그냥 건너뛴다.

### 5.4 빌드 산출물 (git 제외)

```
build/                      PyInstaller 중간 파일 (~19MB, 지워도 됨)
dist/DataSegmentLabeling/   배포물 (~302MB)
  DataSegmentLabeling.exe
  README.txt                외부 설치 안내
  _internal/                PySide6·numpy·cv2·onnxruntime 등
    dsl_src/dsl/            학습용 .py 소스 (외부 파이썬이 임포트, 2.3절)
  weights/ configs/ datasets/ runs/     빌드 시 생성
```

`build/` `dist/` 는 `.gitignore` 에 있다. 커밋하지 않는다.
둘 다 언제든 지워도 된다 — `scripts/build_exe.py` 로 다시 만들어진다.
저장소를 가볍게 유지할 때 `rm -rf build dist` 와 `__pycache__` 정리로
약 330MB 가 줄어든다(소스 자체는 1MB 미만).

---

## 6. 라벨 파일 포맷 (단일 원본, Single Source of Truth)

`labels/<이미지 상대경로 그대로>.json` — **확장자를 포함**한다
(`a/b.jpg` → `labels/a/b.jpg.json`). 확장자를 떼면 `b.jpg` 와 `b.png` 가
같은 라벨 파일을 덮어쓴다. 확장자를 뗀 구버전 경로는 읽기만 지원하며,
저장 시 새 이름으로 자동 이관된다.

```json
{"version":1,"image_path":"a/b.jpg","width":1920,"height":1080,"verified":false,
 "shapes":[{"class_id":0,"points":[[x,y],...],"holes":[[[x,y],...]],
            "meta":{},
            "shape_type":"polygon","score":0.97,"source":"sam2",
            "group_id":null,"uid":"3f2a..."}]}
```

- 좌표는 **원본 이미지 픽셀 절대좌표**(리사이즈/줌과 무관).
- 내보내기(YOLO/COCO/PNG)는 항상 이 JSON에서 파생된다. 역방향 금지.
- **COCO 구멍**: COCO 의 폴리곤 리스트는 *파트의 합집합*이라 구멍을 표현할 수
  없다(구멍이 채워져 나간다). 따라서 `holes` 가 있는 도형은 **비압축 RLE**
  (`{"counts":[...],"size":[h,w]}`)로 내보내고 `area` 도 그 마스크에서 센다.
  구멍이 없으면 지금처럼 폴리곤으로 내보낸다.

---

## 7. 작업 순서 규칙

1. 지침(이 문서) 갱신 → 2. core → 3. sam/train 백엔드 → 4. ui →
5. 문서(README/USER_GUIDE) → 6. **`CHANGELOG.md` 갱신**.
- 새 의존성을 넣으면 `requirements*.txt` 와 README 설치 항목을 함께 갱신한다.

### 7.1 USER_GUIDE 갱신 (기능이 바뀌면 필수)

**사용자가 보는 동작이 바뀌면 `docs/USER_GUIDE.md` 를 같은 커밋에서 고친다.**
코드만 바꾸고 문서를 두면, 문서가 틀린 것보다 나쁜 상태(있다고 적힌 기능을
못 찾는 상태)가 된다.

갱신 대상은 다음이 바뀔 때다:
- **단축키·메뉴 이름·툴바 버튼** → `단축키 요약` 표와 해당 절.
  `main_window.SHORTCUTS_TEXT`(앱 안 F1)도 **같이** 고친다. 둘은 같은 내용을
  담고 있어 한쪽만 고치면 곧바로 어긋난다.
- **새 도구/모드** → `4. 라벨링` 아래에 절을 만들고, 비슷한 기존 도구와
  **무엇이 다른지** 한 줄 적는다 (예: `SAM 박스` vs `박스 라벨`).
- **새 내보내기 포맷·학습 모델** → `7. 데이터셋 내보내기` / `8. 학습` 표.
- **설정 항목** → 어느 메뉴에 있는지와 기본값.
- 조작이 막히거나 헷갈릴 수 있는 변경 → `문제 해결` 표에 증상 → 대처로.

`tests/smoke_gui.py` 가 USER_GUIDE 에 적힌 `Ctrl+…` 단축키가 실제 앱에
존재하는지 검사한다. 문서에만 있는 단축키를 적으면 테스트가 실패한다.

### 7.2 CHANGELOG 규칙

**작업하면서 계속 갱신한다.** 끝나고 몰아 쓰지 않는다 — 왜 그렇게 고쳤는지는
그 시점에만 정확히 기억난다.

- 형식은 [Keep a Changelog](https://keepachangelog.com/ko/1.1.0/).
  구분은 `추가 / 변경 / 수정 / 제거 / 알려진 제약`.
- 진행 중 작업은 `## [Unreleased] — <날짜>` 아래에 쌓고, 릴리스할 때 버전을 붙인다.
- **버그 수정은 증상과 원인을 같이 적는다.** "probe_hardware 수정"이 아니라
  "CUDA DLL이 없으면 앱이 시작조차 못 하던 문제 — `ImportError` 만 잡고 있었음".
  나중에 같은 걸 다시 밟지 않으려면 원인이 남아야 한다.
- 경로/포맷/디렉터리 구조가 바뀌면 **이전 → 이후를 표나 코드블록으로** 남긴다.
  기존 사용자가 자기 파일을 어디로 옮겨야 하는지 알아야 한다.
- 고치지 못하고 남긴 제약은 `알려진 제약` 에 적는다. 조용히 빼지 않는다.

---

## 8. 기존 라벨 가져오기 → 세그먼트 변환 (필수 기능)

이미 만들어 둔 **박스/폴리곤 라벨을 재활용**해서 세그먼트 데이터로 승격시킨다.

지원 입력 포맷 (`dsl/core/importers.py`):
- YOLO detection txt (`cls cx cy w h`, 정규화)
- YOLO segmentation txt (`cls x1 y1 x2 y2 ...`, 정규화)
- COCO instances json (bbox + segmentation)
- Pascal VOC xml (bndbox)
- LabelMe json (polygon/rectangle)

동작 규칙:
1. 박스는 `shape_type="rectangle"`, `source="imported"` 로 저장하고
   캔버스에서 **점선 박스**로 구분해 그린다.
2. `박스→세그먼트` 모드에서 박스를 **클릭하면** 그 박스를 SAM 박스 프롬프트로 넣어
   마스크를 만들고, 같은 `class_id`를 유지한 채 폴리곤으로 **교체**한다.
   (원본 박스는 `meta.src_bbox` 로 남겨 되돌릴 수 있게 한다.)
3. `현재 이미지 일괄 변환` / `프로젝트 전체 일괄 변환`(백그라운드 워커) 제공.
   전체 변환도 이미지는 한 번에 한 장씩만 메모리에 올린다(3장 규칙).
4. 변환 결과가 마음에 안 들면 Ctrl+Z 로 되돌리고, 점 프롬프트로 보정한다.
