# DataSegmentLabeling

**SAM(1 / 2 / 2.1 / 3 / 3.1)으로 클릭 한 번에 세그먼트를 만들고, 그대로 세그멘테이션
모델을 학습시키는 데스크톱 툴.**

이미지를 불러와 객체를 클릭하면 SAM이 마스크를 제안하고, 클래스를 지정해 저장합니다.
저장된 라벨은 YOLO-seg / COCO / 시맨틱 마스크 PNG로 내보내 YOLO26·YOLO12·YOLO11·
YOLOv9·YOLOv8·RT-DETR·Mask R-CNN·ConvNeXt·SegFormer 학습에 바로 쓸 수 있습니다.

---

## 왜 PySide6 인가 (CustomTkinter 대신)

| | CustomTkinter (Tk Canvas) | **PySide6 (Qt6, LGPLv3)** |
|---|---|---|
| 줌/팬 | 좌표를 직접 다 계산해야 함 | `QGraphicsView` 변환으로 기본 제공 |
| 폴리곤 수천 정점 | 히트테스트·렌더가 급격히 느려짐 | 씬 그래프 인덱싱으로 안정적 |
| 정점 드래그 편집 | 직접 구현 | `QGraphicsItem` 이동/선택 내장 |
| 무거운 작업 | Tk는 사실상 단일 스레드 | 시그널/슬롯 기반 워커 스레드 |
| 도킹 패널·툴바 | 없음 | `QDockWidget`, `QToolBar` |
| 라이선스 | MIT | LGPLv3 (동적 링크 시 상용 무료) |

라벨링 툴은 "큰 이미지 위에서 수천 개의 점을 다루는 캔버스 앱"이라 Qt가 정답입니다.

---

## 설치

파이썬 인터프리터·가상환경·의존성은 [uv](https://docs.astral.sh/uv/)로 관리합니다.
`.python-version`(3.13)에 맞춰 자동으로 받아 씁니다.

```bash
# 0) uv 설치 (최초 1회)
curl -LsSf https://astral.sh/uv/install.sh | sh   # Windows는 아래 공식 문서 참고

# 1) 파이썬 3.13 + 가상환경
uv python install
uv venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

# 2) 필수
uv pip install -r requirements.txt

# 3) SAM 백엔드는 쓸 것만 (자세한 건 requirements-optional.txt)
uv pip install ultralytics                                              # SAM 2.1 / MobileSAM 간편
uv pip install git+https://github.com/facebookresearch/sam2.git         # SAM 2 / 2.1 정식
uv pip install git+https://github.com/facebookresearch/segment-anything.git   # SAM 1
uv pip install "transformers>=4.57" huggingface_hub                     # SAM 3 / 3.1

# 4) 학습 프레임워크도 쓸 것만
uv pip install ultralytics       # YOLO 계열 + RT-DETR
uv pip install transformers      # SegFormer / ConvNeXt-UPerNet
```

> SAM 라이브러리와 학습 프레임워크가 **하나도 없어도 앱은 실행**되며, 수동 폴리곤
> 라벨링과 기존 라벨 가져오기/내보내기는 그대로 동작합니다.

### CUDA 없이도 됩니다

GPU는 가속 수단이지 실행 조건이 아닙니다. torch가 아예 없어도, CPU 전용
빌드여도, CUDA DLL이 깨져 있어도 앱은 뜨고 라벨링이 됩니다. torch 로딩에
실패하면 그 사유를 **화면에 보여준 뒤** CPU로 계속 갑니다.

torch 없이 SAM까지 쓰려면 ONNX 백엔드만 설치하면 됩니다 (60MB):

```bash
uv pip install onnxruntime
python -m dsl.sam.downloader --install sam2.1_tiny_onnx
```

NVIDIA GPU가 있는데 torch가 그걸 못 쓰고 있으면, **도움말 → 이 PC 정보** 가
정확한 설치 명령을 알려줍니다. GPU가 없으면 아무것도 권하지 않습니다.

## 실행

```bash
python main.py                # 프로젝트 없이 시작
python main.py D:/data/myproj # 프로젝트를 열고 시작
python main.py --paths        # 데이터 폴더 위치 확인
python main.py --deps         # 선택 패키지 설치 상태
```

### 이미지 폴더만 열어서 바로 시작하기

**파일 → 이미지 폴더 열기 (Ctrl+Shift+O)** 로 이미지가 든 폴더를 고르면 바로
라벨링을 시작합니다. 프로젝트를 미리 만들 필요가 없고, 폴더는 다른 드라이브나
네트워크 드라이브에 있어도 됩니다.

**고른 폴더에는 아무것도 쓰지 않습니다.** `project.json` 과 라벨은
`projects/<폴더명>_<해시>/` 에 따로 저장되므로 읽기 전용·공유 폴더도 안전하게
쓸 수 있고, 같은 폴더를 다시 열면 작업을 이어서 합니다.

## 모델 체크포인트 — 내 PC에 맞춰 그때그때 다운로드

번들된 가중치는 없습니다. GPU/VRAM/RAM/디스크를 조사해 맞는 모델을 추천하고,
필요한 것만 `weights/` 로 내려받습니다. (이어받기 지원)

GUI: **도구 → SAM 모델 관리자 (Ctrl+M)**

CLI:

```bash
python -m dsl.sam.downloader --probe                 # 내 PC 정보
python -m dsl.sam.downloader --list                  # 전체 카탈로그 + 설치 여부
python -m dsl.sam.downloader --recommend             # 이 PC 추천 모델
python -m dsl.sam.downloader --install-recommended   # 추천 모델 바로 받기
python -m dsl.sam.downloader --install sam2.1_hiera_small
python -m dsl.sam.downloader --uninstall sam1_vit_h
```

추천 기준 (요약)

| VRAM | 추천 |
|---|---|
| 16GB+ | SAM 3.1 / SAM 3 (텍스트 프롬프트) · SAM 2.1 Hiera-L |
| 10–16GB | SAM 2.1 Hiera-L (fp16) |
| 6–10GB | SAM 2.1 Hiera-B+ |
| 4–6GB | SAM 2.1 Hiera-S |
| 4GB 미만 | SAM 2.1 Hiera-T |
| GPU 없음 | MobileSAM (int8) |

**양자화**는 별도 파일을 받지 않고 로드 시점에 적용합니다 — 정밀도별 체크포인트는
공식 배포처에 존재하지도 않습니다. CUDA는 가중치를 `model.half()` 로 캐스팅하고
(활성값은 autocast), CPU는 `quantize_dynamic` 기반 int8을 씁니다.
모델 패널에서 정밀도를 고를 수 있습니다.

RTX 4060(8GB) · 1280×720 · 점 프롬프트 1회 실측:

| 모델 | fp32 가중치 / 피크 VRAM | fp16 가중치 / 피크 VRAM | 속도 |
|---|---|---|---|
| SAM 1 ViT-B | 358MB / 2770MB | 179MB / 1395MB | 2.3배 |
| SAM 1 ViT-H | 2446MB / 5731MB | 1223MB / 2845MB | 2.1배 |
| SAM 2.1 Hiera-L | 856MB / 1445MB | 428MB / 1001MB | 1.1배 |
| SAM 2.1 Hiera-S | 176MB / 618MB | 88MB / 435MB | 4배 |

CPU int8은 SAM 1 ViT-B 기준 파라미터 358MB → 18MB. 마스크 점수는 SAM 1/2/2.1
전부 fp32와 동일했고, MobileSAM만 fp16에서 품질이 떨어져(0.962 → 0.693)
기본값을 fp32/int8로 둡니다.

### ONNX 백엔드 — torch 없이 돌리기

`onnx-community` 의 SAM 2.1 변환본을 쓰면 **torch 설치 없이** 라벨링할 수 있습니다.
여기서는 정밀도별 파일이 실제로 배포되므로 파일을 골라 받습니다.

```bash
uv pip install onnxruntime
python -m dsl.sam.downloader --install sam2.1_tiny_onnx
```

CPU 실측(1280×720, 점 프롬프트): fp32 1.04s · int8 0.77s · q4 0.81s,
마스크 점수 0.994 / 0.990 / 0.949. int8이 품질 손실 거의 없이 가장 빠릅니다.

### SAM 3 / 3.1

SAM 3 는 **클릭·박스·텍스트** 프롬프트를 모두 지원합니다. 클릭은 내장 트래커가
처리하며, 비전 피처를 이미지당 한 번만 계산해 재사용합니다
(첫 클릭 1.76초 → 이후 0.06초).

두 저장소 모두 **승인이 필요한 게이트 저장소**라 로그인이 필요합니다:

```bash
# huggingface.co/facebook/sam3 에서 접근 승인을 먼저 받으세요
python scripts/download_sam3.py        # 토큰을 숨김 입력으로 물어봅니다
```

> **SAM 3.1 은 아직 불러올 수 없습니다.** 저장소에 Meta 원본 체크포인트
> (`sam3.1_multiplex.pt`)만 있고 transformers 가 읽는 `model.safetensors` 가
> 없습니다. 변환본이 공개되기 전까지는 SAM 3 를 쓰세요.

> 저장소 ID가 바뀌어 404가 나면 모델 관리자의 **[URL/저장소 수정]** 으로 고칠 수
> 있습니다 (`weights/catalog_overrides.json` 에 저장되며 코드를 건드리지 않습니다).

---

## 기존 라벨 재활용: 박스 → 세그먼트

이미 만들어 둔 검출/폴리곤 라벨을 가져와 세그먼트로 승격시킬 수 있습니다.

지원: **YOLO det/seg txt · COCO json · Pascal VOC xml · LabelMe json**

1. `Ctrl+I` → 라벨 폴더/파일 선택 (포맷 자동 감지)
2. 가져온 박스는 캔버스에 **점선**으로 표시됩니다
3. `Ctrl+5` **박스→세그먼트** 도구로 박스를 **클릭** → SAM이 마스크로 변환
4. 또는 SAM 패널의 **[현재 이미지 박스 전체 변환]** / **[프로젝트 전체 변환]**
5. 마음에 안 들면 `Ctrl+Z`, 클릭 프롬프트로 보정

원본 박스 좌표는 `meta.src_bbox` 에 남아 언제든 되돌아볼 수 있습니다.

---

## 메모리: 보이는 만큼만 불러옵니다

수만 장 데이터셋을 전제로 설계했습니다.

- **파일 목록** — `QAbstractListModel` 가상화. 화면에 그려지는 행만 썸네일을
  요청하고, 디코딩은 워커 스레드가 `IMREAD_REDUCED` 로 처리합니다.
  썸네일은 개수 제한 LRU(기본 512개)에만 남습니다.
- **원본 이미지** — 메모리에는 **현재 + 앞뒤 1장**만. 바이트 예산 LRU(기본 1GB,
  `설정 → 이미지 캐시 크기`)를 넘으면 오래된 것부터 즉시 해제합니다.
- **SAM 임베딩** — 이미지당 수백 MB이므로 **항상 1장분만**. 이미지 전환 시
  `reset_image()` + `torch.cuda.empty_cache()`.
- **완료 처리** — 저장/검증하면 그 이미지는 캐시에서 즉시 방출되고,
  `미라벨만` 필터에서는 목록에서 내려갑니다. `저장 후 다음 미라벨로 이동` 옵션도 있습니다.
- **라벨/통계** — 전체 라벨을 올리지 않고, 저장 시 갱신되는 `labels/_index.json`
  으로 통계를 계산합니다.

상태바에 `캐시 2장 / 5MB (예산 1024MB) · 썸네일 128` 처럼 실시간으로 표시됩니다.

---

## 학습

**학습 탭**에서 데이터셋 내보내기 → 모델/하이퍼파라미터 선택 → 시작.
학습은 **별도 프로세스**로 돌아 GUI가 멈추지 않고, 죽어도 라벨링 세션은 살아남습니다.

| 모델 | 과제 | 데이터셋 포맷 |
|---|---|---|
| YOLO26-seg / YOLO11-seg / YOLOv9-seg / YOLOv8-seg | 인스턴스 | YOLO |
| YOLO12 | 검출 (세그 체크포인트 미공개) | YOLO |
| RT-DETR (Baidu) | 검출 | YOLO |
| Mask R-CNN (torchvision) | 인스턴스 | COCO |
| SegFormer (MiT-b0~b5) | 시맨틱 | 마스크 PNG |
| ConvNeXt + UPerNet | 시맨틱 | 마스크 PNG |

결과는 `runs/<모델키>/<실행이름>/` 에 저장됩니다. 모델별로 묶이므로 같은 모델의
여러 학습을 나란히 비교할 수 있습니다.

```
runs/yolo11-seg/yolo11-seg_20260906_101530/
  job.json          이 학습의 설정 (그대로 다시 돌릴 수 있음)
  weights/best.pt   가장 좋은 체크포인트
  weights/last.pt
```

CLI로도 돌릴 수 있습니다:

```bash
python -m dsl.train.cli runs/yolo11-seg/yolo11-seg_20260906_101530/job.json
```

---

## 라벨 저장 형식 (단일 원본)

`labels/<이미지와 같은 상대경로>.json` (확장자 포함 — `a/b.jpg` → `labels/a/b.jpg.json`)

```json
{"version": 1, "image_path": "a/b.jpg", "width": 1920, "height": 1080,
 "verified": false,
 "shapes": [{"class_id": 0, "points": [[x, y], ...], "holes": [],
             "shape_type": "polygon", "score": 0.97, "source": "sam2.1_hiera_small",
             "group_id": null, "uid": "3f2a…", "meta": {}}]}
```

좌표는 **원본 이미지 픽셀 절대좌표**입니다. YOLO/COCO/PNG는 전부 여기서 파생되며,
역방향으로 덮어쓰지 않습니다.

## 폴더 구조

```
dsl/core/    프로젝트·라벨 모델, 캐시, 가져오기/내보내기
dsl/sam/     체크포인트 카탈로그, 다운로더, SAM 1/2/3 백엔드
dsl/train/   모델 레지스트리, 잡, subprocess 러너, 트레이너
dsl/ui/      캔버스, 패널, 다이얼로그, 메인 윈도우

weights/                     내려받은 체크포인트 (프레임워크별로 분리)
  sam/sam1|sam2|sam2.1|sam3|sam3.1/    SAM 체크포인트
  sam/onnx/                            ONNX 변환본 (torch 불필요)
  sam/quantized/                       양자화본
  yolo/pretrained/                     YOLO·RT-DETR 사전학습 .pt
datasets/                    내보낸 데이터셋
runs/<모델키>/<실행이름>/     학습 결과 (모델별로 묶임)
```

`weights/` `datasets/` `runs/` `configs/` 는 모두 **실행파일(또는 저장소 루트)과
같은 위치**에 만들어집니다. 현재 작업 디렉터리와 무관합니다.

## 아이콘

`assets/icon.svg` 가 원본입니다. 고친 뒤 다시 만들면 됩니다:

```bash
uv run --no-project --python .venv/Scripts/python.exe scripts/make_icon.py
```

PNG 7종(16~256px)과 `assets/icon.ico` 가 생성됩니다. PNG 는 git 에서 제외되고,
`.ico` 는 빌드에 필요해 커밋합니다. **PNG/ICO 를 직접 편집하지 마세요** —
다음 재생성 때 사라집니다.

## 실행파일 빌드 (Windows)

```bash
uv pip install pyinstaller
uv run --no-project --python .venv/Scripts/python.exe scripts/build_exe.py
```

`dist/DataSegmentLabeling/` 에 **약 300MB** 짜리 폴더가 나옵니다. 이 폴더째
복사하면 다른 PC에서 그대로 실행됩니다.

```
dist/DataSegmentLabeling/
  DataSegmentLabeling.exe
  README.txt                  외부 설치 안내 (빌드 시 생성)
  _internal/                  PySide6 · numpy · cv2 · onnxruntime
    dsl_src/dsl/              학습용 .py 소스 (외부 파이썬이 임포트)
  weights/ configs/ datasets/ runs/     빈 폴더로 생성
```

빌드가 잘 됐는지는 이걸로 확인합니다 (`dist/` 가 없으면 그냥 넘어갑니다):

```bash
uv run --no-project --python .venv/Scripts/python.exe tests/smoke_frozen.py
```

`build/` 와 `dist/` 는 git 에서 제외돼 있고 **언제든 지워도 됩니다** — 위
명령으로 다시 만들어집니다. 처음부터 다시 빌드하려면 `--clean` 을 붙이세요.

**무거운 패키지는 일부러 넣지 않습니다.** torch 하나가 약 4GB라 프로그램보다
훨씬 커지고, 지연 임포트라 대부분의 세션은 건드리지도 않습니다. 대신
`onnxruntime` 은 포함되어 있어 **exe만으로 SAM 세그먼트가 됩니다**.

| | 상태 |
|---|---|
| 라벨링 · 가져오기/내보내기 | 바로 됨 |
| SAM 세그먼트 (ONNX) | 바로 됨 |
| GPU 가속 · SAM 1/2/3 · 학습 | `torch` 등을 외부에서 설치 |

설치 상태와 명령은 앱의 **도움말 → 선택 패키지 설치 상태** 또는
`DataSegmentLabeling.exe --deps` 에서 확인합니다.

`weights/` `configs/` `datasets/` `runs/` 는 **실행파일과 같은 폴더**에
만들어집니다. 폴더째 옮겨도 그대로 동작합니다 (`--paths` 로 확인).

### 실행파일에서 학습하기

실행파일은 학습을 직접 돌리지 못합니다(torch 를 넣지 않았으니까요). 대신
**torch 가 설치된 파이썬을 지정**하면 그대로 학습됩니다:

1. 학습용 환경을 준비합니다.
   ```bash
   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
   uv pip install ultralytics
   ```
2. 앱에서 **설정 → 학습용 파이썬 경로…** 로 그 `python.exe` 를 지정합니다.

학습에 필요한 소스는 실행파일과 함께 배포되므로(`_internal/dsl_src/`) 별도로
저장소를 받을 필요는 없습니다. 지정하지 않고 학습을 누르면 무엇을 설치하고
어디를 지정해야 하는지 안내가 뜹니다.

## 문서

- [사용 설명서](docs/USER_GUIDE.md) — 라벨링 절차와 단축키
- [구현 계획서](docs/IMPLEMENTATION_PLAN.md) — 설계 근거, 진행 상황, 남은 작업
- [개발 지침](docs/DEVELOPMENT.md) — 기여 시 반드시 먼저 읽어야 할 규칙
- [변경 이력](CHANGELOG.md) — 무엇이 왜 바뀌었는지
