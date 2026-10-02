# 구현 계획서 (Implementation Plan)

작성일 2026-09-06 · 대상 저장소 `d:/python_workspace/DataSegmentLabeling`

---

## 1. 목표와 범위

| 구분 | 내용 |
|---|---|
| 문제 | 세그멘테이션 학습용 라벨을 손으로 그리는 비용이 너무 큼 |
| 해결 | SAM(1/2/2.1/3/3.1)으로 클릭 한 번에 마스크 생성 → 클래스 부여 → 그대로 학습 |
| 부가 | 이미 있는 **박스/폴리곤 라벨을 세그먼트로 승격** |
| 제약 | 수만 장 데이터셋에서도 RAM이 터지지 않을 것 |
| 비목표 | 웹/서버 배포, 다중 사용자 협업, 영상 트래킹 라벨링(추후) |

## 2. 기술 선택과 근거

### 2.1 GUI: PySide6

`QGraphicsView` 는 뷰 변환(줌/팬), BSP 기반 아이템 인덱싱, 아이템 단위 선택/드래그를
프레임워크 차원에서 제공한다. 라벨링 캔버스는 "수천 개의 정점을 가진 벡터 씬"이므로
이 구조가 그대로 요구사항이 된다. CustomTkinter의 Tk `Canvas` 는 뷰 변환이 없어
좌표를 매번 다시 계산해야 하고, 아이템 수가 늘면 히트테스트가 선형으로 느려진다.
추가로 Qt는 `QThread`/시그널로 SAM 추론과 다운로드를 안전하게 분리할 수 있고,
`QDockWidget` 로 패널 레이아웃을 사용자가 재배치할 수 있다.
라이선스는 LGPLv3 — 동적 링크 사용이면 상용 포함 무료.

### 2.2 라벨 저장: 이미지 1장 = JSON 1개

전체를 한 파일에 담으면 수만 장에서 로드/저장이 모두 무거워진다. 파일을 쪼개면
부분 로딩·부분 저장이 자연스럽고 diff도 읽힌다. 통계만 `labels/_index.json`
으로 따로 캐시해 매번 전체 스캔하지 않는다.

### 2.3 학습: 별도 subprocess

GUI 프로세스에서 학습을 돌리면 (a) UI가 멈추고 (b) CUDA OOM이 앱 전체를 죽이며
(c) 프레임워크가 심는 전역 상태(로깅, 시그널 핸들러)가 GUI와 충돌한다.
`python -m dsl.train.cli job.json` 으로 분리하고 stdout만 읽어 로그 패널에 흘린다.
트레이너는 epoch마다 `@@DSL_METRIC@@ {json}` 한 줄을 찍고 GUI가 이를 파싱한다.

### 2.4 SAM 백엔드 추상화

버전마다 API가 전혀 다르다(SAM1 `SamPredictor`, SAM2 `SAM2ImagePredictor`,
SAM3 transformers, Ultralytics 패키지형). `SamBackend` 하나로 감싸 UI는
`set_image / predict / predict_text` 만 안다. SAM 4가 나와도 클래스 하나 +
카탈로그 한 줄이면 끝난다.

### 2.5 체크포인트: 번들하지 않고 그때그때 다운로드

가중치 합계가 10GB를 넘고, 사용자의 VRAM에 따라 필요한 모델이 다르다.
`probe_hardware()` 로 GPU/VRAM/RAM/디스크를 조사해 순위를 매기고,
필요한 것만 `weights/<family>/` 로 이어받기 다운로드한다.
**양자화는 파일이 아니라 로드 시점 옵션** — CUDA는 fp16/bf16 autocast,
CPU는 `quantize_dynamic` int8. 파일 종류를 늘리지 않고 어떤 PC에도 맞춘다.

## 3. 아키텍처

```
                    ┌──────────────── GUI 스레드 ────────────────┐
 파일목록(가상화) ── │ MainWindow ── CanvasView(QGraphicsView)    │
 클래스/도형 패널 ── │     │            └ ShapeItem / MaskPreview │
 SAM/학습 패널  ──  │     │                                       │
                    └─────┼───────────────────────────────────────┘
                          │ Signal (Queued)
        ┌─────────────────┴──────────────┬───────────────────────┐
        ▼                                ▼                       ▼
  SamWorker(QThread)             ThumbnailPool(QThreadPool)  TrainingProcess
  ├ load / set_image             └ IMREAD_REDUCED 디코딩       └ subprocess
  ├ predict / predict_text                                       dsl.train.cli
  └ batch_convert (박스→세그먼트)                                  stdout 파이프
        │
        └ SamBackend ── Sam1 / Sam2 / Ultralytics / Sam3
```

데이터 흐름은 한 방향이다.

```
이미지 클릭 → 프롬프트 → SamWorker → 마스크 → mask_to_polygons → Shape
   → ImageAnnotation(JSON) → exporters → 학습 데이터셋 → TrainJob → runs/
```

## 4. 메모리 설계 (핵심 제약)

| 대상 | 정책 | 구현 |
|---|---|---|
| 파일 목록 | 행 위젯 미생성, 보이는 행만 | `QAbstractListModel` + `uniformItemSizes` |
| 썸네일 | 개수 LRU(512) + 축소 디코딩 | `LRUCountCache`, `ThumbnailPool` |
| 원본 이미지 | 현재 ±1장, 바이트 예산 LRU | `LRUImageCache.keep_only()` |
| 프리페치 | 연속 이동 시 1회로 합침 | 단일 `QTimer` 재시작 |
| SAM 임베딩 | 항상 1장분 | `reset_image()` + `empty_cache()` |
| 완료 이미지 | 즉시 방출 + 목록에서 내림 | `cache.evict()`, `미라벨만` 필터 |
| 라벨/통계 | 전량 로드 금지 | `labels/_index.json` 증분 갱신 |
| 학습 로그 | 5000줄 링버퍼 | `QPlainTextEdit.maximumBlockCount` |
| 일괄 변환 | 이미지 1장씩 로드/해제 | `SamWorker.batch_convert` |

검증: 1280×720 이미지 12장을 빠르게 넘겨도 캐시는 **2장 / 5MB** 로 유지됨
(`prefetch_neighbors=1`, 예산 64MB 테스트).

## 5. 모듈 구성

| 모듈 | 책임 |
|---|---|
| `core/annotation.py` | `ClassDef` / `Shape` / `ImageAnnotation` 직렬화 |
| `core/project.py` | 프로젝트, 이미지 스캔, 라벨 I/O, 통계 인덱스 |
| `core/mask_utils.py` | 마스크↔폴리곤, 시맨틱 라벨맵 |
| `core/image_cache.py` | 유니코드 경로 I/O, 바이트/개수 LRU |
| `core/importers.py` | YOLO / COCO / VOC / LabelMe 가져오기 |
| `core/exporters.py` | YOLO-seg / COCO / 마스크 PNG 내보내기, 결정적 분할 |
| `core/settings.py` | 설정 영속화, 장치 선택 |
| `sam/catalog.py` | 체크포인트 카탈로그(계열·크기·VRAM·정밀도·URL) |
| `sam/downloader.py` | 하드웨어 조사, 추천, 이어받기 다운로드, CLI |
| `sam/base.py` | `SamBackend` 인터페이스, autocast/양자화 헬퍼 |
| `sam/backends.py` | SAM1 / SAM2 / Ultralytics / SAM3 구현 |
| `train/registry.py` | 학습 가능 모델 표(9종) |
| `train/jobs.py` | 잡 정의 JSON |
| `train/runner.py` | subprocess 실행·스트리밍·중지 |
| `train/cli.py` | 학습 프로세스 진입점 |
| `train/trainer_*.py` | ultralytics / maskrcnn / hfseg 트레이너 |
| `ui/*` | 캔버스, 패널, 다이얼로그, 메인 윈도우, 워커 |

## 6. 진행 상황

### 완료

- [x] 개발 지침(DEVELOPMENT.md) 확정 — 메모리 규칙 포함
- [x] 라벨 데이터 모델 + 프로젝트/인덱스
- [x] 마스크↔폴리곤 변환(구멍 지원), 결정적 train/val/test 분할
- [x] 가져오기 4종(YOLO det·seg / COCO / VOC / LabelMe)
- [x] 내보내기 3종(YOLO-seg / COCO / 시맨틱 마스크)
- [x] SAM 카탈로그 + 하드웨어 추천 + 이어받기 다운로더 + CLI
- [x] SAM 백엔드 4종(SAM1 / SAM2·2.1 / Ultralytics / SAM3·3.1)
- [x] 캔버스(줌·팬·정점편집·프롬프트·후보전환), 다크 테마
- [x] 가상화 파일 목록 + 백그라운드 썸네일 + LRU 캐시
- [x] 클래스/도형/SAM/학습 패널, 모델 관리자, 가져오기·내보내기 다이얼로그
- [x] 박스→세그먼트 단건 클릭 변환 + 이미지/프로젝트 일괄 변환
- [x] 학습 레지스트리 9종 + subprocess 러너 + 트레이너 3종
- [x] 스모크 테스트(데이터 파이프라인 / 오프스크린 GUI) 통과
- [x] README · 사용 설명서 · 구현 계획서

### 실기기 검증 결과 (2026-09-07, RTX 4060 8GB / torch 2.14.0+cu126)

| 항목 | 상태 | 비고 |
|---|---|---|
| SAM 1/2/2.1 추론 | ✅ **검증 완료** | 14종 × 정밀도 × 장치 = **45개 조합 전부 성공** |
| 로드 시점 양자화 | ✅ **검증 완료** | fp16이 가중치·피크 VRAM을 정확히 절반으로. 마스크 점수 동일 |
| SAM 3 추론 | ✅ **검증 완료** | 클릭·박스·텍스트 모두 동작. fp32/fp16/bf16/CPU 4개 조합 성공 |
| SAM 3.1 | ⛔ **불가(외부 요인)** | 저장소에 transformers 형식 가중치가 없음. 변환본 공개 대기 |
| ONNX 백엔드 | ✅ **신규 추가** | torch 없이 동작. fp32/int8/q4 검증 |
| 학습 9종 | ✅ **검증 완료** | 전 모델 1 epoch 관통, 메트릭 파싱·summary.json·best 저장까지 확인 |

검증 중 발견해 고친 실제 버그:

- `trainer_ultralytics`: `loss_items` 가 dict인 ultralytics 버전에서 콜백이
  터져 **학습 전체가 실패**하던 문제 (`AttributeError: 'dict' has no 'tolist'`)
- `trainer_maskrcnn`: 검증 패스가 **BatchNorm 69개의 통계를 매 epoch 오염**
  (v2 모델은 `FrozenBatchNorm2d` 가 아님) + 빈 타깃에 가짜 인스턴스 주입
- `trainer_hfseg`: 검증 분할이 없고 손실이 1.0을 넘으면 `best/` 가 저장되지
  않는데 요약은 그 경로를 가리키던 문제
- `registry`: YOLOv12 세그먼트 체크포인트는 **존재하지 않음**(ultralytics
  v8.4.0 자산 확인). 이름도 `yolo12` → 검출 전용 항목으로 정정
- `Sam3Backend`: 클릭이 `input_points` 미지원 경로로 가 **모든 클릭이 실패**.
  트래커 서브모델 경로로 재구현 (첫 클릭 1.76s → 이후 0.06s)

### 다음 단계 (우선순위)

1. **실모델 검증** — SAM 2.1 Hiera-S 다운로드 → 클릭 라벨링 → 박스 일괄 변환 →
   YOLO11-seg 1 epoch 학습까지 한 번 관통.
2. **학습된 모델로 사전 라벨링(auto-label)** — 학습 결과를 다시 불러와 미라벨
   이미지에 예측을 채우고 사람이 검수하는 루프. 라벨링 속도가 가장 크게 오르는 지점.
3. **SAM 3 트래커로 연속 프레임 전파** — 촬영 시퀀스에서 같은 객체 ID 유지.
4. **품질 점검 패널** — 클래스 불균형, 아주 작은/겹치는 도형, 미검증 목록.
5. **다중 이미지 선택 일괄 작업** — 클래스 일괄 변경, 선택 항목만 내보내기.
6. **단위 테스트 정식화** — `tests/smoke_*.py` 를 pytest 케이스로 분해하고 CI 연결.

## 7. 위험 요소와 대응

| 위험 | 영향 | 대응 |
|---|---|---|
| SAM 3/3.1 저장소·API 변경 | 로딩 실패 | 클래스명 다중 시도 + 사용자 오버라이드 + 명확한 오류 안내 |
| ultralytics 버전이 YOLO26 미지원 | 학습 실패 | 실패 시 `pip install -U ultralytics` 안내 메시지 |
| 초대형 이미지(1억 화소) | 인코딩 지연/OOM | `sam_max_side` 로 SAM 입력만 축소, 마스크는 원본 해상도로 복원 |
| Windows 심볼릭 링크 권한 | 내보내기 실패 | 실패 시 자동으로 복사로 폴백 |
| 라벨 파일 손상 | 이미지 1장 손실 | `.tmp` → `replace` 원자적 저장, 손상 시 빈 주석으로 복구 |
| GPU 공유(라벨링+학습 동시) | OOM | 학습은 별도 프로세스라 격리, VRAM 부족 시 batch 축소 안내 |

## 8. 검증 방법

```bash
# 데이터 파이프라인: 프로젝트 생성 → 라벨 → 3종 내보내기 → 4종 가져오기
python tests/smoke_pipeline.py

# GUI: 오프스크린으로 창을 만들고 실제로 조작
python tests/smoke_gui.py        # QT_QPA_PLATFORM=offscreen 자동 설정

# 다운로더
python -m dsl.sam.downloader --probe --list --recommend

# 학습 CLI (패키지 없을 때 안내 메시지 확인)
python -m dsl.train.cli runs/<모델키>/<실행이름>/job.json
```

현재 두 스모크 모두 통과하며, GUI 테스트는 캐시 상한(≤3장) 유지까지 검사한다.
