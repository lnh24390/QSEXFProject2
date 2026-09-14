# 모델 공통 분석

초기 학습, 전이학습, 외부 Ultralytics `.pt` 모델을 같은 평가 코드로 분석합니다. 검출과 분할은 서로 다른 과제로 취급합니다. 모델을 다시 학습하지 않습니다.

## 폴더

| 경로 | 용도 |
|---|---|
| `02_training/model_analysis` | 공통 평가·비교·과거 결과 가져오기 코드, 모델 목록 |
| `02_training/external_models` | 외부 가중치를 직접 보관할 때 사용할 위치. 외부 절대 경로도 지원 |
| `04_results/model_analysis/evaluations/<bbox 또는 seg>/<출처>/<실행 ID>` | 새로 측정한 결과 |
| `04_results/model_analysis/comparisons/<실행 ID>` | 같은 조건의 모델 비교 |
| `04_results/model_analysis/historical/<실행 ID>` | 기존 측정값의 표준화 사본 및 과거 학습 요약 |
| `04_results/model_analysis/training/<프로젝트>/<실행 ID>` | 이후 완료되는 학습의 CSV 기반 요약 |

실행 ID에는 시각·모델 이름·임의 ID가 들어갑니다. 같은 이름의 `best.pt`를 여러 번 평가해도 결과를 덮어쓰지 않습니다. 기존 가중치·실험 원본은 프로젝트별 `04_results` 폴더에 유지합니다.

## 실행

아래 명령은 프로젝트 루트에서 실행합니다. 모델 ID와 기본 평가 데이터는 `models.json`에서 관리합니다.

```powershell
uv run --frozen python 02_training/model_analysis/analyze.py list

# GPU를 쓰지 않고 기존 기록 정리
uv run --frozen python 02_training/model_analysis/analyze.py import-history

# 모델·데이터·클래스 연결만 확인
uv run --frozen python 02_training/model_analysis/analyze.py evaluate --task bbox --models bbox_original bbox_first bbox_final --check-only

# 검출 3개를 순차 평가하고 비교표 생성
uv run --frozen python 02_training/model_analysis/analyze.py evaluate --task bbox --models bbox_original bbox_first bbox_final

# 분할 3개를 순차 평가하고 비교표 생성
uv run --frozen python 02_training/model_analysis/analyze.py evaluate --task seg --models seg_original seg_first seg_final

# 외부 파일과 기존 모델을 함께 평가. 예시 경로를 실제 파일로 바꾸세요.
uv run --frozen python 02_training/model_analysis/analyze.py evaluate --task bbox --models bbox_original bbox_final "D:/models/custom.pt"
```

외부 모델은 현재 Ultralytics 환경에서 불러올 수 있는 검출·분할 모델이어야 합니다. 일반 PyTorch `state_dict`만 들어 있는 파일, 별도 사용자 정의 구조가 필요한 파일은 구조를 연결하는 추가 작업이 필요합니다. 임의의 `.pt` 모두를 지원한다는 뜻은 아닙니다.

기본값은 `imgsz=640`, batch 16, FP32, conf 0.001, NMS IoU 0.7, max_det 300입니다. 검출과 분할은 각자의 고정 test 목록을 씁니다. `--split val`로 검증 분할을 선택하거나 `--data "D:/evaluation/data.yaml"`로 별도 평가 세트를 지정할 수 있습니다. **평가 데이터에는 정답 라벨이 필요합니다.** 배경도 빈 `.txt`를 둡니다.

클래스 이름이 같고 번호만 다르면 예측 번호를 평가 데이터의 번호에 자동 대응합니다. 모델 가중치와 정답 라벨은 바꾸지 않습니다. 이름이 다른 경우 `--class-map "D:/models/class_map.json"`을 지정하세요. 예: `{"0":"paper","1":"plastic","2":"can","3":"glass","4":"battery","5":"vinyl","6":"general"}`. 파일에는 외부 모델 클래스 전체에 대한 정확한 1:1 대응이 필요합니다. COCO 80클래스를 쓰레기 7클래스와 자동으로 같다고 간주하지 않습니다.

## 결과와 비교 기준

각 평가에는 `result.json`, `metrics.csv`, `summary.md`, 실제 이미지·라벨 해시가 담긴 `dataset_manifest.json`이 생깁니다. 박스·마스크, 전체·클래스별 precision/recall/mAP50/mAP50–95를 구분하며 미측정 값은 null/빈 칸입니다. AP가 없는 클래스에 전체 평균을 채워 넣지 않습니다. YAML과 라벨 캐시는 해당 결과 폴더에 저장합니다.

기존 `result.json`끼리 비교하려면:

```powershell
uv run --frozen python 02_training/model_analysis/analyze.py compare --records "D:/results/first/result.json" "D:/results/second/result.json"
```

과제, 실제 이미지·라벨 내용, 분할, 평가 조건 및 소프트웨어 버전이 같아야 비교됩니다. 과거 기록은 당시 조건·데이터 해시가 완전하지 않아 새 측정 결과와 자동으로 섞지 않습니다. 기존 최종 비교 보고서 안에서 함께 측정된 모델만 과거 비교표로 정리합니다.

외부 모델의 학습 이력을 알 수 없으므로 평가 데이터가 학습에 사용되지 않았다는 보장은 없습니다. 기존 테스트 성능으로 수건·모니터·코팅지 등의 새 현장 사례 개선을 단정하지 않습니다. 기록된 속도는 검증 중 시간이며 별도의 추론 벤치마크 순위가 아닙니다.

## 기존 코드와의 연결

`02_training/baseline_training/test.py`는 같은 평가 기능을 호출합니다. 기본 평가 세트는 배경 포함 공통 640 test로 통일했습니다. 이전 배경 제외 test가 필요하면 `--data`를 지정하거나 `compare_bg.py`를 사용하세요. `compare_bg.py`의 배경 포함/제외 평가는 조건별로 공통 기록을 남기며 전용 오탐 통계도 유지합니다.

`03_transfer_learning`의 세 프로젝트는 검증·최종 평가를 공통 평가 함수로 수행하고, 단계 선택에 필요한 기존 보고서도 계속 남깁니다. 학습 내부 매 에포크 검증은 Ultralytics가 그대로 수행합니다. 완료된 학습을 이 변경 때문에 다시 실행하지 않습니다.

Mosaic·해상도·TensorRT 실험은 목적과 조건이 다른 별도 벤치마크로 유지합니다. 공통 정확도 비교표에 섞지 않습니다. 공통 경로는 루트 `workspace_layout.json`의 `model_analysis_code`, `model_analysis_results`, `external_models`에서 지정합니다.

검증 명령:

```powershell
uv run --frozen python -m unittest discover -s 02_training/model_analysis/tests -v
```

구현 근거: [Ultralytics 평가 문서](https://docs.ultralytics.com/modes/val/) 및 현재 설치된 8.4.146 소스. 라이브러리 업그레이드 후에는 위 테스트와 소량 실제 평가를 먼저 확인하세요.
