# 최적 설정 학습

검출(bbox)과 세그멘테이션(seg) 데이터셋을 실측으로 정한 최적 설정으로 학습합니다.
설정 근거는 [../work/BENCH_REPORT.md](../../04_results/dataset_preparation/BENCH_REPORT.md), 작업 원칙은 [../PROJECT_GUIDE.md](../../PROJECT_GUIDE.md) 를 보세요.

## 파일

| 파일 | 역할 |
|---|---|
| `train.py` | 학습, 이어서 학습(resume) |
| `test.py` | test 분할 최종 평가, 결과를 `test_result.txt` 로 저장 |
| `metrics_logger.py` | 학습 중 수치를 텍스트로 기록 (`train.py` 가 자동으로 붙임) |
| `merge_resume_logs.py` | 이어서 학습해 끊긴 시간 기록을 하나로 합침 |
| `wait_and_merge.ps1` | 세그 학습이 끝나면 `merge_resume_logs.py` 를 자동 실행 |
| `pipeline_after_seg.ps1` | 세그 학습이 끝나면 배경 제외 학습(yolo26s) → 비교를 순서대로 자동 실행 |
| `tensorrt_bench.py` | (사용 안 함) 예측 속도가 필요할 때만: TensorRT 엔진 변환 후 PyTorch FP32 / FP16 / TensorRT FP16 비교 |
| `data_nobg/` | 배경 이미지를 뺀 이미지 목록(`*_train/val/test.txt`)과 yaml |
| `compare_bg.py` | 배경 포함/제외 학습 모델을 같은 test 목록으로 평가해 한 표로 비교 |
## 실행

`bbox_data` 폴더에서 실행합니다.

    uv run python 02_training/baseline_training/train.py              # 검출 학습
    uv run python 02_training/baseline_training/train.py seg          # 세그멘테이션 학습
    uv run python 02_training/baseline_training/train.py both         # 검출 → 세그 순서로 연속 학습
    uv run python 02_training/baseline_training/train.py nobg         # 배경 제외 검출 → 세그 (비교용)
    uv run python 02_training/baseline_training/train.py bbox_nobg    # 배경 제외 검출만 (seg_nobg 도 가능)

    uv run python 02_training/baseline_training/train.py seg resume   # 멈춘 학습을 last.pt 부터 이어서

    uv run python 02_training/baseline_training/test.py               # 검출 test 평가
    uv run python 02_training/baseline_training/test.py seg           # 세그 test 평가

    uv run python 02_training/baseline_training/compare_bg.py         # 배경 포함/제외 네 모델 비교 (학습이 모두 끝난 뒤)

### 오래 걸리는 학습은 창과 분리해서 실행

학습은 수십 시간 걸립니다. 터미널이나 작업 세션 안에서 돌리면 창·세션을 닫을 때 **학습도 같이 멈춥니다.**
PowerShell 에서 아래처럼 실행하면 창을 닫아도 계속됩니다. (PC 절전·재부팅 시에는 멈춥니다. 절전을 꺼두세요.)

    $env:PYTHONIOENCODING = 'utf-8'
    Start-Process uv -ArgumentList 'run','python','best_train/train.py','seg' `
      -WorkingDirectory 'C:\Users\USER\Desktop\trash_detection_project' `
      -RedirectStandardOutput 'C:\Users\USER\Desktop\trash_detection_project\04_results\baseline_training\runs\seg_stdout.log' `
      -RedirectStandardError  'C:\Users\USER\Desktop\trash_detection_project\04_results\baseline_training\runs\seg_stderr.log' `
      -WindowStyle Hidden

진행 상황은 `04_results/baseline_training/runs/<폴더>/metrics.txt` 를 열어 보면 됩니다.

## 데이터셋

| 과제 | 모델 | 데이터 | train | val | test |
|---|---|---|---:|---:|---:|
| bbox | `02_training/pretrained_models/yolo11n.pt` | `01_datasets/original_bbox` | 49,874 | 9,628 | 2,734 |
| seg | `02_training/pretrained_models/yolo11n-seg.pt` | `01_datasets/original_seg` | 49,873 | 9,629 | 2,734 |

- 두 데이터셋 모두 클래스마다 train/val/test 를 약 **80 / 15 / 5** 로 맞췄습니다.
- 두 데이터셋 모두 **배경 이미지 3,000장**(4.8%)이 들어 있습니다. COCO 에서 7개 클래스와 헷갈릴 물체가 없는 사진(사람, 모니터, 의자, 책상, 침대 등)을 골랐고, 라벨은 빈 파일입니다. 7개 클래스가 없는 장면에서 **없는 물체를 있다고 하는 오탐**을 줄이는 목적입니다. 파일 이름이 `bg_coco_` 로 시작합니다.
- **클래스 ID 순서가 두 데이터셋에서 다릅니다** (vinyl=5 만 같음). 가중치와 결과를 섞지 마세요.
- **두 데이터셋의 분할이 같지 않습니다.** 5,786장이 서로 다른 분할에 있습니다(배경 이미지는 모두 같음). 검출과 세그의 val·test 결과는 같은 이미지로 잰 값이 아니며, 한쪽 모델을 다른 쪽 test 로 평가하면 학습 이미지가 섞입니다.

### 배경 제외 비교 학습 (nobg)

**배경 이미지를 뺀 라벨 데이터만** 학습한 결과도 만듭니다. 요청에 따라 모델은 **yolo26s / yolo26s-seg(소형)** 입니다. (2026-09-11 에 yolo26l → yolo26m → yolo26s 순으로 낮춤, 속도 우선 결정)

| 과제 | 데이터 | train | val | test | 결과 폴더 |
|---|---|---:|---:|---:|---|
| bbox_nobg (yolo26s) | `data_nobg/bbox_nobg.yaml` | 47,474 | 9,178 | 2,584 | `04_results/legacy_root/runs/bbox_yolo26s_nobg` |
| seg_nobg (yolo26s-seg) | `data_nobg/seg_nobg.yaml` | 47,473 | 9,179 | 2,584 | `04_results/legacy_root/runs/seg_yolo26s_nobg` |

- 파일을 옮기지 않고, `bg_coco_*` 를 뺀 **이미지 목록 파일**로 고릅니다. train·val·test 모두에서 뺐습니다.
- **모델이 배경 포함 학습(yolo11n)과 다릅니다.** 그래서 두 결과의 차이에는 배경 효과와 모델 크기 효과가 섞여 있습니다. 배경만의 효과를 알려면 같은 모델로 배경을 넣고/빼고 학습해야 합니다.
- 모델 크기(공식 표, COCO 기준): YOLO26s 9.5M 파라미터·20.9B FLOPs, YOLO26s-seg 10.4M·34.5B FLOPs. yolo11n 대비 연산량 약 3배이고, yolo26m(68.4B / 121.7B)의 약 1/3 입니다. COCO 성능은 yolo26m보다 검출 mAP 4.5, 세그 mask mAP 4.1 낮습니다(검출 48.6 vs 53.1, 세그 40.0 vs 44.1). 우리 데이터에서의 차이는 학습해 봐야 압니다. ([YOLO26](https://docs.ultralytics.com/models/yolo26/), [세그멘테이션](https://docs.ultralytics.com/tasks/segment/))
- **예상 학습 시간 (추정, 2026-09-11)**: RTX 4060 에서 YOLO26 학습 시간 실측 자료는 찾지 못해, 이 PC 의 실측값과 연산량으로 추정했습니다.

  | 모델 | 에포크당 | 100에포크 |
  |---|---|---|
  | yolo26s (검출) | 약 12~20분 | 약 0.8~1.4일 |
  | yolo26s-seg | 약 18~30분 | 약 1.25~2.1일 |
  | 합계 | | **약 2~3.5일** |

  공식 학습 가이드의 COCO 파인튜닝 에포크는 **N 245, S 70, M 80, L 60, X 40** 입니다([YOLO26 Training Recipe](https://docs.ultralytics.com/guides/yolo26-training-recipe/)). 큰 모델일수록 대체로 적은 에포크에서 수렴하지만 **단조롭게 줄지는 않습니다**(S 70 < M 80). 문서도 "크기별 하이퍼파라미터 탐색 결과라 엄밀히 단조롭지 않다"고 밝히고 있습니다. 우리가 쓰는 **S 는 70 에포크**입니다. `patience=10` 으로 조기 종료가 걸리면 더 짧아질 수 있습니다. 정확한 시간은 학습 시작 후 첫 에포크가 끝나면 `metrics.txt` 에서 확인하세요.
- batch 는 **16** 입니다(사용자 요청, 2026-09-11. 24 에서 메모리 여유를 위해 낮춤). 검증은 ultralytics 가 batch 의 2배(32)로 합니다.
  - yolo26s 는 yolo11n 보다 연산량이 약 3배라 8GB 에서 메모리가 모자랄 수 있습니다. **첫 에포크 학습 중** 메모리가 부족하면 ultralytics 가 batch 를 절반으로 줄여 최대 3번 다시 시도합니다(16→8→4→2). 실제로 쓴 batch 는 `config.txt` 에 기록됩니다.
  - **첫 에포크 이후나 검증 중** 메모리가 부족하면 학습이 멈춥니다. 그때는 `last.pt` 로 이어서 하되 batch 를 낮춰야 합니다.
- epochs 100, imgsz 640, workers 4, AMP, seed 0 은 배경 포함 학습과 같습니다. **patience 는 10** 입니다(배경 포함 학습은 30, 2026-09-12 사용자 요청). 처음에 patience 30 으로 시작한 yolo26s 검출 학습은 1에포크쯤에서 멈추고 처음부터 다시 시작했습니다.
- 목록 파일로 학습하면 데이터셋의 `labels/train.cache`, `labels/val.cache` 를 배경 포함 학습과 같이 씁니다. 파일 목록이 다르면 ultralytics 가 알아서 캐시를 다시 만들므로(30초 정도) 결과에는 영향이 없습니다.
- **학습 중 표시되는 val 성능은 두 학습을 그대로 비교할 수 없습니다.** 배경 포함 학습의 val 에는 배경 450장이 들어 있어 오탐이 성능에 반영되고, 배경 제외 학습의 val 에는 없습니다. 비교는 두 모델을 **같은 test 목록**(배경 포함 test, 배경 제외 test 각각)으로 평가해서 합니다.

#### 비교 스크립트 (`compare_bg.py`)

| 항목 | 내용 |
|---|---|
| 평가 조합 | 검출 모델(bbox, bbox_nobg)은 검출 test 로만, 세그 모델(seg, seg_nobg)은 세그 test 로만. 두 데이터셋은 분할이 달라 섞으면 학습 이미지가 평가에 들어갑니다 |
| 배경 포함 test | 검출·세그 각각 2,734장 (라벨 2,584 + 배경 150). 배경에서 오탐하면 P·mAP 가 깎임 |
| 배경 제외 test | 2,584장. 라벨된 물체를 찾는 성능만 봄 |
| 배경 사진 오탐 | test 배경 150장만 예측해, conf 0.25·0.5 기준으로 잘못 검출한 이미지 수·검출 수·많이 틀린 클래스 |
| 출력 | 전체 성능(세그는 박스·마스크), 배경 오탐, 클래스별 성능과 차이(배경 포함 학습 - 배경 제외 학습) |
| 결과 파일 | `04_results/legacy_root/runs/compare_bg/compare_result.txt`, `compare_result.csv` |

- 학습 프로세스가 돌고 있으면 GPU 평가를 시작하지 않고 멈춥니다(`--force` 로 무시 가능).
- 동작만 확인할 때: `--device cpu --limit 20` (test 에서 20장만 사용)
- 가중치가 아직 없는 모델은 건너뛰고 표에서 뺍니다.
- 표에는 모델 이름이 함께 나옵니다(`yolo11n 배경 포함`, `yolo26s 배경 제외`). 차이에 모델 크기 효과가 섞여 있다는 안내도 결과 파일에 들어갑니다.
- test 는 최종 평가용입니다. 이 결과를 보고 설정을 다시 고르면 test 가 오염됩니다.

## 설정

`train.py` 위쪽 '설정' 부분만 고치면 됩니다. 과제별 모델·데이터·결과 폴더 이름은 `TASKS` 에 있습니다.

| 설정 | 값 | 이유 (실측) |
|---|---|---|
| 모델 | `yolo11n` / `yolo11n-seg` | 같은 batch 에서 YOLO26n 보다 13~27% 빠릅니다 |
| BATCH | 16 | 8→16 에서 15% 빨라지고 그 뒤로는 거의 같습니다. 32·48 은 더 느립니다 |
| EPOCHS | 100 | 최대치입니다 |
| PATIENCE | 10 | 조기 종료. 10에포크 동안 판단 기준(아래 '최고 성능 기준')이 나아지지 않으면 멈춥니다. 2026-09-12 사용자 요청으로 30 → 10 (이미 끝난 yolo11n 학습 2개는 30) |
| imgsz | 640 | 데이터 해상도. 낮추면 배터리·비닐 인식률이 떨어집니다 |
| workers | 4 | 4/8/12 차이 1.4% 이내. 병목이 GPU 연산이라 올려도 소용없습니다 |
| cache | 끔 | Windows 에서 RAM 캐시는 실패하고, 우회하면 2.3배 느립니다 |
| AMP | 켬 | 메모리·연산 부담을 줄입니다 |

### 자주 바꾸는 경우

- **정확도를 우선할 때**: `TASKS` 의 모델을 `yolo26n.pt` 로, `BATCH = 32`. 약 21% 느리지만 4에포크 시험에서 vinyl 재현율이 0.347 → 0.629 로 크게 앞섰습니다.
- **GPU 를 학습에만 쓸 때** (브라우저·Docker 등 모두 종료): `BATCH = 24`. 가장 빠르지만 메모리를 4.62GB 써서 여유가 줄어듭니다. 세그는 메모리를 더 쓰므로 16 을 권장합니다.
- **동작만 빠르게 확인할 때**: `EPOCHS = 3`, `TASKS["bbox"]["data"]` 를 `01_datasets/auxiliary/benchmark_subset/data.yaml` 로.

## 결과 파일

결과는 `04_results/baseline_training/runs/` 아래 과제별 폴더(`bbox_yolo11n`, `seg_yolo11n`, `bbox_yolo26s_nobg`, `seg_yolo26s_nobg`)에 저장됩니다.

| 파일 | 내용 |
|---|---|
| `config.txt` | 학습 환경·설정, 학습/검증 이미지 수와 배경 비중. 이어서 학습한 시각도 뒤에 추가됨 |
| `metrics.txt` | 에포크마다 한 줄. 학습 중에도 열어볼 수 있음 |
| `summary.txt` | 학습 종료 후 요약: 총 시간, 조기 종료 여부, 최고 성능 에포크, 클래스별 정밀도·재현율 (**박스 기준**) |
| `test_result.txt` | `test.py` 실행 결과. 세그는 마스크 mAP 도 기록 |
| `results.csv`, `results.png` | ultralytics 기본 기록과 학습 곡선. 세그는 마스크 지표 `metrics/*(M)` 열도 에포크마다 기록됨 |
| `02_training/pretrained_models/root_weights/best.pt` | 판단 기준(아래 '최고 성능 기준')이 가장 높았던 에포크의 가중치 |
| `02_training/pretrained_models/root_weights/last.pt` | 마지막 에포크 가중치. 이어서 학습할 때 씀 |
| `merge_log.txt` | 기록 통합을 실행했을 때만 생김 |
| `*_raw.*` | 기록 통합 전 원본 |

### metrics.txt 열

| 열 | 뜻 |
|---|---|
| ep | 에포크 |
| 시간 / 누적 | 그 에포크 소요 시간(검증 포함) / 누적 학습 시간 |
| box, cls, dfl | 학습 손실. 낮을수록 좋음 |
| P, R | 검증 정밀도, 재현율 (박스 기준) |
| mAP50, mAP50-95 | 검증 성능 (박스 기준) |
| GPU_GB | 최대 GPU 메모리 |
| lr | 학습률 |
| 비고 | `<- 최고`: 최고 기록 갱신, `[재개]`: 이어서 학습을 시작한 에포크(통합 후 표시) |

세그 학습도 `metrics.txt` 와 `summary.txt` 의 P/R/mAP 는 **박스 기준**입니다. 마스크 성능은 `results.csv` 의 `metrics/precision(M)`, `recall(M)`, `mAP50(M)`, `mAP50-95(M)` 열(에포크별)과 `test_result.txt`(최종)에서 확인하세요.

### 최고 성능 기준

조기 종료와 `best.pt` 선택은 ultralytics 가 아래 값으로 판단합니다 (설치된 8.4.146 코드로 확인).

| 과제 | 판단 기준 | `metrics.txt` 의 `<- 최고` 와 같은가 |
|---|---|---|
| 검출 | 박스 mAP50-95 | 같음 |
| 세그 | 박스 mAP50-95 **+ 마스크 mAP50-95** | **다를 수 있음** |

세그의 `<- 최고` 표시와 `summary.txt` 의 '최고 mAP50-95' 에포크는 박스만 보고 정한 값이라, `best.pt` 가 저장된 에포크와 다를 수 있습니다. 세그의 실제 최고 에포크는 `results.csv` 에서 `mAP50-95(B) + mAP50-95(M)` 이 가장 큰 행입니다. 같은 이유로 세그 `summary.txt` 의 '조기 종료됨 … mAP50-95 가 나아지지 않았습니다' 문구도 정확히는 두 값의 합을 뜻합니다.

정밀도가 낮으면 오탐이 많다는 뜻이고, 재현율이 낮으면 놓치는 물체가 많다는 뜻입니다.

## 학습이 멈췄을 때

1. 멈췄는지 확인합니다. `metrics.txt` 마지막 줄 시각이 에포크 시간(검출 약 8분, 세그 약 10~12분)보다 오래 지났고, `nvidia-smi` GPU 사용률이 낮으면 멈춘 것입니다.
2. 이어서 학습합니다: `uv run python 02_training/baseline_training/train.py seg resume` (위 `Start-Process` 방식 권장)
   - `last.pt` 에 에포크 번호·옵티마이저·학습률이 저장돼 있어 멈춘 다음 에포크부터 이어집니다.
   - 끝내지 못한 에포크의 진행분은 버려지고 그 에포크를 처음부터 다시 합니다.
   - `metrics.txt`, `config.txt` 는 지우지 않고 뒤에 이어 씁니다.
3. 학습이 끝나면 기록을 합칩니다.

       uv run python 02_training/baseline_training/merge_resume_logs.py seg_yolo11n

   이어서 학습하면 `results.csv` 의 time 열과 `metrics.txt` 누적 시간이 0부터 다시 시작하고, `summary.txt` 의 총 시간·에포크 평균도 재개 뒤 시간만 반영합니다. 이 스크립트가
   - 에포크 시간을 이어 붙여 누적 시간을 다시 계산하고 (멈춰 있던 시간은 제외)
   - `summary.txt` 에 총 학습 시간, 전체 경과 시간, 구간별 중단·재개 기록을 넣습니다.
   - 원본은 `*_raw` 로 한 번만 보관하므로 여러 번 실행해도 결과가 같습니다.

`wait_and_merge.ps1` 는 `summary.txt` 가 생기면 3번을 자동으로 실행합니다. 동작 기록은 `seg_yolo11n/merge_watch_log.txt` 에 남습니다. PC 를 재부팅하면 감시도 꺼지므로 그때는 직접 실행하세요.

### PC 가 재부팅됐을 때

PC 가 꺼지거나 재부팅되면 **학습, `wait_and_merge.ps1`, `pipeline_after_seg.ps1` 이 모두 꺼집니다.** (2026-09-11 에 비정상 종료가 10:05, 17:21 두 번 있었습니다.)

1. 원인 확인 (PowerShell): `Get-WinEvent -FilterHashtable @{LogName='System'; Id=41,6008} -MaxEvents 5`
2. 학습 이어서 시작 (위 `Start-Process` 방식, 인자 `'best_train/train.py','seg','resume'`)
3. 두 대기 스크립트 다시 실행

       Start-Process powershell.exe -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','C:\Users\USER\Desktop\trash_detection_project\02_training\baseline_training\wait_and_merge.ps1' -WindowStyle Hidden
       Start-Process powershell.exe -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','C:\Users\USER\Desktop\trash_detection_project\02_training\baseline_training\pipeline_after_seg.ps1' -WindowStyle Hidden

   yolo26s 학습 중에 멈췄다면 2번 인자를 `'bbox_nobg','resume'` 또는 `'seg_nobg','resume'` 로 바꿉니다. 이때 파이프라인은 다시 실행하지 말고, 학습이 끝난 뒤 남은 단계를 직접 실행하세요(파이프라인은 처음부터 `nobg` 전체를 새로 시작하기 때문).

## TensorRT (사용 안 함)

**2026-09-11 결정: 학습 속도에 영향이 없어 사용하지 않습니다.** 파이프라인에서도 뺐습니다.

근거(공식 문서):
- NVIDIA: "NVIDIA TensorRT is an SDK for high-performance deep learning **inference** on NVIDIA GPUs." — [github.com/NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT)
- Ultralytics: TensorRT 는 학습이 끝난 `.pt` 를 `.engine` 으로 바꿔 NVIDIA GPU 에서 빠르게 **예측**하는 배포용 형식이며, 학습 가속에 대한 내용은 없습니다. — [docs.ultralytics.com/integrations/tensorrt](https://docs.ultralytics.com/integrations/tensorrt/)

TensorRT 10.16.1.11 과 `tensorrt_bench.py` 는 남겨 두었습니다. 나중에 **서비스에서 예측 속도**가 필요할 때 아래 방법으로 쓰면 됩니다.

**TensorRT 는 학습이 아니라 예측(추론)을 빠르게 합니다.** 학습 시간은 줄지 않습니다. 학습이 끝난 `best.pt` 를 이 PC 의 GPU 에 맞춰 최적화한 `best.engine` 으로 바꾸고 FP16 으로 계산합니다.

`tensorrt_bench.py` 가 같은 test 분할을 이미지 1장씩(실제 서비스와 같은 조건) 예측해 세 가지를 비교합니다.

| 형식 | 파일 | 목적 |
|---|---|---|
| PyTorch FP32 | `best.pt` | 기준 |
| PyTorch FP16 | `best.pt` | 속도 향상 중 FP16 의 몫을 가려냄 |
| TensorRT FP16 | `best.engine` | 최종 가속 |

기록 항목: 파일 크기, box·mask mAP50-95, 전처리·추론·후처리 ms, FPS, 추론 배수(PyTorch FP32 대비), 정확도 변화(0.01 이상이면 경고), 엔진 변환 시간.
결과: `04_results/legacy_root/runs/tensorrt/tensorrt_summary.txt` (모델별 원본 `04_results/legacy_root/runs/tensorrt/<모델>.json`)

엔진으로 예측하기:

    from ultralytics import YOLO
    model = YOLO(r"C:\Users\USER\Desktop\trash_detection_project\04_results\baseline_training\runs\bbox_yolo11n\weights\best.engine")
    results = model.predict("사진.jpg", imgsz=640)

주의:
- 설치된 TensorRT 는 **10.16.1.11** 입니다(`uv pip install "tensorrt-cu12>=10.0,<11,!=10.1.0" onnx onnxslim`). TensorRT 11 은 FP16 변환에 NVIDIA ModelOpt 가 추가로 필요해 10.x 를 골랐습니다.
- `.engine` 은 **이 PC 의 GPU(RTX 4060)와 TensorRT 버전 전용**입니다. 다른 PC·GPU 에서는 그곳에서 다시 변환하세요.
- 엔진은 입력 크기 640×640, batch 1 로 고정해 만듭니다.
- 후처리(NMS, 마스크 계산)는 TensorRT 로 빨라지지 않아, 세그는 전체 속도 향상 폭이 추론 배수보다 작습니다.
- 학습 중에는 실행하지 않습니다(학습 프로세스가 있으면 스스로 멈춤).

## GPU 확인

`train.py` 는 GPU 를 못 쓰면 학습을 시작하지 않고 멈춥니다. CPU 로 학습하면 **25.8배 느리기** 때문입니다 (에포크 550.7초 vs 21.4초).

    uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

`2.14.0+cu126 True` 가 나와야 합니다. `+cpu` 나 `False` 면 교체하세요.

    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

## 최종 보고서

학습 4회의 결과, 모델 비교(검출 vs 세그, yolo11n vs yolo26s), 배경 이미지 효과, 개선 방법을 정리한 문서입니다.

→ **[runs/FINAL_REPORT_20260912.md](../../04_results/baseline_training/runs/FINAL_REPORT_20260912.md)**

## 학습 기록

| 과제 | 상태 | 에포크 | 소요 | 최고 mAP50-95 (검증) |
|---|---|---|---|---|
| bbox (yolo11n, 배경 포함) | 완료 (9/11 06:46) | 100 / 100, 조기 종료 없음 | 13시간 18분 | **0.8346** (92에포크) |
| seg (yolo11n-seg, 배경 포함) | 완료 (9/12 00:29) | 100 / 100 | 15시간 40분 (PC 비정상 종료로 2회 중단, 전체 경과 17시간 42분) | 박스 **0.8522** / 마스크 **0.8098** (96에포크) |
| bbox_nobg (yolo26s, 배경 제외) | 완료 (9/12 21:18) | 100 / 100 | 20시간 31분 | **0.8540** (90에포크) |
| seg_nobg (yolo26s-seg, 배경 제외) | **중단** (9/12 22:38, 사용자 결정) | 4 / 100 | 1시간 19분 | 박스 0.7105 / 마스크 0.6976 (4에포크) |
| 비교 (`compare_bg.py`) | 완료 (9/12 22:44) | — | 약 10분 | → `04_results/legacy_root/runs/compare_bg/compare_result.txt` |

**모든 학습이 끝났고, 현재 돌고 있는 학습·대기 스크립트는 없습니다.** 종합 결과와 개선 방법은 [runs/FINAL_REPORT_20260912.md](../../04_results/baseline_training/runs/FINAL_REPORT_20260912.md) 에 있습니다.

### test 분할 최종 성능 (같은 목록으로 비교)

| 모델 | 박스 mAP50-95 | 마스크 mAP50-95 | 배경 150장 오탐 (conf≥0.25) |
|---|---:|---:|---:|
| bbox yolo11n | 0.7386 | — | 5장 (3.3%) |
| seg yolo11n-seg | 0.7562 | 0.6951 | **1장 (0.7%)** |
| bbox yolo26s (배경 제외) | **0.7803** | — | 40장 (**26.7%**) |

- 같은 크기(yolo11n)에서는 **세그 모델의 박스 성능이 검출 전용보다 0.018 높습니다.** 마스크 점수가 낮은 것은 지표가 더 엄격해서입니다.
- yolo26s 는 7개 클래스 모두에서 정확도 1위지만, 배경을 빼고 학습해 **오탐이 8배 많습니다.**
- 세 학습 모두 val 보다 test 가 0.07~0.12 낮습니다. val 값을 실제 성능으로 쓰지 마세요.
- **test 로 모델을 골랐으므로 test 는 이미 판단에 쓰였습니다.** 다음 개선 실험의 비교는 val 로 하세요.
- 중단된 `seg_nobg` 는 `last.pt` 가 있어 5에포크부터 이어서 학습할 수 있습니다: `uv run python 02_training/baseline_training/train.py seg_nobg resume`
- `test.py` 는 아직 실제 모델로 실행해 보지 않았습니다(비교는 `compare_bg.py` 로 했습니다). 처음 실행할 때 출력이 정상인지 확인하세요.

## 평가 경로 변경 (2026-09-14)

`test.py`는 공통 640 test 데이터와 공통 평가 함수를 사용합니다. 결과는 `04_results/model_analysis` 아래에 실행마다 별도 저장하며, 기존 run의 `test_result.txt`를 덮어쓰지 않습니다. `compare_bg.py`는 배경 포함/제외 별도 조건을 유지하고 각각 공통 평가 기록을 남깁니다. [외부 모델과 함께 평가하는 방법](../model_analysis/README.md).
