# trash_detection_project (코드·설정·문서 사본)

쓰레기 7클래스 검출(bbox)·분할(seg) 학습 프로젝트입니다. 이 폴더는 GitHub 업로드용 사본이며 **코드, 설정, 학습 목록, 보고서만** 들어 있습니다. 이미지·라벨·가중치·로그 같은 큰 파일은 빠져 있으므로, 아래 안내대로 내려받아 같은 위치에 넣어야 실행됩니다.

원본 작업 폴더는 `C:/Users/USER/Desktop/trash_detection_project`입니다.

## 폴더 구성

| 폴더 | 들어 있는 것 |
|---|---|
| `01_datasets` | 데이터 설명·클래스 정의·수집 출처 기록(`sources`)·라벨 기준(`policy`). **이미지와 라벨은 없음** |
| `02_training` | 초기 학습 코드(`baseline_training`), 데이터 정리·벤치마크 코드(`dataset_preparation`). **사전학습 `.pt` 없음** |
| `03_transfer_learning` | 전이학습 세 프로젝트의 코드·데이터 YAML·분할 목록(`01_initial_finetuning`, `02_model_scaling`, `03_staged_finetuning`) |
| `04_results` | 보고서·지표 텍스트·평가 JSON. **가중치(`.pt`)·실행 로그·그림 없음** |
| 루트 파일 | `workspace_paths.py`, `workspace_layout.json`, `workspace_trainers.py`(경로 해석), `pyproject.toml`, `uv.lock`(의존성) — 코드 실행에 필요해 함께 담았습니다 |

## 빠진 것과 용량

| 빠진 것 | 원본 용량 | 어디서 구하나 |
|---|---:|---|
| 학습 이미지·라벨 (`01_datasets` 전체) | 약 31 GB / 647,258개 | 아래 "데이터셋 준비" |
| 학습 결과 가중치 `.pt` (`04_results/**/weights`, `checkpoints`) | 약 1.25 GB / 46개 | 아래 "가중치" |
| 사전학습 모델 (`02_training/pretrained_models/*.pt`) | 약 59 MB / 5개 | Ultralytics 공개 가중치 |
| 학습 로그 `.log`(191 MB), 그림 `.png/.jpg`, 백업 `.zip`, 캐시 `.cache` | 약 290 MB | 다시 만들면 되는 실행 기록 |

분할 목록 `.txt`(최대 7.7 MB)와 평가 JSON은 학습 재현에 필요해 **그대로 담았습니다**.

## 데이터셋 준비

### 1) 기존 7클래스 데이터 (`original_bbox`, `original_seg`)

직접 수집·가공한 비공개 자료라 공개 배포처가 없습니다. 원본 보관 위치는 `C:/Users/USER/Desktop/dataset/YOLO_7CLASS_10000_PER_CLASS_20260909`(검출)이며, 세그 사본은 별도 백업이 없습니다. 다른 PC에서 쓰려면 이 폴더를 그대로 복사해 아래 구조로 둡니다.

```text
01_datasets/original_bbox/
├─ images/{train,val,test}/*.jpg
├─ labels/{train,val,test}/*.txt
├─ data.yaml, classes.txt, summary.json
└─ selection_manifest.jsonl.gz      # 분할 그룹(split_group) 정보
01_datasets/original_seg/
├─ images/{train,val,test}/*.jpg    # 파일명: images__<원래분할>__<원본stem>.jpg
├─ labels/{train,val,test}/*.txt
└─ data.yaml
```

**클래스 순서가 서로 다릅니다. 섞지 마세요.**

- 검출: `0 paper, 1 plastic, 2 can, 3 glass, 4 battery, 5 vinyl, 6 general`
- 분할: `0 general, 1 can, 2 plastic, 3 paper, 4 glass, 5 vinyl, 6 battery`

현재 분할(배경 포함): 검출 train 49,874 / val 9,628 / test 2,734, 세그 train 49,873 / val 9,629 / test 2,734.

### 2) 배경 이미지 (COCO)

7클래스가 없는 사진 3,000장을 오탐 감소용으로 넣었습니다. 라벨은 **빈 `.txt`** 입니다.

1. COCO 2017 train 이미지와 주석을 받습니다.
   - 이미지: <http://images.cocodataset.org/zips/train2017.zip>
   - 주석: <http://images.cocodataset.org/annotations/annotations_trainval2017.zip> → `instances_train2017.json`
2. `01_datasets/auxiliary/coco/annotations/instances_train2017.json`에 주석을 둡니다.
3. `02_training/dataset_preparation/coco_select_bg.py` → `coco_fetch_bg.py` → `add_backgrounds.py` 순으로 실행하면 `auxiliary/coco/bg_640`에 640×640 배경이 만들어지고 두 데이터셋에 하드링크로 들어갑니다. 선별 기준(제외 클래스, 면적 3% 이상 등)은 스크립트 상단 주석에 있습니다.

### 3) 전이학습용 640 데이터 (`finetuning_640`)

여러 공개 데이터에서 모아 640×640으로 변환한 자료입니다. 수집 기록은 `01_datasets/finetuning_640/sources/`의 JSON에 원본 URL·SHA-256·파일 수까지 남아 있습니다.

| 출처 | 받는 곳 | 기록 파일 |
|---|---|---|
| TACO | <https://doi.org/10.5281/zenodo.3587843> (`TACO.zip`, 2.72 GB, sha256 `51cd6b00…`) | `sources/taco_archive/download_receipt.json` |
| Garbage Classification v2 (Kaggle) | `kaggle datasets download sumn2u/garbage-classification-v2` (1.15 GB) | `sources/garbage_v2/download_receipt.json` |
| Object Detection Batteries (Kaggle) | `kaggle datasets download markcsizmadia/object-detection-batteries-dices-and-toy-cars` (244 MB) | `sources/household_batteries/download_receipt.json` |
| Open Images | 매니페스트의 `download_url`로 개별 내려받기 (train/val/test 합계 1,000여 장) | `sources/openimages/*_download_manifest.json` |
| Wikimedia Commons | 검색어별 내려받기 기록 | `sources/commons/queries.json`, `download_manifest.json` |
| 기타(coin_cells, rvm, warp 등) | 각 폴더의 매니페스트 참조 | `sources/<이름>/` |

변환 후 최종 구조는 아래와 같습니다. 이미지는 모두 640×640 JPEG이며 EXIF·GPS는 제거했습니다.

```text
01_datasets/finetuning_640/
├─ training_640_bbox/{images,labels}/{train,val,test}/
├─ training_640_seg/{images,labels}/{train,val,test}/
├─ policy/    (재질·품목 라벨 기준 — 이 사본에 포함)
└─ sources/   (수집 기록 — 이 사본에 포함)
```

어떤 이미지를 어느 분할에 쓰는지는 `03_transfer_learning/*/data/*.txt` 목록이 정합니다.

### 4) 절대 경로 수정

데이터 YAML과 목록 `.txt`에는 `C:\Users\USER\Desktop\trash_detection_project\...` 절대 경로가 들어 있습니다. 다른 위치에서 쓰려면 경로를 새 루트로 바꿔야 합니다.

```powershell
# 예: 목록·YAML 안의 루트 경로를 일괄 치환
Get-ChildItem -Recurse -Include *.yaml,*.txt |
  ForEach-Object {
    (Get-Content $_ -Raw) -replace 'C:\\Users\\USER\\Desktop\\trash_detection_project', $PWD.Path |
      Set-Content $_ -Encoding utf8
  }
```

`workspace_layout.json`의 경로도 함께 확인하세요.

## 가중치

- **사전학습 모델**: Ultralytics 공개 가중치를 `02_training/pretrained_models/`에 둡니다(`yolo11n.pt`, `yolo11n-seg.pt`, `yolo26s.pt`, `yolo26s-seg.pt`). `YOLO("yolo11n.pt")`를 처음 호출하면 자동으로 내려받습니다.
- **학습 결과 가중치**: 이 저장소에 없습니다. 최종 선택 모델은 원본 PC의 `04_results/03_staged_finetuning/checkpoints/bbox_final_selected.pt`, `seg_final_selected.pt`입니다. 필요하면 원본 PC에서 따로 받아 같은 경로에 두세요.

## 실행 환경

- Python 패키지 관리자는 **uv**를 씁니다. `pip`·`conda`를 직접 쓰지 않습니다.
- torch는 **CUDA 빌드(cu126)** 여야 합니다. 학습 전 `torch.cuda.is_available()`이 `True`인지 확인하고, `False`면 학습하지 않습니다(CPU는 실측 25.8배 느림).
- 의존성 정의(`pyproject.toml`, `uv.lock`)는 이 사본에 함께 담았습니다.

```powershell
uv sync --frozen
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

- 한글 출력이 깨지면 `PYTHONIOENCODING=utf-8`을 설정합니다. PowerShell 5.1로 실행할 `.ps1`에 한글이 있으면 **UTF-8 BOM**으로 저장해야 합니다.

## 학습·평가 실행

```powershell
# 초기 학습 (bbox 또는 seg)
uv run python 02_training/baseline_training/train.py bbox
uv run python 02_training/baseline_training/train.py seg resume   # 중단 시 이어서

# 단계별 전이학습
uv run python 03_transfer_learning/03_staged_finetuning/scripts/pipeline.py
```

- 학습은 **GPU로만** 하고, 한 번에 하나만 돌립니다.
- 최종 성능은 학습이 끝난 뒤 **test 분할로 한 번만** 잽니다. 설정 선택에는 val을 씁니다.
- 검출과 세그는 클래스 순서와 분할이 달라 **결과를 직접 비교하지 않습니다**(전체 62,236장 중 5,786장이 서로 다른 분할에 있음).

## 현재까지의 결과 (참고)

test mAP50-95(%) 기준입니다. 검출과 분할은 test 구성이 달라 서로 비교하지 않습니다.

| 지표 | 초기 모델 | 1차 전이학습 | 최종 선택 |
|---|---:|---:|---:|
| 검출 박스 | 76.11 | 68.14 | 74.11 |
| 분할 마스크 | 68.66 | 68.20 | 68.71 |

초기 모델을 넘어서지 못했습니다. 자세한 내용은 `04_results/03_staged_finetuning/최종결과_요약.md`에 있습니다.
