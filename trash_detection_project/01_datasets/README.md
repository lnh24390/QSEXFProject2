# 데이터셋

- `original_bbox`: 기존 검출 데이터 보존본.
- `original_seg`: 기존 분할 데이터 보존본.
- `finetuning_640/training_640_bbox`: 실제 전이학습에 사용한 640 검출 이미지·라벨.
- `finetuning_640/training_640_seg`: 실제 전이학습에 사용한 640 분할 이미지·라벨.
- `finetuning_640/sources`, `finetuning_640/policy`: 수집 출처와 재질·품목 라벨 기준.
- `auxiliary/coco`, `auxiliary/benchmark_subset`: 배경 원천 및 과거 속도 측정용 자료.
- `auxiliary/cleanup_archive`: 이전 정리에서 남긴 복구 자료. 학습 목록에 포함하지 않습니다.

전이학습 코드는 [03_transfer_learning](../03_transfer_learning/README.md), 데이터 구축·학습 검증 보고서는 [1차 전이학습 결과](../04_results/01_initial_finetuning/README.md)에 있습니다. 이미지와 라벨은 이동만 했으며 내용과 기존 분할을 바꾸지 않았습니다.
