# 데이터 구축 및 1차 전이학습 결과

- `runs/bbox_transfer_640`, `runs/seg_transfer_640`: 1차 학습 결과.
- `checkpoints`: 초기 best_train에서 복사한 출발 가중치.
- `evaluations`: 기존 모델과 1차 모델의 평가 산출물.
- `reports`: 데이터 수집·변환·검증, 혼합 분포, 학습 상태와 비교 결과.
- `logs`: 1차 학습 및 파이프라인 로그.
- `benchmarks`: 당시 Mosaic 속도 비교 기록.

[코드·설정·학습 목록](../../03_transfer_learning/01_initial_finetuning/README.md) / [실제 데이터](../../01_datasets/finetuning_640)
