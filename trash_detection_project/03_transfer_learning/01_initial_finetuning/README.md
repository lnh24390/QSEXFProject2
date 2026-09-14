# 1차 전이학습 프로젝트

기존 `KOREA_WASTE_RULES_20260913_v1` 안에서 코드·설정·학습 목록을 분리했습니다.

- `scripts`: 수집·변환·검증 및 1차 전이학습 코드.
- `configs`: 당시 증강·학습 설정과 철회된 설정 이력.
- `finetune`: bbox/seg YAML 및 train/val/test 이미지 경로 목록, 선정 기록.
- 실제 데이터: [640 데이터셋](../../01_datasets/finetuning_640).
- 가중치·로그·검증 기록: [프로젝트 결과](../../04_results/01_initial_finetuning/README.md).

1차 학습은 완료됐습니다. `scripts/train_sequential.py`가 해당 실행 코드이며, 완료된 실행을 다시 학습하지 않는 기존 검사를 유지했습니다. 수집·변환 이력 스크립트가 참조하는 삭제 완료 원본은 디렉토리 변경으로 복원하지 않았습니다.

## 공통 평가 연결

별도 검증과 최종 평가는 [공통 분석 코드](../../02_training/model_analysis/README.md)를 사용합니다. 새 표준 결과는 `04_results/model_analysis`에 저장하며, 기존 단계 선택용 보고서도 유지합니다. 학습 설정과 가중치 경로는 유지합니다.
