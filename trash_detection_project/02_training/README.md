# 초기 학습 프로젝트

- [baseline_training](baseline_training/README.md): 초기 검출·분할 학습, 평가, 비교 코드.
- `dataset_preparation`: 기존 데이터 정리 및 속도 측정 스크립트. 일부는 데이터를 이동하거나 학습을 시작하는 이력 도구이므로 단순 확인 목적으로 실행하지 마세요.
- `pretrained_models`: 모델 규모별 사전학습 가중치. 내부의 `root_weights`, `preparation`, `preparation_weights`는 출처 위치별로 보존한 모델 묶음입니다.

학습 결과는 [04_results/baseline_training](../04_results/baseline_training/README.md), 정리·벤치마크 결과는 [04_results/dataset_preparation](../04_results/dataset_preparation)에 저장됩니다. `baseline_training/data_nobg`의 목록은 새 데이터 경로로 갱신했습니다.

## 공통 분석 및 외부 모델

`model_analysis`는 초기·전이·외부 모델의 공통 평가 코드입니다. `external_models`는 외부 가중치 보관 위치이며 절대 경로로 직접 평가해도 됩니다. [사용법](model_analysis/README.md)을 확인하세요. 결과는 `../04_results/model_analysis`에 저장합니다.

초기 학습 모델 5개와 사전학습 파일 12개의 [종합 분석](../04_results/model_analysis/reviews/20260914T093059245267Z_initial_training_9158da7c/summary.md)을 확인할 수 있습니다. 완료·중단 모델을 구분하고 클래스별 성능과 배경 오탐을 함께 비교했습니다.
