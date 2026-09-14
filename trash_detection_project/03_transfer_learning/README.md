# 전이학습·재학습 프로젝트

| 프로젝트 | 역할 | 결과 |
|---|---|---|
| [01_initial_finetuning](01_initial_finetuning/README.md) | 데이터 구축 코드·설정 및 기존 best_train 모델에서 출발한 1차 전이학습 | [1차 결과](../04_results/01_initial_finetuning/README.md) |
| [02_model_scaling](02_model_scaling/README.md) | 모델 크기 비교 및 사용자 요청으로 중단된 YOLO26m 재학습 | [V2 결과](../04_results/02_model_scaling/README.md) |
| [03_staged_finetuning](03_staged_finetuning/README.md) | 1차 결과에서 이어간 2차 → 3차 → 마지막 미세조정 | [단계별 결과](../04_results/03_staged_finetuning/README.md) |

세 프로젝트는 별개입니다. 검출과 분할의 클래스 순서, 설정, 데이터 목록, 가중치를 섞지 않습니다. 완료된 결과를 보려면 위 결과 링크를 사용하세요.

## 공통 분석 연결

세 프로젝트의 별도 검증·최종 평가 함수는 `../02_training/model_analysis`를 공통으로 사용합니다. 새 측정과 학습 요약은 `../04_results/model_analysis`에 실행별로 저장합니다. 단계 선택용 기존 보고서 연결과 가중치 위치는 유지합니다. [사용법](../02_training/model_analysis/README.md).
