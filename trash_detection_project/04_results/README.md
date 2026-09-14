# 프로젝트별 산출물

실제 파일을 프로젝트별 하위 폴더로 이동했습니다. 심볼릭 링크나 동일 이름 파일의 병합을 사용하지 않았습니다.

| 폴더 | 산출물 |
|---|---|
| [baseline_training](baseline_training/README.md) | 초기 학습·배경 비교·해상도 비교의 runs |
| [01_initial_finetuning](01_initial_finetuning/README.md) | 1차 전이학습 runs/evaluations, 시작 가중치, 데이터 구축·검증 reports, 로그, Mosaic 벤치마크 |
| [02_model_scaling](02_model_scaling/README.md) | 중단된 m 실험, 크기별 벤치마크, 진단, 모델 선택 기록 |
| [03_staged_finetuning](03_staged_finetuning/README.md) | 검출·분할 2차/3차/final 실행, 선택 가중치, 최종 평가와 보고서 |
| `dataset_preparation` | 이전 데이터 정리·이동 기록, 벤치마크, 차트 |
| `legacy_root/runs` | 이전 루트 runs 보존 |
| `project_analysis` | 프로젝트 분석·학습 전략·진행 보고서와 근거 |
| `guides` | 학습 속도 개선 가이드 |

예를 들어 초기 `best.pt`는 `baseline_training/runs/<실험>/weights/best.pt`, 단계별 `best.pt`는 `03_staged_finetuning/runs/<bbox|seg>_<stage2|stage3|final>/weights/best.pt`에 있습니다. 프로젝트 이름과 실험/단계 경로를 유지해 충돌을 방지했습니다.

최종 선택 모델도 자동 배포한 모델을 뜻하지 않습니다. [최종 결과 요약](03_staged_finetuning/최종결과_요약.md)의 초기 모델 대비 비교를 확인하세요.

## 모델 통합 분석

[model_analysis](model_analysis/README.md)는 기존 모델과 외부 모델을 같은 형식으로 평가·비교하는 결과 폴더입니다. [최신 결과 목록](model_analysis/index.md)에서 새 평가와 과거 결과를 구분해 확인하세요. `project_analysis`는 기존 프로젝트 분석 문서와 전략 이력입니다.
