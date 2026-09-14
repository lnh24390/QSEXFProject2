# 1차 결과에서 이어가는 단계별 전이학습

[전략서](../../04_results/03_staged_finetuning/전략서.md)와 [자동 갱신 진행보고서](../../04_results/03_staged_finetuning/진행보고서.md)를 참고한다.

모델은 검출 YOLO26s와 분할 YOLO11n-seg다. 각 1차 전이학습 best.pt에서 출발해 2차, 3차, 마지막 미세조정을 GPU에서 순차 실행한다. YOLO26m 실험은 사용자 요청으로 중단하고 보존했다.

이미지는 `../../01_datasets/finetuning_640/training_640_bbox` 및 `training_640_seg`를 참조한다. 설정은 이 프로젝트의 `configs`, 목록은 `data`에 있다. 출발점·선택 가중치, 이벤트·지표, runs와 로그는 [프로젝트 결과 폴더](../../04_results/03_staged_finetuning/README.md)에 분리했다.

새 모델이 기존보다 좋아졌다는 뜻은 아니다. 단계별 검증을 통과한 가중치만 다음 단계로 선택하며 최종 원본 대비 평가 전에는 자동 배포하지 않는다.

## 공통 평가 연결

별도 검증과 최종 평가는 [공통 분석 코드](../../02_training/model_analysis/README.md)를 사용합니다. 새 표준 결과는 `04_results/model_analysis`에 저장하며, 기존 단계 선택용 보고서도 유지합니다. 학습 설정과 가중치 경로는 유지합니다.
