# 두 번째 학습 전략 및 실험

첫 전이학습의 성능 하락을 분석하고 YOLO26m/l의 GPU 적합성을 실측하는 별도 실험 폴더다. 기존 원본 데이터, best_train 가중치, 첫 실험 결과는 보존한다.

- 분석과 전략: `분석_및_재학습전략.md`
- 데이터 목록과 클래스 분포: `data/`, `reports/data_manifest.json`
- 원인 조사: `reports/failure_audit.json`, `reports/warmup_control.json`
- 모델 크기별 실측: `reports/benchmark_*.json`
- 새 학습 설정·결과·진행 상태: `configs/`, `04_results/legacy_root/runs/`, `reports/state.json`

이미지는 기존 새 데이터셋의 메타정보를 제거한 640 JPEG를 참조한다. 이 폴더에 이미지 원본이나 증강 이미지 사본을 중복 저장하지 않는다. 라벨은 과제별 기존 640 변환 라벨을 사용한다. 출처·라이선스는 해당 데이터셋 sources/reports에 보존되어 있다.

확대 모델은 해당 크기의 COCO 사전학습 가중치에서 시작한다. 이전 YOLO26s 또는 YOLO11n 쓰레기 모델의 학습 상태를 그대로 확대하는 방식은 아니다. 기존 모델은 비교 기준으로 보존하며, 검증 성능 개선이 확인되지 않은 모델을 자동으로 교체하지 않는다.

사용자 요청으로 YOLO26m 학습을 중단했습니다. 후속 작업과 자동 보고서는 ../03_staged_finetuning/에 있습니다. m 체크포인트와 기록은 보존되어 있습니다.

## 공통 평가 연결

별도 검증과 최종 평가는 [공통 분석 코드](../../02_training/model_analysis/README.md)를 사용합니다. 새 표준 결과는 `04_results/model_analysis`에 저장하며, 기존 단계 선택용 보고서도 유지합니다. 학습 설정과 가중치 경로는 유지합니다.
