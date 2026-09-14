# 사전학습 가중치

초기 학습 코드가 참조하는 `yolo11n.pt`, `yolo11n-seg.pt`, `yolo26s.pt`, `yolo26s-seg.pt`와 벤치마크용 `preparation/yolo26n.pt`를 한 벌씩 유지합니다.

2026-09-14에 사용하지 않는 YOLO26l 두 파일과 중복 사본 다섯 개를 휴지통으로 이동했습니다. YOLO26m 계열은 `04_results/02_model_scaling/checkpoints`의 동일 해시 사본을 공용 경로로 참조합니다. 옛 중복 경로도 `workspace_layout.json`에서 남은 파일로 연결합니다.

대형 모델 벤치마크를 다시 실행하려면 제거된 가중치를 먼저 명시적으로 준비해야 합니다. 누락된 파일을 자동으로 다운로드하지 않습니다.
이 파일들은 80클래스 사전학습 모델입니다. 실제 쓰레기 7클래스 학습 결과는 `04_results`의 프로젝트별 가중치를 사용합니다.

정리 목록과 해시: `C:/Users/USER/Desktop/trash_detection_project_history/directory_refactor/pretrained_cleanup/cleanup.json`.
