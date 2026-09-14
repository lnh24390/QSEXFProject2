> 최신 사용자 확정: 학습용 사본 640×640, JPEG 품질 85 및 압축 최적화, 이미지 내부 메타데이터 제거, 기존 best_train 가중치 복사본으로 전이학습. 아래 960 비교 제안은 철회되었으며 configs/retired_960_not_used에만 기록을 보관한다. 640 내장 Mosaic과 오프라인 방식의 실제 GPU 속도 비교 결과를 후속 보고서에 기록한다.

# YOLO 내장 증강·Dropout 팩트체크 및 적용안

확인일: 2026-09-13. 설치 버전: Ultralytics 8.4.146. 현재 과제는 detect와 segment다.

## 확인 결과

1. Ultralytics는 학습 중 Mosaic을 수행한다. 원본 파일을 미리 네 장씩 합쳐 저장할 필요가 없다. 원본 사진의 해상도를 유지하고 로더가 입력 크기에 맞게 변환하도록 한다. 이번 reports/augmentation_preview의 640 그림은 실제 로더 동작 확인용이며 staged 학습 이미지에 편입하지 않았다.
2. 공식 dropout 인자는 classify 전용이다. 설치된 ClassificationTrainer.get_model은 기존 torch.nn.Dropout 모듈의 p를 바꾼다. Classify 헤드에는 Dropout이 있지만 표준 Detect/Segment 헤드 생성자에는 없다. 따라서 detect/segment에 dropout=0.1을 적는 것으로 정규화 층이 생기지 않는다. 이번 설정에는 넣지 않고 모델 구조도 변경하지 않는다.
3. 내장 증강은 과적합 완화에 도움이 되지만 방지를 보장하지 않는다. 같은 제품·동영상 프레임·증강 사본이 평가 자료로 새면 Mosaic으로 해결되지 않는다. 출처/물체 단위 분할과 독립 평가가 필요하다.
4. 내장 Mosaic에서도 작은 물체가 축소되거나 잘릴 수 있다. 640이 항상 부족하다는 뜻은 아니지만 동전 건전지와 얇은 종이 경계는 해상도 비교가 필요하다. 이미 640으로 축소된 원본을 960으로 키워도 잃어버린 질감은 복구되지 않는다.

## 반영한 비교 설정

- configs/finetune_960_native.yaml: imgsz=960, mosaic=0.5, close_mosaic=5. 주 후보.
- configs/finetune_960_no_mosaic.yaml: 같은 조건에서 mosaic=0.0. Mosaic 효과 비교.
- configs/finetune_640_reference.yaml: 같은 증강에서 imgsz=640. 해상도와 비용 비교.

세 파일은 기존 체크포인트와 검수된 데이터 경로를 별도로 지정하는 학습 덮어쓰기 설정이다. 실제 학습은 아직 실행하지 않았다. 960은 32의 배수이며 640 대비 픽셀 수가 2.25배다. 이는 시간/메모리의 정확한 배수가 아니다. GPU 여유에 맞춰 배치를 조정하고 속도·메모리를 측정한다. 기존 완료 실험 설정은 덮어쓰지 않았다.

색·밝기 변환과 약한 회전/이동/크기 변환은 유지한다. mixup/cutmix/copy_paste는 0으로 시작한다. 마지막 5에포크에 Mosaic을 끈다. 설치 버전은 close_mosaic에서 copy_paste/mixup/cutmix도 함께 끄며, 일반 색상·기하 증강까지 모두 끄는 것은 아니다. 검증·시험은 무작위 학습 증강 없이 수행한다.

## 판단 방법

같은 초기 가중치·학습 자료·분할·seed·에포크 예산을 사용한다. 후보 해상도별 결과와 함께 공통 배포 입력 크기에서도 평가해 해상도 증가 효과와 가중치 개선 효과를 구분한다. 전체 mAP50-95, 클래스별 재현율, 작은 건전지 재현율, 수건→비닐·모니터→종이/배터리·코팅종이→플라스틱/비닐 오탐을 측정한다. 후보 선정은 val로 하고 고정 test는 최종 확인에 사용한다. 기존 품목 성능 하락도 확인한다.

현재 자료 중 미검수 후보와 분류만 있는 사진은 완성된 객체 탐지 정답으로 간주하지 않는다. 새로운 전이학습은 검수·중복/누수 확인 뒤 수행한다. 정확도가 개선됐다는 주장은 실제 학습·평가 전에는 하지 않는다.

## 근거와 검증 기록

- [공식 학습 인자](https://docs.ultralytics.com/modes/train/): dropout은 classification 정규화 인자.
- [공식 설정](https://docs.ultralytics.com/usage/cfg/): imgsz와 close_mosaic의 의미.
- [공식 증강 안내](https://docs.ultralytics.com/guides/yolo-data-augmentation/): Mosaic 적용과 주의 조건.
- reports/augmentation_dropout_factcheck.json: 설치된 분류 트레이너, Classify/Detect/Segment 생성자 및 close_mosaic 소스 스냅샷.
- reports/training_config_validation.json: 세 설정을 detect·segment 각각의 공식 설정 파서로 검증함. 학습 성능 검증은 아님.
