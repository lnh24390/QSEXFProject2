# 순차 전이학습 최종 산출물

[최종 결과 요약](최종결과_요약.md) / [진행보고서](진행보고서.md) / [전략서](전략서.md) / [점검 기록](예약점검_기록.md)

- `runs`: bbox/seg 각각 stage2, stage3, final 및 평가 결과.
- `checkpoints/bbox_final_selected.pt`: 검출 최종 선택 모델.
- `checkpoints/seg_final_selected.pt`: 분할 최종 선택 모델. 마지막 미세조정을 채택하지 않아 3차 모델을 유지했습니다.
- `reports`: 단계별 입력·후보·채택 여부, 에포크 지표, 최종 비교 JSON.
- `logs`: 단계별 실행 로그.

[프로젝트 코드와 설정](../../03_transfer_learning/03_staged_finetuning/README.md). 원래 초기 모델은 교체하지 않았습니다.
