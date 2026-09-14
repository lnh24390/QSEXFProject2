"""덧붙이기만 하는 이벤트 기록과 단계별 결과로 한국어 진행 보고서를 다시 쓴다."""
# 현재 작업 디렉토리와 상관없이 공용 작업공간 경로를 찾는다.
from pathlib import Path as _WorkspacePath
import sys as _workspace_sys
_workspace_root = next(p for p in _WorkspacePath(__file__).resolve().parents if (p / 'workspace_paths.py').is_file())
_workspace_sys.path.insert(0, str(_workspace_root))
from workspace_paths import workspace_path

from common import R,OLD,STAGES,read,now
def render():
 s=read(workspace_path('STAGED_FINETUNING_20260913', 'reports/state.json')) if (workspace_path('STAGED_FINETUNING_20260913', 'reports/state.json')).exists() else {}
 cp=read(workspace_path('STAGED_FINETUNING_20260913', 'reports/checkpoints.json'));data=read(workspace_path('STAGED_FINETUNING_20260913', 'reports/data_manifest.json'))
 lines=['# 단계별 전이학습 진행 보고서',f'갱신: {now()}',f'\n현재 상태: **{s.get("status","준비")}**. 과제: {s.get("task","—")}, 단계: {s.get("stage","—")}, 진행 에포크: {s.get("epoch","—")}.', '\nYOLO26m 작업은 사용자 요청으로 중단했다. 저장된 가중치와 로그는 MODEL_SCALING_20260913에 보존했다. 이번 실험은 1차 전이학습 결과의 정확한 복사본에서 출발한다. 검출은 YOLO26s, 분할은 YOLO11n-seg다. 원래 best_train의 가중치나 m의 가중치로 출발점을 바꾸지 않았다.', '\n## 단계와 데이터', '\n| 과제 | 단계 | 고유 학습 사진 | 에포크 목록 항목 | 추가 반복 |','|---|---|---:|---:|---:|']
 for task in ['bbox','seg']:
  for stage in STAGES:
   d=data[task]['stages'][stage];lines.append(f'| {task} | {stage} | {d["unique_images"]:,} | {d["epoch_entries"]:,} | {d["hard_extra_exposures"]:,} |')
 lines+=['\n같은 640 압축 JPEG와 변환 라벨을 참조한다. 새 이미지 파일은 생성하지 않는다. 과제별 클래스 ID와 자체 val/test를 유지한다. 기존 원본 train 전체를 포함하며 3차의 제한된 추가 반복은 독립 이미지 증가가 아니다.', '\n## 단계별 실제 결과','\n| 과제·단계 | 완료 에포크 | 준비 에포크 제외 평균(분) | 입력 val 점수 | 후보 val 점수 | 다음 단계 채택 |','|---|---:|---:|---:|---:|---|']
 for task in ['bbox','seg']:
  for stage in STAGES:
   result=workspace_path('STAGED_FINETUNING_20260913', f'reports/{task}_{stage}_result.json');progress=workspace_path('STAGED_FINETUNING_20260913', f'reports/{task}_{stage}_epochs.json');rows=read(progress) if progress.exists() else []
   measured=[x['seconds_including_validation'] for x in rows if not x.get('startup_epoch',False)]
   avg=f'{sum(measured)/len(measured)/60:.2f}' if measured else '—'
   if result.exists():
    d=read(result);lines.append(f'| {task} {stage} | {len(rows)} | {avg} | {d["input_score"]:.5f} | {d["candidate_score"]:.5f} | {"채택" if d["promotion"]["accepted"] else "입력 유지"} |')
   else:lines.append(f'| {task} {stage} | {len(rows)} | {avg} | — | — | {"진행 중" if s.get("task")==task and s.get("stage")==stage else "대기"} |')
 lines+=['\n점수는 bbox의 박스 mAP50–95, seg의 박스+마스크 mAP50–95 합이다. 두 과제의 값을 직접 비교하지 않는다. 평균은 각 실행 시작 에포크를 제외하며 검증 시간이 포함된다. 시작 에포크도 원시 JSON 기록에는 남긴다.', '\n## 채택과 원본 성능 회복 여부', '\n각 단계의 입력 가중치와 후보를 같은 val에서 비교한다. 점수 상승, 과제별 평균 비하락, 각 클래스 AP 하락 1%p 이내를 모두 만족할 때만 다음 단계 입력으로 채택한다. 실패한 단계의 후보도 별도로 보존한다. 이 규칙은 실험 전에 정한 운영 기준이며 통계적 유의성을 증명하지 않는다. 원본 best_train보다 좋아졌는지와 1차 전이학습보다 회복했는지는 별도로 표시한다.']
 for task in ['bbox','seg']:
  final=workspace_path('STAGED_FINETUNING_20260913', f'reports/{task}_final_comparison.json')
  if final.exists():
   d=read(final);lines.append(f'\n- {task}: 원본 검증 성능 회복 기준 통과={d["original_validation_gate"]["accepted"]}, 자동 배포=False. 최종 후보: {d["selected_checkpoint"]}')
 lines+=['\n## 기록 위치', '\n- 설정: configs/stage2.yaml, stage3.yaml, final.yaml\n- 출발점과 SHA256: reports/checkpoints.json\n- 이벤트 이력: reports/events.jsonl\n- 에포크 지표·시간: reports/*_epochs.json\n- 단계별 입력/후보/채택 및 클래스 지표: reports/*_result.json\n- 최종 평가: reports/*_final_comparison.json\n- 학습 원시 로그: logs/\n- 단계별 가중치: runs/\n- 다음 단계로 넘기는 복사본: checkpoints/', '\n성능이 계속 상승한다고 보장하지 않는다. 시험 자료는 단계별 설정이나 선택에 사용하지 않으며 각 과제의 최종 선택 뒤에 평가한다. 이미 열람한 기존 시험 자료라는 한계와 한국 현장·흐림 등 별도 미학습 평가 자료 부족을 함께 기록한다.']
 (workspace_path('STAGED_FINETUNING_20260913', '진행보고서.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
if __name__=='__main__':render()
