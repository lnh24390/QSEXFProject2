# 현재 작업 디렉토리와 상관없이 공용 작업공간 경로를 찾는다.
from pathlib import Path as _WorkspacePath
import sys as _workspace_sys
_workspace_root = next(p for p in _WorkspacePath(__file__).resolve().parents if (p / 'workspace_paths.py').is_file())
_workspace_sys.path.insert(0, str(_workspace_root))
from workspace_paths import workspace_path

from pathlib import Path
import json,hashlib,shutil,datetime
R=workspace_path('MODEL_SCALING_20260913');P=workspace_path('');OLD=workspace_path('', 'KOREA_WASTE_RULES_20260913_v1')
def read(p):return json.loads(p.read_text('utf-8'))
def main():
 report={};table=['| 과제 | 모델 | batch | 이미지/초 | 최대 할당 MiB |','|---|---|---:|---:|---:|']
 for task,size,batch in [('bbox','m',8),('bbox','l',8),('seg','m',4),('seg','m',8),('seg','l',8)]:
  d=read(workspace_path('MODEL_SCALING_20260913', f'reports/benchmark_{task}_{size}_b{batch}.json'));v=d['rows'][-1]
  assert v['images_per_second']>0
  table.append(f'| {task} | YOLO26{size} | {batch} | {v["images_per_second"]:.2f} | {v["peak_allocated_mib"]:.0f} |')
 for task,batch in [('bbox',8),('seg',4)]:
  src=workspace_path('', f'yolo26m{"-seg" if task=="seg" else ""}.pt');dst=workspace_path('MODEL_SCALING_20260913', 'checkpoints', src.name)
  if src.resolve()!=dst.resolve():shutil.copy2(src,dst)
  sha=hashlib.sha256(src.read_bytes()).hexdigest();assert hashlib.sha256(dst.read_bytes()).hexdigest()==sha
  d=read(workspace_path('MODEL_SCALING_20260913', f'reports/benchmark_{task}_m_b{batch}.json'));count=read(workspace_path('MODEL_SCALING_20260913', 'reports/data_manifest.json'))[task]['counts']['train'];seconds=count/d['rows'][-1]['images_per_second']
  report[task]={'scale':'m','batch':batch,'checkpoint':str(dst),'source':str(src),'sha256':sha,'initialization':'Size-matched public COCO pretrained checkpoint, not a resumed 7-class s/n checkpoint.','estimated_training_steps_seconds_per_epoch':seconds,'estimate_excludes':'Validation, EMA, checkpoint saves, warmup and long-run variation','reason':'m has lower measured compute and memory cost than l; seg batch 4 is close in throughput to 8 with substantially more memory headroom.'}
 report['selection_basis']='Hardware feasibility, not task accuracy; l remains an untrained follow-up candidate.'
 (workspace_path('MODEL_SCALING_20260913', 'reports/model_selection.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 summary='\n## 7. 모델 크기 실측 및 선택 결과\n\n'+'\n'.join(table)+'\n\n검출은 YOLO26m batch 8, 분할은 YOLO26m-seg batch 4를 선택했다. seg batch 8은 4보다 약 3.5% 빠르지만 메모리 할당이 약 2.9→5.5GiB로 늘어 작은 batch를 선택했다. 검출 m은 l보다 약 23% 높은 처리량이다. l은 실행 불가능한 모델이 아니라 추가 시간 대비 실제 검증 향상을 아직 확인하지 않은 후보다.\n\n짧은 측정의 학습 스텝 처리량에 전체 train 수를 나눈 추정은 bbox 약 24분/에포크, seg 약 37분/에포크다. 검증·저장 등을 포함하면 더 길어진다. 두 모델이 모두 100회를 채우면 학습 스텝만 약 102시간인 큰 작업이다. 실제 첫 에포크 및 조기 종료 시점에 따라 달라지며, 이전 소형 데이터 전이학습의 6시간 예상은 적용하지 않는다.\n'
 path=workspace_path('MODEL_SCALING_20260913', '분석_및_재학습전략.md');text=path.read_text('utf-8');text=text.split('\n## 7. 모델 크기 실측 및 선택 결과')[0];path.write_text(text+summary,encoding='utf-8')
 final=read(workspace_path('KOREA_WASTE_RULES_20260913_v1', 'reports/final_training_data_validation.json'));assert final['passed'] and final['all_image_metadata_removed']
 # 진단이 끝나야 본 순차 파이프라인을 시작한다.
 diag=workspace_path('MODEL_SCALING_20260913', 'reports/warmup_control.json');assert diag.exists(),'Diagnostic must finish before training gate is enabled'
 diagnostic=read(diag);assert diagnostic['result']['metrics']['metrics/mAP50-95(B)']>0
 (workspace_path('MODEL_SCALING_20260913', 'reports/ready.json')).write_text(json.dumps({'ready':True,'checked_at':datetime.datetime.now().isoformat(),'image_validation_source':str(workspace_path('KOREA_WASTE_RULES_20260913_v1', 'reports/final_training_data_validation.json')),'no_new_image_files':True,'models':report,'diagnostic':diagnostic},ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
