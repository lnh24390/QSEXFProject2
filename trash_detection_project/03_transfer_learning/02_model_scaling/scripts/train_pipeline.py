"""V2 는 GPU 로만 한다. 선택한 검출 모델을 먼저, 그다음 세그를 학습한다. 자동 배포는 하지 않는다."""
# 현재 작업 디렉토리와 상관없이 공용 작업공간 경로를 찾는다.
from pathlib import Path as _WorkspacePath
import sys as _workspace_sys
_workspace_root = next(p for p in _WorkspacePath(__file__).resolve().parents if (p / 'workspace_paths.py').is_file())
_workspace_sys.path.insert(0, str(_workspace_root))
from workspace_paths import workspace_path
_workspace_sys.path.insert(0, str(workspace_path("02_training")))
from model_analysis.schema import summary
from model_analysis.evaluation import evaluate_model
from model_analysis.history import save_training_summary
from workspace_trainers import trainer_for

from pathlib import Path
import json,sys,time,datetime,subprocess,os,hashlib
import torch,yaml
from ultralytics import YOLO
R=workspace_path('MODEL_SCALING_20260913');OLD=workspace_path('.', 'KOREA_WASTE_RULES_20260913_v1')
def read(p):return json.loads(p.read_text('utf-8'))
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def state(**values):
 p=workspace_path('MODEL_SCALING_20260913', 'reports/state.json');d=read(p) if p.exists() else {};d.update(values,updated_at=datetime.datetime.now().isoformat(timespec='seconds'));tmp=p.with_suffix('.tmp');write(tmp,d);tmp.replace(p)
def worker(task):
 assert torch.cuda.is_available(),'GPU required'
 ready=read(workspace_path('MODEL_SCALING_20260913', 'reports/ready.json'));assert ready['ready']
 selection=read(workspace_path('MODEL_SCALING_20260913', 'reports/model_selection.json'))[task];ck=Path(selection['checkpoint'])
 assert hashlib.sha256(ck.read_bytes()).hexdigest()==selection['sha256']
 config=yaml.safe_load((workspace_path('MODEL_SCALING_20260913', 'configs/larger_model.yaml')).read_text());config['batch']=selection['batch'];config['augmentations']=[]
 data=workspace_path('MODEL_SCALING_20260913', f'data/{task}.yaml');name=f'{task}_yolo26{selection["scale"]}_v2';out=workspace_path('MODEL_SCALING_20260913', 'runs', name);last=out/'weights/last.pt';result_path=workspace_path('MODEL_SCALING_20260913', f'reports/{task}_result.json')
 if result_path.exists():return
 resuming=last.exists();completed=0
 if resuming:
  checkpoint=torch.load(last,map_location='cpu',weights_only=False)
  assert checkpoint.get('optimizer') is not None,'Completed/stripped checkpoint cannot resume'
  assert checkpoint['train_args']['epochs']==config['epochs']
  completed=checkpoint['epoch']+1;del checkpoint
 model=YOLO(str(last if resuming else ck));progress=workspace_path('MODEL_SCALING_20260913', f'reports/{task}_epochs.json')
 rows=[x for x in read(progress) if x['epoch']<=completed] if resuming and progress.exists() else []
 baseline=read(workspace_path('./KOREA_WASTE_RULES_20260913_v1', f'reports/{task}_baseline_val.json'))
 basefitness=baseline['box']['map']+(baseline.get('seg',{}).get('map',0))
 def train_start(t):
  assert t.args.warmup_bias_lr==0 and t.epochs==100
  assert list(t.model.names.values())==list(yaml.safe_load(data.read_text())['names'].values())
  if resuming:
   for row in rows:t.stopper(row['epoch'],row['fitness'])
  state(status='training',active_task=task,model=f'YOLO26{selection["scale"]}'+('-seg' if task=='seg' else ''),batch=t.batch_size,max_epochs=t.epochs,started_at=datetime.datetime.now().isoformat())
 def begin(t):
  t._v2_start=time.perf_counter();state(epoch=t.epoch+1)
 def end(t):
  # final_eval 이 이 콜백을 다시 부를 수 있다. 실제로 끝난 에포크만 남긴다.
  if not hasattr(t,'_v2_start') or (rows and rows[-1]['epoch']>=t.epoch+1):return
  row={'epoch':t.epoch+1,'seconds_including_validation':time.perf_counter()-t._v2_start,'fitness':float(t.fitness),'metrics':{k:float(v) for k,v in t.metrics.items()},'beats_original_val_fitness':bool(t.fitness>=basefitness)};del t._v2_start;rows.append(row);write(progress,rows)
  state(last_completed_epoch=row['epoch'],last_epoch_seconds=row['seconds_including_validation'],metrics=row['metrics'],beats_original_val_fitness=row['beats_original_val_fitness'])
 model.add_callback('on_train_start',train_start);model.add_callback('on_train_epoch_start',begin);model.add_callback('on_fit_epoch_end',end)
 state(status='initializing',active_task=task,detail='Full original replay and public pretrained larger model; old weights preserved')
 model.train(trainer=trainer_for(task),data=str(data),project=str(workspace_path('MODEL_SCALING_20260913', 'runs')),name=name,exist_ok=resuming,resume=str(last) if resuming else False,**config)
 save_training_summary(out, '02_model_scaling')
 best=out/'weights/best.pt';assert best.exists();del model;torch.cuda.empty_cache()
 state(status='evaluating',active_task=task)
 final=YOLO(str(best))
 val=summary(evaluate_model(best, data, model=final, split='val', task=task, source='02_model_scaling', name=task+'_best_val'))
 test=summary(evaluate_model(best, data, model=final, split='test', task=task, source='02_model_scaling', name=task+'_final_test'))
 oldtest=read(workspace_path('./KOREA_WASTE_RULES_20260913_v1', f'reports/{task}_transfer_comparison.json'))['baseline']
 keys=['box']+(['seg'] if task=='seg' else [])
 gates={kind:val[kind]['map']>=baseline[kind]['map'] for kind in keys}
 result={'model':selection,'best_checkpoint':str(best),'baseline_val':baseline,'new_val':val,'baseline_legacy_test':oldtest,'new_test':test,'validation_gates':gates,'all_validation_gates_passed':all(gates.values()),'deployment':'Not deployed; target-case independent evaluation still needed.','legacy_test_note':'Previously inspected legacy test; no hyperparameter selection based on this comparison.'}
 write(result_path,result);state(status='task_complete',active_task=task,all_validation_gates_passed=all(gates.values()))
def main():
 if len(sys.argv)>1:
  try:worker(sys.argv[1])
  except Exception as e:state(status='failed',active_task=sys.argv[1],error=str(e));raise
  return
 lock=workspace_path('MODEL_SCALING_20260913', 'reports/pipeline.lock')
 with lock.open('x') as f:f.write(str(os.getpid()))
 try:
  state(status='starting',pipeline_pid=os.getpid(),sequence=['bbox','seg'])
  for task in ['bbox','seg']:
   with (workspace_path('MODEL_SCALING_20260913', f'logs/{task}_training.log')).open('a',encoding='utf-8') as log:
    process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),task],stdout=log,stderr=subprocess.STDOUT);state(worker_pid=process.pid);rc=process.wait()
   if rc:raise RuntimeError(f'{task} failed ({rc}); next task not started')
  state(status='complete',active_task=None,completed_at=datetime.datetime.now().isoformat())
 except Exception as e:state(status='failed',error=str(e));raise
 finally:lock.unlink(missing_ok=True)
if __name__=='__main__':main()
