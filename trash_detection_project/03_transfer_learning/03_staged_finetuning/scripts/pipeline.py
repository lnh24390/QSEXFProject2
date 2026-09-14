"""GPU 실행 6개를 엄격히 순서대로 돌린다. 단계 전환은 측정한 검증 기준으로 판단한다."""
# 현재 작업 디렉토리와 상관없이 공용 작업공간 경로를 찾는다.
from pathlib import Path as _WorkspacePath
import sys as _workspace_sys
_workspace_root = next(p for p in _WorkspacePath(__file__).resolve().parents if (p / 'workspace_paths.py').is_file())
_workspace_sys.path.insert(0, str(_workspace_root))
from workspace_paths import workspace_path
from workspace_trainers import trainer_for
_workspace_sys.path.insert(0, str(workspace_path("02_training")))
from model_analysis.evaluation import evaluate_model
from model_analysis.history import save_training_summary

from pathlib import Path
import sys,time,subprocess,os,shutil,math
import torch,yaml
from ultralytics import YOLO
from common import R,OLD,STAGES,read,write,sha,event,state,summary,score,promotion,now
from report import render
def transforms(stage):
 if stage!='stage3':return []
 import albumentations as A
 return [A.OneOf([A.GaussianBlur(blur_limit=(3,3),sigma_limit=(.2,.8),p=1),A.MotionBlur(blur_limit=(3,3),allow_shifted=False,p=1)],p=.10)]
def evaluate(path,task,name,split='val'):
 net=YOLO(str(path));result=evaluate_model(path, workspace_path('STAGED_FINETUNING_20260913', f'data/{task}_stage2.yaml'), model=net, split=split, task=task, source='03_staged_finetuning', name=name)
 d=summary(result);del net;torch.cuda.empty_cache();return d
def incoming(task,stage):
 idx=STAGES.index(stage)
 if idx==0:
  cp=read(workspace_path('STAGED_FINETUNING_20260913', 'reports/checkpoints.json'))[task];path=Path(cp['copy']);assert sha(path)==cp['sha256'];baseline=workspace_path('STAGED_FINETUNING_20260913', f'reports/{task}_first_transfer_val.json')
  if not baseline.exists():
   state(status='baseline_validation',task=task,stage=stage);render();write(baseline,evaluate(path,task,task+'_first_transfer_val'))
  return path,read(baseline)
 prev=read(workspace_path('STAGED_FINETUNING_20260913', f'reports/{task}_{STAGES[idx-1]}_result.json'));path=Path(prev['selected_checkpoint']);assert sha(path)==prev['selected_sha256'];return path,prev['selected_validation']
def stage_worker(task,stage):
 assert torch.cuda.is_available(),'GPU required';assert read(workspace_path('STAGED_FINETUNING_20260913', 'reports/preparation.json'))['passed'] and read(workspace_path('STAGED_FINETUNING_20260913', 'reports/verification.json'))['passed']
 result_path=workspace_path('STAGED_FINETUNING_20260913', f'reports/{task}_{stage}_result.json')
 if result_path.exists():return
 source,baseline=incoming(task,stage);cfg=yaml.safe_load((workspace_path('STAGED_FINETUNING_20260913', f'configs/{stage}.yaml')).read_text());name=f'{task}_{stage}';out=workspace_path('STAGED_FINETUNING_20260913', 'runs', name);last=out/'weights/last.pt';completed_marker=workspace_path('STAGED_FINETUNING_20260913', f'reports/{name}_training_complete.json');progress=workspace_path('STAGED_FINETUNING_20260913', f'reports/{name}_epochs.json')
 attempt=workspace_path('STAGED_FINETUNING_20260913', f'reports/{name}_input.json')
 if not attempt.exists():write(attempt,{'source':str(source),'sha256':sha(source),'validation':baseline,'config':cfg,'started_at':now()})
 else:assert read(attempt)['sha256']==sha(source)
 rows=read(progress) if progress.exists() else [];start_epoch=0;resuming=last.exists() and not completed_marker.exists()
 if not completed_marker.exists():
  if resuming:
   checkpoint=torch.load(last,map_location='cpu',weights_only=False)
   if checkpoint.get('optimizer') is None:
    # 가벼워진 체크포인트는 Ultralytics 가 끝냈다는 뜻이다. 기존 실행을 마무리한다.
    assert (out/'weights/best.pt').exists() and rows
    write(completed_marker,{'recovered_after_final_eval':True,'completed_at':now()});resuming=False
   else:
    assert checkpoint['train_args']['epochs']==cfg['epochs'];start_epoch=checkpoint['epoch']+1;rows=[r for r in rows if r['epoch']<=start_epoch]
   del checkpoint
  if not completed_marker.exists():
   model=YOLO(str(last if resuming else source));source_score=score(baseline,task);session_start=start_epoch
   def on_start(t):
    assert t.args.warmup_bias_lr==0 and t.epochs==100
    assert {str(k):v for k,v in t.model.names.items()}==baseline['names']
    # 이어받은 검증 점수를 10에포크 동안 넘지 못하면 조기 종료한다.
    t.stopper.best_fitness=source_score;t.stopper.best_epoch=0
    for r in rows:t.stopper(r['epoch'],r['fitness'])
    state(status='training',task=task,stage=stage,model=read(workspace_path('STAGED_FINETUNING_20260913', 'reports/checkpoints.json'))[task]['model'],max_epochs=t.epochs,batch=t.batch_size,source_checkpoint=str(source));event('training_started',task=task,stage=stage,resume=resuming,start_epoch=start_epoch,source_sha256=sha(source));render()
   def begin(t):
    t._stage_start=time.perf_counter();t._stage_tick=time.monotonic();state(epoch=t.epoch+1,completed_epochs=len(rows));render()
   def tick(t):
    if stage=='stage2' and not hasattr(t,'_frozen_checked'):
     prefix=tuple(f'model.{i}.' for i in range(10))
     frozen=[p for n,p in t.model.named_parameters() if n.startswith(prefix)];assert frozen and all(not p.requires_grad for p in frozen)
     norms=[m for n,m in t.model.named_modules() if n.startswith(prefix) and isinstance(m,torch.nn.BatchNorm2d)];assert all(not m.training for m in norms)
     t._frozen_checked=True;event('freeze_runtime_verified',task=task,stage=stage,frozen_parameter_tensors=len(frozen),frozen_bn_modules=len(norms))
    if time.monotonic()-t._stage_tick>=60:
     t._stage_tick=time.monotonic();state(current_epoch_elapsed_seconds=time.perf_counter()-t._stage_start);render()
   def end(t):
    if not hasattr(t,'_stage_start'):return
    elapsed=time.perf_counter()-t._stage_start;del t._stage_start
    metrics={k:float(v) for k,v in t.metrics.items()};assert all(math.isfinite(v) for v in metrics.values())
    row={'epoch':t.epoch+1,'seconds_including_validation':elapsed,'startup_epoch':t.epoch==session_start,'fitness':float(t.fitness),'metrics':metrics,'above_stage_input':t.fitness>source_score+.0001}
    rows.append(row);write(progress,rows);state(completed_epochs=len(rows),last_epoch_seconds=elapsed,metrics=metrics);event('epoch_completed',task=task,stage=stage,**row);render()
   model.add_callback('on_train_start',on_start);model.add_callback('on_train_epoch_start',begin);model.add_callback('on_train_batch_end',tick);model.add_callback('on_fit_epoch_end',end)
   model.train(trainer=trainer_for(task),data=str(workspace_path('STAGED_FINETUNING_20260913', f'data/{task}_{stage}.yaml')),project=str(workspace_path('STAGED_FINETUNING_20260913', 'runs')),name=name,exist_ok=resuming,resume=str(last) if resuming else False,augmentations=transforms(stage),**cfg)
   save_training_summary(out, '03_staged_finetuning')
   write(completed_marker,{'completed_at':now(),'epochs':len(rows)});del model;torch.cuda.empty_cache()
 best=out/'weights/best.pt';assert best.exists();state(status='stage_validation',task=task,stage=stage);render()
 candidate=evaluate(best,task,name+'_candidate_val');gate=promotion(baseline,candidate,task)
 selected_source=best if gate['accepted'] else source;selected=workspace_path('STAGED_FINETUNING_20260913', f'checkpoints/{task}_{stage}_selected.pt');shutil.copy2(selected_source,selected)
 result={'task':task,'stage':stage,'input_checkpoint':str(source),'input_sha256':sha(source),'input_validation':baseline,'input_score':score(baseline,task),'candidate_checkpoint':str(best),'candidate_validation':candidate,'candidate_score':score(candidate,task),'promotion':gate,'selected_checkpoint':str(selected),'selected_sha256':sha(selected),'selected_validation':candidate if gate['accepted'] else baseline,'completed_at':now()}
 write(result_path,result);event('stage_selected',task=task,stage=stage,accepted=gate['accepted'],selected_checkpoint=str(selected),checks=gate['checks']);state(status='stage_complete',task=task,stage=stage,accepted=gate['accepted']);render()
def final_worker(task):
 dest=workspace_path('STAGED_FINETUNING_20260913', f'reports/{task}_final_comparison.json')
 if dest.exists():return
 final=read(workspace_path('STAGED_FINETUNING_20260913', f'reports/{task}_final_result.json'));selected=Path(final['selected_checkpoint']);assert sha(selected)==final['selected_sha256']
 state(status='final_test',task=task,stage='evaluation');render();test=evaluate(selected,task,task+'_selected_final_test',split='test')
 original=read(workspace_path('KOREA_WASTE_RULES_20260913_v1', f'reports/{task}_baseline_val.json'));legacy=read(workspace_path('KOREA_WASTE_RULES_20260913_v1', f'reports/{task}_transfer_comparison.json'))
 d={'selected_checkpoint':str(selected),'selected_sha256':sha(selected),'selected_validation':final['selected_validation'],'original_validation':original,'original_validation_gate':promotion(original,final['selected_validation'],task),'new_test':test,'original_legacy_test':legacy['baseline'],'first_transfer_legacy_test':legacy['finetuned'],'deployed':False,'note':'Final selected model only; legacy test was inspected previously, not used for stage selection. Independent target-case evaluation remains necessary.'}
 write(dest,d);event('task_final_evaluation',task=task,original_validation_gate=d['original_validation_gate']);render()
def main():
 if len(sys.argv)>1:
  task,stage=sys.argv[1:3]
  try:final_worker(task) if stage=='evaluate' else stage_worker(task,stage)
  except Exception as e:state(status='failed',task=task,stage=stage,error=str(e));event('failure',task=task,stage=stage,error=str(e));render();raise
  return
 lock=workspace_path('STAGED_FINETUNING_20260913', 'reports/pipeline.lock')
 with lock.open('x') as f:f.write(str(os.getpid()))
 try:
  state(status='starting',pipeline_pid=os.getpid());event('pipeline_started',pid=os.getpid());render()
  for task in ['bbox','seg']:
   for stage in STAGES+['evaluate']:
    with (workspace_path('STAGED_FINETUNING_20260913', f'logs/{task}_{stage}.log')).open('a',encoding='utf-8') as log:
     p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),task,stage],stdout=log,stderr=subprocess.STDOUT);state(worker_pid=p.pid);rc=p.wait()
    if rc:raise RuntimeError(f'{task} {stage} failed ({rc}); subsequent GPU jobs not started')
  state(status='complete',task=None,stage=None,completed_at=now());event('pipeline_complete');render()
 except Exception as e:state(status='failed',error=str(e));event('pipeline_failure',error=str(e));render();raise
 finally:lock.unlink(missing_ok=True)
if __name__=='__main__':main()
