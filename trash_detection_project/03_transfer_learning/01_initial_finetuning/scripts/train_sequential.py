"""복사한 초기 학습 체크포인트에서 GPU 로만 순차 전이학습한다."""
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

from pathlib import Path
import json,sys,time,datetime,hashlib,subprocess,os
import torch,yaml
from ultralytics import YOLO
from augmentation_runtime import training_transforms
from extended_trainers import ExtendedDetectionTrainer, ExtendedSegmentationTrainer
R=workspace_path('KOREA_WASTE_RULES_20260913_v1')
def now():return datetime.datetime.now().isoformat(timespec='seconds')
def state(**kw):
 p=workspace_path('KOREA_WASTE_RULES_20260913_v1', 'reports/finetune_state.json');old=json.loads(p.read_text('utf-8')) if p.exists() else {};old.update(kw,updated_at=now());tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(old,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)

def worker(task):
 assert json.loads((workspace_path('KOREA_WASTE_RULES_20260913_v1', 'reports/training_ready.json')).read_text('utf-8'))['ready'], 'Final readiness gate not passed'
 assert torch.cuda.is_available(),'GPU required';validation=json.loads((workspace_path('KOREA_WASTE_RULES_20260913_v1', 'reports/validation.json')).read_text('utf-8'));resize=json.loads((workspace_path('KOREA_WASTE_RULES_20260913_v1', 'reports/resize_640_validation.json')).read_text());assert validation['passed'] and resize['all_images_verified_640'] and resize['metadata_removed_verified']
 cp=json.loads((workspace_path('KOREA_WASTE_RULES_20260913_v1', 'reports/checkpoint_copies.json')).read_text());which='bbox_yolo26s_nobg' if task=='bbox' else 'seg_yolo11n';entry=next(x for x in cp if x['run']==which);ck=Path(entry['copy']);assert hashlib.file_digest(ck.open('rb'),'sha256').hexdigest()==entry['sha256']
 data=workspace_path('KOREA_WASTE_RULES_20260913_v1', 'finetune', f'{task}.yaml');cfg=yaml.safe_load((workspace_path('KOREA_WASTE_RULES_20260913_v1', 'configs/finetune_640_native.yaml')).read_text('utf-8'));runname=task+'_transfer_640';project=workspace_path('KOREA_WASTE_RULES_20260913_v1', 'finetune/runs');out=project/runname
 last=out/'weights/last.pt';resuming=last.exists()
 assert not (workspace_path('KOREA_WASTE_RULES_20260913_v1', f'reports/{task}_transfer_comparison.json')).exists(),'Refuse to retrain a completed run'
 if resuming:
  saved=torch.load(last,map_location='cpu',weights_only=False);assert saved.get('optimizer') is not None and 0<=saved['epoch']<cfg['epochs']-1
  completed_epoch=saved['epoch']+1;del saved
  state(status='resuming',active_task=task,detail=f'{task}: resume saved epoch {completed_epoch}, extend ceiling to {cfg["epochs"]}')
 else:
  assert not (out/'weights/best.pt').exists(),'Existing result has no resumable last.pt'
  state(status='baseline_validation',active_task=task,detail=f'{task}: copied original checkpoint baseline on own validation set',started_at=now())
  original=YOLO(str(ck));assert list(original.names.values())==entry['names'];base=evaluate_model(ck, data, model=original, split='val', task=task, source='01_initial_finetuning', name=task+'_baseline_val')
  (workspace_path('KOREA_WASTE_RULES_20260913_v1', f'reports/{task}_baseline_val.json')).write_text(json.dumps(summary(base),ensure_ascii=False,indent=2),encoding='utf-8');del original;torch.cuda.empty_cache()
 model=YOLO(str(last if resuming else ck));progress=workspace_path('KOREA_WASTE_RULES_20260913_v1', f'reports/{task}_epoch_progress.json');times=json.loads(progress.read_text()) if resuming and progress.exists() else []
 if resuming:times=[x for x in times if x['epoch']<=completed_epoch]
 def restore_early_stop(trainer):
  if resuming:
   for row in times:
    metrics=row['metrics'];fitness=metrics['metrics/mAP50-95(B)']+metrics.get('metrics/mAP50-95(M)',0)
    trainer.stopper(row['epoch'],fitness)
   assert trainer.epochs==cfg['epochs'] and trainer.start_epoch==completed_epoch
 model.add_callback('on_train_start',restore_early_stop)
 def begin(trainer):
  trainer._measured_start=time.perf_counter();state(status='training',active_task=task,epoch=trainer.epoch+1,max_epochs=trainer.epochs,detail=f'{task}: training from copied best.pt; sequential GPU execution')
 def end(trainer):
  if not hasattr(trainer, '_measured_start'):return
  started=trainer._measured_start;del trainer._measured_start
  elapsed=time.perf_counter()-started;metrics={k:float(v) for k,v in trainer.metrics.items()};row={'epoch':trainer.epoch+1,'seconds_including_validation':elapsed,'metrics':metrics,'best_fitness':float(trainer.best_fitness) if trainer.best_fitness is not None else None};times.append(row)
  (workspace_path('KOREA_WASTE_RULES_20260913_v1', f'reports/{task}_epoch_progress.json')).write_text(json.dumps(times,indent=2),encoding='utf-8');state(status='training',active_task=task,epoch=trainer.epoch+1,max_epochs=trainer.epochs,last_epoch_seconds=elapsed,metrics=metrics,detail=f'{task}: epoch completed; validation metrics recorded')
 model.add_callback('on_train_epoch_start',begin);model.add_callback('on_fit_epoch_end',end)
 cfg['resume']=str(last) if resuming else False
 trainer_class=ExtendedDetectionTrainer if task=='bbox' else ExtendedSegmentationTrainer
 model.train(trainer=trainer_class,data=str(data),project=str(project),name=runname,exist_ok=resuming,plots=False,save=True,save_period=-1,verbose=False,augmentations=training_transforms(),**cfg)
 save_training_summary(out, '01_initial_finetuning')
 best=out/'weights/best.pt';assert best.exists();state(status='evaluating',active_task=task,detail=f'{task}: comparing original and transferred weights on own fixed test set')
 comparison={}
 for name,path in [('baseline',ck),('finetuned',best)]:
  net=YOLO(str(path));result=evaluate_model(path, data, model=net, split='test', task=task, source='01_initial_finetuning', name=task+'_'+name+'_test');comparison[name]=summary(result);del net;torch.cuda.empty_cache()
 comparison.update(original_checkpoint=str(ck),new_checkpoint=str(best),new_checkpoint_sha256=hashlib.file_digest(best.open('rb'),'sha256').hexdigest(),completed_at=now(),note='Same own-task test images and corrected 640 labels; no deployment or replacement of old best_train files.')
 (workspace_path('KOREA_WASTE_RULES_20260913_v1', f'reports/{task}_transfer_comparison.json')).write_text(json.dumps(comparison,ensure_ascii=False,indent=2),encoding='utf-8');state(status='task_complete',active_task=task,detail=f'{task}: training and baseline comparison complete')

def main():
 if len(sys.argv)>1:
  try:worker(sys.argv[1])
  except Exception as e:state(status='failed',active_task=sys.argv[1],detail=str(e));raise
  return
 lock=workspace_path('KOREA_WASTE_RULES_20260913_v1', 'finetune/pipeline.lock')
 with lock.open('x',encoding='utf-8') as f:f.write(str(os.getpid()))
 try:
  state(status='starting',detail='Sequential bbox then seg pipeline',pipeline_pid=os.getpid())
  for task in ['bbox','seg']:
   with (workspace_path('KOREA_WASTE_RULES_20260913_v1', f'finetune/{task}_training.log')).open('a',encoding='utf-8') as log:
    p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),task],stdout=log,stderr=subprocess.STDOUT);state(worker_pid=p.pid);rc=p.wait()
   subprocess.run([sys.executable,str(workspace_path('KOREA_WASTE_RULES_20260913_v1', 'scripts/write_progress_report.py'))],check=False)
   if rc:raise RuntimeError(f'{task} failed, exit {rc}; next model will not start')
  state(status='complete',active_task=None,detail='Both models trained sequentially and compared to original checkpoints',completed_at=now())
 except Exception as e:state(status='failed',detail=str(e));raise
 finally:
  lock.unlink(missing_ok=True);subprocess.run([sys.executable,str(workspace_path('KOREA_WASTE_RULES_20260913_v1', 'scripts/write_progress_report.py'))],check=False)
if __name__=='__main__':main()
