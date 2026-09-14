"""간단한 목록 파일과 1차 전이학습 체크포인트의 동일 사본을 준비한다."""
# 현재 작업 디렉토리와 상관없이 공용 작업공간 경로를 찾는다.
from pathlib import Path as _WorkspacePath
import sys as _workspace_sys
_workspace_root = next(p for p in _WorkspacePath(__file__).resolve().parents if (p / 'workspace_paths.py').is_file())
_workspace_sys.path.insert(0, str(_workspace_root))
from workspace_paths import workspace_path

import shutil,json,re,collections
from pathlib import Path
import yaml,torch
from common import R,OLD,read,write,sha,now,event,state
def key(path):return re.sub(r'^images__(train|val|test)__','',Path(path).stem)
def main():
 for directory in ['data','configs','reports','checkpoints','runs','logs']:(workspace_path('STAGED_FINETUNING_20260913', directory)).mkdir(parents=True,exist_ok=True)
 v2=workspace_path('.', 'MODEL_SCALING_20260913');mstate=read(workspace_path('./MODEL_SCALING_20260913', 'reports/state.json'))
 c=torch.load(workspace_path('./MODEL_SCALING_20260913', 'runs/bbox_yolo26m_v2/weights/last.pt'),map_location='cpu',weights_only=False)
 mstate.update(status='stopped_by_user',active_task=None,detail='User requested stopping m and continuing from first transfer in staged learning.',stopped_at=now(),last_saved_epoch=c['epoch']+1,seg_started=False)
 write(workspace_path('./MODEL_SCALING_20260913', 'reports/state.json'),mstate);write(workspace_path('STAGED_FINETUNING_20260913', 'reports/stopped_m.json'),mstate);del c
 (workspace_path('./MODEL_SCALING_20260913', 'reports/pipeline.lock')).unlink(missing_ok=True)
 checkpoints={};manifest={}
 for task in ['bbox','seg']:
  comparison=read(workspace_path('KOREA_WASTE_RULES_20260913_v1', f'reports/{task}_transfer_comparison.json'));source=Path(comparison['new_checkpoint']);assert sha(source)==comparison['new_checkpoint_sha256']
  target=workspace_path('STAGED_FINETUNING_20260913', f'checkpoints/{task}_first_transfer_best.pt');shutil.copy2(source,target);assert sha(target)==sha(source)
  checkpoints[task]={'model':'YOLO26s' if task=='bbox' else 'YOLO11n-seg','source':str(source),'copy':str(target),'sha256':sha(target),'origin':'First transfer result, explicitly requested by user; not original best_train or m checkpoint.'}
  cfg=yaml.safe_load((workspace_path('./MODEL_SCALING_20260913', f'data/{task}.yaml')).read_text('utf-8'));base=(workspace_path('./MODEL_SCALING_20260913', f'data/{task}_train.txt')).read_text('utf-8').splitlines();assert len(base)==len(set(base))
  sets={split:set((workspace_path('./MODEL_SCALING_20260913', f'data/{task}_{split}.txt')).read_text('utf-8').splitlines()) for split in ['train','val','test']}
  assert not ({key(p) for p in sets['train']} & ({key(p) for p in sets['val']}|{key(p) for p in sets['test']}))
  previous=read(workspace_path('KOREA_WASTE_RULES_20260913_v1', f'finetune/{task}_selection.json'));lookup={r['path']:r for r in previous};hard=[]
  for row in previous:
   path=row['path']
   if path not in sets['train']:continue
   is_hard=row['kind']=='real_general_replay_repeat' or (row['kind'].startswith('new') and (not row['classes'] or set(row['classes'])&{'general','paper','vinyl','battery'})) or row['kind']=='packaging_repeat'
   if is_hard:hard.append(path)
  hard=sorted(set(hard));assert len(hard)<.1*len(base)
  for split in ['val','test']:
   dst=workspace_path('STAGED_FINETUNING_20260913', f'data/{task}_{split}.txt');shutil.copy2(workspace_path('./MODEL_SCALING_20260913', f'data/{task}_{split}.txt'),dst);cfg[split]=str(dst)
  stages={}
  for stage in ['stage2','stage3','final']:
   paths=base+(hard if stage=='stage3' else []);dest=workspace_path('STAGED_FINETUNING_20260913', f'data/{task}_{stage}_train.txt');dest.write_text('\n'.join(paths)+'\n',encoding='utf-8')
   dc=dict(cfg,train=str(dest));(workspace_path('STAGED_FINETUNING_20260913', f'data/{task}_{stage}.yaml')).write_text(yaml.safe_dump(dc,allow_unicode=True,sort_keys=False),encoding='utf-8')
   stages[stage]={'unique_images':len(set(paths)),'epoch_entries':len(paths),'hard_extra_exposures':len(hard) if stage=='stage3' else 0,'list_sha256':sha(dest)}
  write(workspace_path('STAGED_FINETUNING_20260913', f'data/{task}_hard_selection.json'),[lookup[p] for p in hard]);manifest[task]={'stages':stages,'all_original_own_train_included':True,'common_original_train':47054,'own_validation_test_ids_in_train':0,'source':'Verified existing 640 JPEG and converted native task labels; no images copied.'}
 common=dict(imgsz=640,epochs=100,patience=10,batch=16,nbs=64,workers=4,cache=False,amp=True,device=0,optimizer='AdamW',warmup_bias_lr=0.,warmup_epochs=1.,weight_decay=.0005,lrf=.2,cos_lr=True,close_mosaic=0,mixup=0.,cutmix=0.,copy_paste=0.,flipud=0.,fliplr=.5,seed=73,plots=False,save=True,verbose=False)
 stages={
  'stage2':dict(freeze=10,lr0=.00003,mosaic=0.,hsv_h=.005,hsv_s=.15,hsv_v=.15,degrees=3.,translate=.03,scale=.1),
  'stage3':dict(freeze=0,lr0=.00001,mosaic=.1,hsv_h=.01,hsv_s=.2,hsv_v=.2,degrees=5.,translate=.05,scale=.15),
  'final':dict(freeze=0,lr0=.000005,mosaic=0.,hsv_h=0.,hsv_s=.05,hsv_v=.05,degrees=0.,translate=.01,scale=.03),
 }
 for stage,extra in stages.items():(workspace_path('STAGED_FINETUNING_20260913', f'configs/{stage}.yaml')).write_text(yaml.safe_dump(dict(common,**extra),sort_keys=False),encoding='utf-8')
 write(workspace_path('STAGED_FINETUNING_20260913', 'reports/checkpoints.json'),checkpoints);write(workspace_path('STAGED_FINETUNING_20260913', 'reports/data_manifest.json'),manifest)
 valid=read(workspace_path('KOREA_WASTE_RULES_20260913_v1', 'reports/final_training_data_validation.json'));assert valid['passed'] and valid['all_image_metadata_removed']
 write(workspace_path('STAGED_FINETUNING_20260913', 'reports/preparation.json'),{'passed':True,'at':now(),'image_files_created':0,'previous_full_image_verification':str(workspace_path('KOREA_WASTE_RULES_20260913_v1', 'reports/final_training_data_validation.json')),'native_class_orders_preserved':True,'original_and_m_weights_preserved':True})
 state(status='prepared',sequence=['bbox_stage2','bbox_stage3','bbox_final','seg_stage2','seg_stage3','seg_final'])
 event('prepared',checkpoints=checkpoints,data_manifest=manifest)
 print(json.dumps(manifest,ensure_ascii=False))
if __name__=='__main__':main()
