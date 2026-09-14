"""V2 데이터를 목록 파일로만 준비한다. 기존 이미지와 모델은 그대로 둔다."""
# 현재 작업 디렉토리와 상관없이 공용 작업공간 경로를 찾는다.
from pathlib import Path as _WorkspacePath
import sys as _workspace_sys
_workspace_root = next(p for p in _WorkspacePath(__file__).resolve().parents if (p / 'workspace_paths.py').is_file())
_workspace_sys.path.insert(0, str(_workspace_root))
from workspace_paths import workspace_path

from pathlib import Path
import json,hashlib,collections,datetime
import yaml
P=workspace_path('');R=workspace_path('MODEL_SCALING_20260913');OLD=workspace_path('', 'KOREA_WASTE_RULES_20260913_v1')
def main():
 for name in ['data','configs','reports','checkpoints','runs','logs']: (workspace_path('MODEL_SCALING_20260913', name)).mkdir(exist_ok=True)
 report={}
 for task in ['bbox','seg']:
  src=workspace_path('KOREA_WASTE_RULES_20260913_v1', f'training_640_{task}');cfg=yaml.safe_load((workspace_path('KOREA_WASTE_RULES_20260913_v1', f'finetune/{task}.yaml')).read_text('utf-8'));counts={};classes=collections.Counter()
  for split in ['train','val','test']:
   paths=sorted((src/f'images/{split}').glob('*.jpg'));assert paths
   for p in paths:
    lab=src/f'labels/{split}/{p.stem}.txt';assert lab.exists()
    if split=='train':
     for line in lab.read_text().splitlines():classes[cfg['names'][int(line.split()[0])]]+=1
   dest=workspace_path('MODEL_SCALING_20260913', f'data/{task}_{split}.txt');dest.write_text('\n'.join(str(p) for p in paths)+'\n',encoding='utf-8');cfg[split]=str(dest);counts[split]=len(paths)
   if split=='train':
    sample=sorted(paths,key=lambda p:hashlib.sha256(p.name.encode()).hexdigest())[:256]
    (workspace_path('MODEL_SCALING_20260913', f'data/{task}_benchmark.txt')).write_text('\n'.join(str(p) for p in sample)+'\n',encoding='utf-8')
  (workspace_path('MODEL_SCALING_20260913', f'data/{task}.yaml')).write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf-8')
  report[task]={'counts':counts,'train_class_instances':dict(classes),'duplicates_in_epoch':0,'image_storage':'Existing verified 640 JPEG files; paths only, no image duplication.'}
 report['original_mix']={'common_old_train':47054,'all_old_train_restored':True,'note':'Own task-specific old train included for coverage; never import the other task val/test. Common original images remain the majority.'}
 (workspace_path('MODEL_SCALING_20260913', 'reports/data_manifest.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 comparisons={t:json.loads((workspace_path('KOREA_WASTE_RULES_20260913_v1', f'reports/{t}_transfer_comparison.json')).read_text('utf-8')) for t in ['bbox','seg']}
 audit={'created':datetime.datetime.now().isoformat(),'prior_results':comparisons,'confirmed_configuration_problem':{'lr0':.0001,'warmup_bias_lr':.1,'ratio':1000,'optimizer':'explicit AdamW','evidence':'Previous args.yaml and local BaseTrainer.build_optimizer: only optimizer=auto resets warmup_bias_lr to 0.','causality':'Strong candidate, not isolated causal proof.'},'other_risks':['Only 6444 of roughly 47000 old labeled train images replayed','Multiple simultaneous changes to data, Mosaic, blur, optimizer and schedule','No original-model validation floor: best new epoch selected even when below original','Early stopping before close_mosaic could prevent clean final tuning','Prior test metrics inspected: no hyperparameter selection against those test results']}
 (workspace_path('MODEL_SCALING_20260913', 'reports/failure_audit.json')).write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
 cfg={'imgsz':640,'epochs':100,'patience':10,'batch':8,'nbs':64,'workers':4,'cache':False,'amp':True,'device':0,'optimizer':'AdamW','lr0':.0003,'lrf':.1,'warmup_epochs':3.,'warmup_bias_lr':0.,'weight_decay':.0005,'cos_lr':True,'mosaic':.2,'close_mosaic':10,'mixup':0.,'cutmix':0.,'copy_paste':0.,'hsv_h':.01,'hsv_s':.25,'hsv_v':.2,'degrees':5.,'translate':.05,'scale':.15,'fliplr':.5,'flipud':0.,'seed':73,'plots':False,'save':True,'verbose':False}
 (workspace_path('MODEL_SCALING_20260913', 'configs/larger_model.yaml')).write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
 print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
