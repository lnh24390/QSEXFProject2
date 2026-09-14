"""조건을 맞춘 1에포크 진단이다. 더 큰 모델의 초기 가중치로 쓰지 않는다."""
# 현재 작업 디렉토리와 상관없이 공용 작업공간 경로를 찾는다.
from pathlib import Path as _WorkspacePath
import sys as _workspace_sys
_workspace_root = next(p for p in _WorkspacePath(__file__).resolve().parents if (p / 'workspace_paths.py').is_file())
_workspace_sys.path.insert(0, str(_workspace_root))
from workspace_paths import workspace_path

from pathlib import Path
import sys,json,time
import torch,yaml
from ultralytics import YOLO
R=workspace_path('MODEL_SCALING_20260913');OLD=workspace_path('.', 'KOREA_WASTE_RULES_20260913_v1')
sys.path.insert(0,str(workspace_path('./KOREA_WASTE_RULES_20260913_v1', 'scripts')))
from augmentation_runtime import training_transforms
def main():
 assert torch.cuda.is_available()
 cfg=yaml.safe_load((workspace_path('./KOREA_WASTE_RULES_20260913_v1', 'configs/finetune_640_native.yaml')).read_text());cfg['epochs']=20;cfg['warmup_bias_lr']=0.
 # 원래 첫 에포크 설정을 그대로 맞춘다. 원래의 20에포크 기준도 포함한다.
 model=YOLO(str(workspace_path('./KOREA_WASTE_RULES_20260913_v1', 'checkpoints/bbox_yolo26s_nobg_best.pt')));rows=[]
 def begin(t):t._diag_start=time.perf_counter()
 def end(t):
  if rows:return
  rows.append({'epoch':t.epoch+1,'seconds':time.perf_counter()-t._diag_start,'metrics':{k:float(v) for k,v in t.metrics.items()}})
  t.stop=True
  (workspace_path('MODEL_SCALING_20260913', 'reports/warmup_control.json')).write_text(json.dumps({'changed_only':'warmup_bias_lr 0.1 -> 0.0; same original first-epoch recipe, source checkpoint and data','result':rows[0],'original_first_epoch_map':.69559,'original_unmodified_val_map':.8491934413965256,'limitation':'Single-seed, single-epoch comparison; does not isolate other failure factors or establish final accuracy.'},indent=2))
 model.add_callback('on_train_epoch_start',begin);model.add_callback('on_fit_epoch_end',end)
 model.train(data=str(workspace_path('./KOREA_WASTE_RULES_20260913_v1', 'finetune/bbox.yaml')),project=str(workspace_path('MODEL_SCALING_20260913', 'runs')),name='diagnostic_s_warmup0',plots=False,save=True,verbose=False,augmentations=training_transforms(),**cfg)
if __name__=='__main__':main()
