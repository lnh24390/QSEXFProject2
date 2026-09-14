"""검증 단계에서 BN 버퍼만 손댄다. 학습된 파라미터는 그대로 둔다."""
# 현재 작업 디렉토리와 상관없이 공용 작업공간 경로를 찾는다.
from pathlib import Path as _WorkspacePath
import sys as _workspace_sys
_workspace_root = next(p for p in _WorkspacePath(__file__).resolve().parents if (p / 'workspace_paths.py').is_file())
_workspace_sys.path.insert(0, str(_workspace_root))
from workspace_paths import workspace_path

from pathlib import Path
import json,torch
from ultralytics import YOLO
R=workspace_path('MODEL_SCALING_20260913');OLD=workspace_path('.', 'KOREA_WASTE_RULES_20260913_v1')
def main():
 assert torch.cuda.is_available()
 base=YOLO(str(workspace_path('./KOREA_WASTE_RULES_20260913_v1', 'checkpoints/bbox_yolo26s_nobg_best.pt'))).model
 target=YOLO(str(workspace_path('MODEL_SCALING_20260913', 'runs/diagnostic_s_warmup0/weights/best.pt')))
 reference=dict(base.named_buffers());count=0
 for name,buffer in target.model.named_buffers():
  if name.endswith(('running_mean','running_var','num_batches_tracked')):
   assert name in reference and buffer.shape==reference[name].shape
   buffer.copy_(reference[name]);count+=1
 result=target.val(data=str(workspace_path('./KOREA_WASTE_RULES_20260913_v1', 'finetune/bbox.yaml')),split='val',imgsz=640,batch=16,device=0,workers=4,plots=False,verbose=False,project=str(workspace_path('MODEL_SCALING_20260913', 'runs')),name='diagnostic_restore_bn')
 report={'intervention':'Restore only BN running statistics from original checkpoint; all newly trained parameters unchanged','restored_buffers':count,'metrics':{k:float(v) for k,v in result.results_dict.items()},'original_model_val_map':.8491934413965256,'diagnostic_before_bn_restore':.66675,'limitation':'Validation-only diagnostic, not a model selected on test; not a production checkpoint.'}
 (workspace_path('MODEL_SCALING_20260913', 'reports/bn_control.json')).write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
