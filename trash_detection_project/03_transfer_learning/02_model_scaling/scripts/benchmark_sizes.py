"""실제 640 학습 이미지와 7클래스 목표로 GPU 를 순차로 재는 소규모 벤치마크."""
# 현재 작업 디렉토리와 상관없이 공용 작업공간 경로를 찾는다.
from pathlib import Path as _WorkspacePath
import sys as _workspace_sys
_workspace_root = next(p for p in _WorkspacePath(__file__).resolve().parents if (p / 'workspace_paths.py').is_file())
_workspace_sys.path.insert(0, str(_workspace_root))
from workspace_paths import workspace_path

from pathlib import Path
import sys,time,json,copy,gc,random
import numpy as np
import torch,yaml
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel,SegmentationModel
from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.build import build_dataloader
R=workspace_path('MODEL_SCALING_20260913');P=workspace_path('')
def main(task,size,batch):
 assert torch.cuda.is_available()
 source=workspace_path('', f'yolo26{size}{"-seg" if task=="seg" else ""}.pt')
 if not source.is_file():raise FileNotFoundError(f"Optional pretrained weights were cleaned up; supply this model explicitly before benchmarking: {source}")
 torch.manual_seed(73);random.seed(73);np.random.seed(73)
 cfg=yaml.safe_load((workspace_path('MODEL_SCALING_20260913', 'configs/larger_model.yaml')).read_text());cfg.update(task='detect' if task=='bbox' else 'segment',batch=batch,augmentations=[])
 hyp=get_cfg(overrides=cfg);data=yaml.safe_load((workspace_path('MODEL_SCALING_20260913', f'data/{task}.yaml')).read_text());data['nc']=7
 ds=YOLODataset(img_path=str(workspace_path('MODEL_SCALING_20260913', f'data/{task}_benchmark.txt')),imgsz=640,augment=True,hyp=hyp,rect=False,cache=False,data=data,task=hyp.task,batch_size=batch)
 loader=build_dataloader(ds,batch=batch,workers=4,shuffle=True,rank=-1);it=iter(loader)
 pretrained=YOLO(str(source)).model.float();cls=DetectionModel if task=='bbox' else SegmentationModel
 model=cls(copy.deepcopy(pretrained.yaml),nc=7,verbose=False);model.names=data['names'];model.load(pretrained,verbose=False);del pretrained
 model.args=hyp;model=model.float().to('cuda').train()
 for p in model.parameters():p.requires_grad_(True)
 optimizer=torch.optim.AdamW(model.parameters(),lr=.0003);scaler=torch.amp.GradScaler('cuda');rows=[]
 for phase,steps in [('warmup',8),('measured',24)]:
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();images=0
  for _ in range(steps):
   try:b=next(it)
   except StopIteration:it=iter(loader);b=next(it)
   b={k:v.to('cuda',non_blocking=True) if isinstance(v,torch.Tensor) else v for k,v in b.items()};b['img']=b['img'].float()/255
   optimizer.zero_grad(set_to_none=True)
   with torch.autocast(device_type='cuda',dtype=torch.float16):loss,_=model(b);loss=loss.sum()
   assert torch.isfinite(loss)
   scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),10);scaler.step(optimizer);scaler.update();images+=len(b['img'])
  torch.cuda.synchronize();elapsed=time.perf_counter()-start;rows.append({'phase':phase,'steps':steps,'seconds':elapsed,'images_per_second':images/elapsed,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20})
 result={'task':task,'scale':size,'batch':batch,'imgsz':640,'gpu':torch.cuda.get_device_name(),'gpu_total_mib':torch.cuda.get_device_properties(0).total_memory/2**20,'train_parameters':sum(p.numel() for p in model.parameters()),'rows':rows,'limitation':'Short AdamW training-step benchmark; no validation/EMA/checkpoint overhead, not an accuracy comparison; benchmark weights discarded.'}
 (workspace_path('MODEL_SCALING_20260913', f'reports/benchmark_{task}_{size}_b{batch}.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main(sys.argv[1],sys.argv[2],int(sys.argv[3]))
