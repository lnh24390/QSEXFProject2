"""GPU 처리량 비교 전용이다. 여기서 생긴 옵티마이저 갱신은 전이학습 입력으로 쓰지 않는다."""
# 현재 작업 디렉토리와 상관없이 공용 작업공간 경로를 찾는다.
from pathlib import Path as _WorkspacePath
import sys as _workspace_sys
_workspace_root = next(p for p in _WorkspacePath(__file__).resolve().parents if (p / 'workspace_paths.py').is_file())
_workspace_sys.path.insert(0, str(_workspace_root))
from workspace_paths import workspace_path

from pathlib import Path
import os,json,time,random,gc,sys
import numpy as np
import torch,yaml
from PIL import Image
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.build import build_dataloader
R=workspace_path('KOREA_WASTE_RULES_20260913_v1');B=workspace_path('KOREA_WASTE_RULES_20260913_v1', 'benchmarks/mosaic_640');B.mkdir(parents=True,exist_ok=True)
NAMES={'bbox':['paper','plastic','can','glass','battery','vinyl','general'],'seg':['general','can','plastic','paper','glass','vinyl','battery']}
# 색·밝기 변환은 고정하고 나머지 기하 변환을 꺼서 Mosaic 효과만 분리한다.
BASE=dict(imgsz=640,task='detect',hsv_h=.01,hsv_s=.35,hsv_v=.30,degrees=0.,translate=0.,scale=0.,shear=0.,perspective=0.,flipud=0.,fliplr=.5,mixup=0.,copy_paste=0.,cutmix=0.,rect=False)
def prepare(task):
 src=workspace_path('KOREA_WASTE_RULES_20260913_v1', f'training_640_{task}');out=workspace_path('KOREA_WASTE_RULES_20260913_v1/benchmarks/mosaic_640', task, 'original');ip=out/'images/train';lp=out/'labels/train';ip.mkdir(parents=True,exist_ok=True);lp.mkdir(parents=True,exist_ok=True)
 paths=sorted((src/'images/train').glob('*.jpg'),key=lambda p:__import__('hashlib').sha256(p.name.encode()).hexdigest())[:512]
 for p in paths:
  if not (ip/p.name).exists():os.link(p,ip/p.name)
  (lp/(p.stem+'.txt')).write_bytes((src/'labels/train'/(p.stem+'.txt')).read_bytes())
 hyp=get_cfg(overrides=dict(BASE,mosaic=.5));ds=YOLODataset(img_path=str(ip),imgsz=640,augment=True,hyp=hyp,rect=False,cache=False,data={'names':dict(enumerate(NAMES[task])),'nc':7},task='detect' if task=='bbox' else 'segment',batch_size=16)
 off=workspace_path('KOREA_WASTE_RULES_20260913_v1/benchmarks/mosaic_640', task, 'offline');oi=off/'images/train';ol=off/'labels/train';oi.mkdir(parents=True,exist_ok=True);ol.mkdir(parents=True,exist_ok=True)
 random.seed(73);np.random.seed(73);t=time.perf_counter();records=[]
 for i in range(len(ds)):
  d=ds.transforms.transforms[0](ds.get_image_and_label(i))
  im=d['img'];inst=d['instances'];inst.convert_bbox('xywh');inst.normalize(640,640);boxes=inst.bboxes;classes=d['cls'].reshape(-1);assert im.shape[:2]==(640,640)
  name=f'offline_{i:05d}';Image.fromarray(im[:,:,::-1]).save(oi/(name+'.jpg'),quality=85,optimize=True)
  lines=[]
  for j,cl in enumerate(classes):
   coords=boxes[j] if task=='bbox' else inst.segments[j].reshape(-1)
   assert np.isfinite(coords).all() and (coords>=0).all() and (coords<=1).all()
   lines.append(str(int(cl))+' '+' '.join(f'{v:.8f}' for v in coords))
  (ol/(name+'.txt')).write_text('\n'.join(lines)+ ('\n' if lines else ''))
  records.append({'output':name,'primary_source':ds.im_files[i],'mosaic_sources':d.get('mix_labels',[]),'objects':len(lines)})
 result={'task':task,'count':len(ds),'precompute_seconds':time.perf_counter()-t,'output_image_bytes':sum(p.stat().st_size for p in oi.glob('*.jpg')),'source_pool':[str(p) for p in paths],'note':'One fixed Mosaic+identity-affine realization per primary image. All sources are train. Benchmark only, never merged into training_640.'}
 (workspace_path('KOREA_WASTE_RULES_20260913_v1/benchmarks/mosaic_640', f'{task}_precompute.json')).write_text(json.dumps(result,indent=2));return out,off

def run(task,mode):
 assert torch.cuda.is_available(),'GPU required'
 root=workspace_path('KOREA_WASTE_RULES_20260913_v1/benchmarks/mosaic_640', task, 'offline' if mode=='offline' else 'original');prob=.5 if mode=='native' else 0.;taskname='detect' if task=='bbox' else 'segment';hyp=get_cfg(overrides=dict(BASE,task=taskname,mosaic=prob));ds=YOLODataset(img_path=str(root/'images/train'),imgsz=640,augment=True,hyp=hyp,rect=False,cache=False,data={'names':dict(enumerate(NAMES[task])),'nc':7},task=taskname,batch_size=16)
 loader=build_dataloader(ds,batch=16,workers=4,shuffle=True,rank=-1);it=iter(loader)
 ck=workspace_path('KOREA_WASTE_RULES_20260913_v1', 'checkpoints', 'bbox_yolo26s_nobg_best.pt' if task=='bbox' else 'seg_yolo11n_best.pt');model=YOLO(str(ck)).model.float().to('cuda').train();model.args=hyp
 for p in model.parameters():p.requires_grad_(True)
 optim=torch.optim.SGD(model.parameters(),lr=.0001,momentum=.9);scaler=torch.amp.GradScaler('cuda');results=[]
 for epoch in range(3):
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();total=0;losses=[]
  for step in range(len(loader)):
   try:batch=next(it)
   except StopIteration:it=iter(loader);batch=next(it)
   for k,v in batch.items():
    if isinstance(v,torch.Tensor):batch[k]=v.to('cuda',non_blocking=True)
   batch['img']=batch['img'].float()/255.;optim.zero_grad(set_to_none=True)
   with torch.autocast(device_type='cuda',dtype=torch.float16):loss,items=model(batch);loss=loss.sum()
   assert torch.isfinite(loss),'non-finite loss'
   scaler.scale(loss).backward();scaler.step(optim);scaler.update();total+=len(batch['img']);losses.append(float(loss.detach()))
  torch.cuda.synchronize();elapsed=time.perf_counter()-start;row={'epoch':epoch+1,'warmup':epoch==0,'seconds':elapsed,'images':total,'images_per_second':total/elapsed,'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'mean_loss':sum(losses)/len(losses)};results.append(row);print(task,mode,row,flush=True)
 report={'task':task,'mode':mode,'checkpoint':str(ck),'gpu':torch.cuda.get_device_name(),'batch':16,'workers':4,'amp':True,'imgsz':640,'epochs':results,'mean_images_per_second':sum(x['images'] for x in results[1:])/sum(x['seconds'] for x in results[1:]),'limitation':'Short training-step throughput, no validation/checkpoint saving overhead; no accuracy conclusion; scratch updated weights discarded.'}
 (workspace_path('KOREA_WASTE_RULES_20260913_v1/benchmarks/mosaic_640', f'{task}_{mode}_speed.json')).write_text(json.dumps(report,indent=2));del model,optim,loader,ds;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':
 task=sys.argv[2] if len(sys.argv)>2 else 'bbox'
 if sys.argv[1]=='prepare':prepare(task)
 else:run(task,sys.argv[1])
