"""설치된 YOLO 증강을 실제로 돌려, 학습 없이 변환된 라벨을 확인한다."""
# 현재 작업 디렉토리와 상관없이 공용 작업공간 경로를 찾는다.
from pathlib import Path as _WorkspacePath
import sys as _workspace_sys
_workspace_root = next(p for p in _WorkspacePath(__file__).resolve().parents if (p / 'workspace_paths.py').is_file())
_workspace_sys.path.insert(0, str(_workspace_root))
from workspace_paths import workspace_path

from pathlib import Path
import os,json,random
import numpy as np
import torch,yaml
from PIL import Image,ImageDraw
from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset
ROOT=workspace_path('KOREA_WASTE_RULES_20260913_v1');out=workspace_path('KOREA_WASTE_RULES_20260913_v1', 'reports/augmentation_preview');base=workspace_path('KOREA_WASTE_RULES_20260913_v1/reports/augmentation_preview', 'sample_dataset');ip=workspace_path('KOREA_WASTE_RULES_20260913_v1/reports/augmentation_preview/sample_dataset', 'images/train');lp=workspace_path('KOREA_WASTE_RULES_20260913_v1/reports/augmentation_preview/sample_dataset', 'labels/train');ip.mkdir(parents=True,exist_ok=True);lp.mkdir(parents=True,exist_ok=True)
files=sorted((workspace_path('KOREA_WASTE_RULES_20260913_v1', 'staged_bbox/images/train')).glob('warp_*.jpg'))[::100][:24]+sorted((workspace_path('KOREA_WASTE_RULES_20260913_v1', 'staged_bbox/images/train')).glob('new_taco_*'))[:8]
for p in files:
 if not (workspace_path('KOREA_WASTE_RULES_20260913_v1/reports/augmentation_preview/sample_dataset/images/train', p.name)).exists():os.link(p,workspace_path('KOREA_WASTE_RULES_20260913_v1/reports/augmentation_preview/sample_dataset/images/train', p.name))
 (workspace_path('KOREA_WASTE_RULES_20260913_v1/reports/augmentation_preview/sample_dataset/labels/train', p.stem+'.txt')).write_text((workspace_path('KOREA_WASTE_RULES_20260913_v1', 'staged_bbox/labels/train', p.stem+'.txt')).read_text(),encoding='utf-8')
cfg=yaml.safe_load((workspace_path('KOREA_WASTE_RULES_20260913_v1', 'configs/augmentation_moderate.yaml')).read_text());names=['paper','plastic','can','glass','battery','vinyl','general'];checks={}
random.seed(19);np.random.seed(19);torch.manual_seed(19)
for mode,mosaic in [('no_mosaic',0.0),('mosaic_demo',1.0)]:
 h=get_cfg(overrides=dict(cfg,mosaic=mosaic));ds=YOLODataset(img_path=str(ip),imgsz=640,augment=True,hyp=h,rect=False,cache=False,data={'names':dict(enumerate(names)),'nc':7},task='detect',batch_size=4)
 sheet=Image.new('RGB',(1280,1330),'white');draw=ImageDraw.Draw(sheet);records=[]
 for i in range(4):
  d=ds[i];im=Image.fromarray(d['img'].permute(1,2,0).numpy());b=d['bboxes'].numpy();c=d['cls'].numpy().reshape(-1)
  assert np.isfinite(b).all() and (b>=0).all() and (b<=1).all() and len(b)==len(c)
  dr=ImageDraw.Draw(im)
  for (x,y,w,h),cl in zip(b,c):
   assert 0<=cl<7 and w>0 and h>0
   x1,y1,x2,y2=(x-w/2)*640,(y-h/2)*640,(x+w/2)*640,(y+h/2)*640;dr.rectangle((x1,y1,x2,y2),outline='red',width=2);dr.text((max(0,x1),max(0,y1)),names[int(cl)],fill='yellow',stroke_fill='black',stroke_width=1)
  xx=(i%2)*640;yy=(i//2)*665;sheet.paste(im,(xx,yy));draw.text((xx+8,yy+642),f'{mode} example {i+1}; objects={len(b)}',fill='black');records.append({'example':i+1,'objects':len(b),'coordinates_valid':True})
 sheet.save(workspace_path('KOREA_WASTE_RULES_20260913_v1/reports/augmentation_preview', mode+'.jpg'));checks[mode]=records
checks['note']='Preview forces mosaic probability 1 only to show examples. Training candidate uses 0.5. No optimizer or model training executed.'
(workspace_path('KOREA_WASTE_RULES_20260913_v1/reports/augmentation_preview', 'validation.json')).write_text(json.dumps(checks,indent=2),encoding='utf-8');print('Actual augmentation preview label checks passed')
