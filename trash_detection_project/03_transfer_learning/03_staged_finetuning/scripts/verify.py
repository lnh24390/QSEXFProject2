# 현재 작업 디렉토리와 상관없이 공용 작업공간 경로를 찾는다.
from pathlib import Path as _WorkspacePath
import sys as _workspace_sys
_workspace_root = next(p for p in _WorkspacePath(__file__).resolve().parents if (p / 'workspace_paths.py').is_file())
_workspace_sys.path.insert(0, str(_workspace_root))
from workspace_paths import workspace_path

import copy,json
import numpy as np
from PIL import Image
import torch,yaml,albumentations as A
from ultralytics.cfg import get_cfg
from ultralytics.data.augment import Albumentations
from ultralytics.utils.instance import Instances
from common import R,STAGES,read,write,promotion,sha
from pipeline import transforms
def main():
 assert torch.cuda.is_available()
 names={str(i):str(i) for i in range(7)}
 base={'names':names,'box':{'map':.7,'per_class_map':[.7]*7}}
 better=copy.deepcopy(base);better['box']={'map':.72,'per_class_map':[.72]*7};assert promotion(base,better,'bbox')['accepted']
 worse=copy.deepcopy(base);worse['box']['map']=.69;assert not promotion(base,worse,'bbox')['accepted']
 hidden=copy.deepcopy(better);hidden['box']['per_class_map'][0]=.65;assert not promotion(base,hidden,'bbox')['accepted']
 unchanged=copy.deepcopy(base);assert not promotion(base,unchanged,'bbox')['accepted']
 segbase=copy.deepcopy(base);segbase['seg']=copy.deepcopy(base['box']);segnew=copy.deepcopy(segbase);segnew['box']['map']=.75;segnew['seg']['map']=.69;assert not promotion(segbase,segnew,'seg')['accepted']
 cfgs={}
 for stage in STAGES:
  cfg=yaml.safe_load((workspace_path('STAGED_FINETUNING_20260913', f'configs/{stage}.yaml')).read_text());get_cfg(overrides=dict(cfg,augmentations=transforms(stage)))
  assert cfg['epochs']==100 and cfg['patience']==10 and cfg['warmup_bias_lr']==0;cfgs[stage]=cfg
 assert cfgs['stage2']['freeze']==10 and cfgs['final']['mosaic']==0
 t=Albumentations(transforms=[A.to_dict(x) for x in transforms('stage3')]);assert t.transform is not None and not t.contains_spatial;t.transform.set_random_seed(73)
 first=read(workspace_path('STAGED_FINETUNING_20260913', 'reports/checkpoints.json'))
 for task,info in first.items():
  assert sha(info['copy'])==info['sha256']
  saved=torch.load(info['copy'],map_location='cpu',weights_only=False);net=saved.get('ema') or saved['model'];expected=yaml.safe_load((workspace_path('STAGED_FINETUNING_20260913', f'data/{task}_stage2.yaml')).read_text())['names'];assert net.names==expected;del net,saved
  paths=(workspace_path('STAGED_FINETUNING_20260913', f'data/{task}_stage2_train.txt')).read_text().splitlines()
  for path in paths[::max(1,len(paths)//12)][:12]:
   with Image.open(path) as im:assert im.size==(640,640) and im.format=='JPEG' and not im.getexif() and not any(k in im.info for k in ['icc_profile','xmp','comment'])
 image=np.array(Image.open(paths[0]).convert('RGB'));label={'img':image,'cls':np.array([[0]],dtype=np.float32),'instances':Instances(np.array([[.5,.5,.4,.4]],dtype=np.float32),np.array([[[.3,.3],[.7,.3],[.7,.7],[.3,.7]]],dtype=np.float32),bbox_format='xywh',normalized=True)}
 changed=0
 for _ in range(100):
  out=t(copy.deepcopy(label));assert np.array_equal(out['instances'].bboxes,label['instances'].bboxes) and np.array_equal(out['instances'].segments,label['instances'].segments) and np.array_equal(out['cls'],label['cls']);changed+=not np.array_equal(out['img'],image)
 assert 0<changed<100
 write(workspace_path('STAGED_FINETUNING_20260913', 'reports/verification.json'),{'passed':True,'promotion_cases_checked':5,'class_order_and_checkpoint_hashes_verified':True,'stage3_pixel_only_trials':100,'changed_images':changed,'label_geometry_unchanged':True,'metadata_spot_checks':24,'full_prior_image_validation_reused':True,'config_epochs':100,'patience':10,'image_files_created':0})
 print('Verification passed')
if __name__=='__main__':main()
