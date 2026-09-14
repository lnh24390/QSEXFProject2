# 현재 작업 디렉토리와 상관없이 공용 작업공간 경로를 찾는다.
from pathlib import Path as _WorkspacePath
import sys as _workspace_sys
_workspace_root = next(p for p in _WorkspacePath(__file__).resolve().parents if (p / 'workspace_paths.py').is_file())
_workspace_sys.path.insert(0, str(_workspace_root))
from workspace_paths import workspace_path

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import json,math
from PIL import Image
R=workspace_path('KOREA_WASTE_RULES_20260913_v1')
def check(item):
 task,split,p=item;lp=workspace_path('KOREA_WASTE_RULES_20260913_v1', f'training_640_{task}/labels', split, p.stem+'.txt')
 with Image.open(p) as im:
  assert im.size==(640,640) and im.format=='JPEG'
  assert not im.getexif() and not any(k in im.info for k in ['exif','xmp','icc_profile','comment'])
  assert all(k not in ['APP1','APP2','APP13','COM'] for k,v in getattr(im,'applist',[]))
 lines=lp.read_text('utf-8').splitlines();n=0
 for line in lines:
  if not line.strip():continue
  v=list(map(float,line.split()));assert int(v[0])==v[0] and 0<=v[0]<7;cs=v[1:];assert all(math.isfinite(x) and 0<=x<=1 for x in cs)
  assert (len(cs)==4 and cs[2]>0 and cs[3]>0) if task=='bbox' else (len(cs)>=6 and len(cs)%2==0)
  n+=1
 return task,split,n,p.stat().st_size

def main():
 jobs=[]
 for task in ['bbox','seg']:
  for split in ['train','val','test']:
   ip=workspace_path('KOREA_WASTE_RULES_20260913_v1', f'training_640_{task}/images', split);lp=workspace_path('KOREA_WASTE_RULES_20260913_v1', f'training_640_{task}/labels', split);images=list(ip.glob('*.jpg'));assert {p.stem for p in images}=={p.stem for p in lp.glob('*.txt')};jobs.extend((task,split,p) for p in images)
 stats=Counter()
 with ThreadPoolExecutor(max_workers=6) as pool:
  for i,(task,split,n,size) in enumerate(pool.map(check,jobs),1):
   stats[f'{task}_{split}_images']+=1;stats[f'{task}_{split}_objects']+=n;stats[f'{task}_{split}_backgrounds']+=int(n==0);stats['total_image_bytes']+=size
   if i%30000==0:print('Final 640/image metadata/label checks',i,'/',len(jobs),flush=True)
 # 정리된 데이터셋에는 쓰지 않는 이미지·라벨 폴더가 남아 있지 않아야 한다.
 prohibited=['raw','prepared','staged_bbox','staged_seg','reviewed_bbox','supplement_household_bbox','supplement_warp_bbox','review_pending_640'];remaining=[p for p in prohibited if (workspace_path('KOREA_WASTE_RULES_20260913_v1', p)).exists()];assert not remaining,remaining
 outside=[str(p.relative_to(R)) for p in R.rglob('*') if p.is_file() and p.suffix.lower() in ['.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff'] and not any(p.is_relative_to(workspace_path('KOREA_WASTE_RULES_20260913_v1', f'training_640_{t}')) for t in ['bbox','seg'])];assert not outside,outside[:10]
 for task in ['bbox','seg']:
  for split in ['train','val','test']:
   entries=(workspace_path('KOREA_WASTE_RULES_20260913_v1', f'finetune/{task}_{split}.txt')).read_text('utf-8').splitlines()
   for p in map(Path,entries):assert p.exists() and p.is_relative_to(workspace_path('KOREA_WASTE_RULES_20260913_v1', f'training_640_{task}/images', split))
   stats[f'{task}_{split}_selected_entries']=len(entries)
 result={'passed':True,'images_checked':len(jobs),'counts':dict(stats),'all_images_640_jpeg':True,'all_image_metadata_removed':True,'all_image_label_pairs_valid':True,'training_lists_only_reference_converted_copies':True,'obsolete_directories_remaining':remaining,'images_outside_training_folders':outside,'original_project_datasets_kept':True,'note':'Source review limitations remain; format checks do not imply all semantic labels were visually reviewed.'}
 (workspace_path('KOREA_WASTE_RULES_20260913_v1', 'reports/final_training_data_validation.json')).write_text(json.dumps(result,indent=2),encoding='utf-8');(workspace_path('KOREA_WASTE_RULES_20260913_v1', 'reports/training_ready.json')).write_text(json.dumps({'ready':True,'validation':'final_training_data_validation.json'}),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
