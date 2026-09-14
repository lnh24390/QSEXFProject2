"""기존 TACO 이미지에 한해 보수적으로 중복 유사본을 걸러낸다."""
# 현재 작업 디렉토리와 상관없이 공용 작업공간 경로를 찾는다.
from pathlib import Path as _WorkspacePath
import sys as _workspace_sys
_workspace_root = next(p for p in _WorkspacePath(__file__).resolve().parents if (p / 'workspace_paths.py').is_file())
_workspace_sys.path.insert(0, str(_workspace_root))
from workspace_paths import workspace_path

from pathlib import Path
import json
from PIL import Image,ImageOps
from concurrent.futures import ThreadPoolExecutor
ROOT=workspace_path('KOREA_WASTE_RULES_20260913_v1');OLD=workspace_path('.', 'YOLO_7CLASS_10000_PER_CLASS_20260909')
src=json.loads((workspace_path('KOREA_WASTE_RULES_20260913_v1', 'sources/taco/annotations.json')).read_text('utf-8'));dims={i['id']:(i['width'],i['height']) for i in src['images']}
def sig(job):
 path,tid,split,crop=job
 with Image.open(path) as p:
  im=ImageOps.exif_transpose(p).convert('L')
  if crop:
   W,H=dims[tid];m=max(W,H);w=im.width*W/m;h=im.height*H/m;im=im.crop(((im.width-w)/2,(im.height-h)/2,(im.width+w)/2,(im.height+h)/2))
  im=im.resize((9,8));v=list(im.getdata());bits=0
  for y in range(8):
   for x in range(8):bits=(bits<<1)|int(v[y*9+x]>v[y*9+x+1])
 return tid,split,bits,str(path)
jobs=[(p,int(p.stem.rsplit('_',1)[1]),p.parent.name,True) for p in OLD.glob('images/*/s3_taco_*.jpg')]
with ThreadPoolExecutor(max_workers=4) as ex:old=list(ex.map(sig,jobs))
rows=json.loads((workspace_path('KOREA_WASTE_RULES_20260913_v1', 'sources/taco/download_manifest.json')).read_text('utf-8'));jobs=[(workspace_path('KOREA_WASTE_RULES_20260913_v1', r['local_path']),r['source_image_id'],'new',False) for r in rows if r['status']=='downloaded']
with ThreadPoolExecutor(max_workers=4) as ex:new=list(ex.map(sig,jobs))
matches=[]
for tid,_,b,path in new:
 for oid,split,c,op in old:
  distance=(b^c).bit_count()
  if distance<=4:matches.append({'id':f'taco_{tid:06d}','legacy_id':f's3_taco_{oid:06d}','legacy_bbox_split':split,'dhash_hamming':distance,'status':'similarity_candidate_not_confirmed_duplicate'})
(workspace_path('KOREA_WASTE_RULES_20260913_v1', 'reports/taco_similarity_candidates.json')).write_text(json.dumps({'scope':'825 known legacy TACO images vs downloaded new TACO only; not all legacy sources','threshold':4,'matches':matches},indent=2),encoding='utf-8')
print('Near-duplicate candidates:',len(matches))
