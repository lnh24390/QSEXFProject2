# 현재 작업 디렉토리와 상관없이 공용 작업공간 경로를 찾는다.
from pathlib import Path as _WorkspacePath
import sys as _workspace_sys
_workspace_root = next(p for p in _WorkspacePath(__file__).resolve().parents if (p / 'workspace_paths.py').is_file())
_workspace_sys.path.insert(0, str(_workspace_root))
from workspace_paths import workspace_path

from pathlib import Path
import zipfile,json,hashlib,io,collections
from PIL import Image,ImageDraw
R=workspace_path('KOREA_WASTE_RULES_20260913_v1').resolve();p=workspace_path('KOREA_WASTE_RULES_20260913_v1', 'sources/taco/download_manifest.json');rows=json.loads(p.read_text('utf-8'));ann=json.loads((workspace_path('KOREA_WASTE_RULES_20260913_v1', 'sources/taco/annotations.json')).read_text('utf-8'));ims={i['id']:i for i in ann['images']}
with zipfile.ZipFile(workspace_path('KOREA_WASTE_RULES_20260913_v1', 'sources/taco_archive/TACO.zip')) as z:
 names=z.namelist();count=0
 for row in rows:
  if row['status']!='failed':continue
  suffix='/'+ims[row['source_image_id']]['file_name'];matches=[n for n in names if n.endswith(suffix) and not n.startswith('__MACOSX')]
  if len(matches)!=1:print('missing',suffix,len(matches));continue
  data=z.read(matches[0]);im=Image.open(io.BytesIO(data));im.verify();im=Image.open(io.BytesIO(data));dest=workspace_path('KOREA_WASTE_RULES_20260913_v1', row['local_path']);dest.write_bytes(data)
  row.update(status='downloaded',recovered_from='https://doi.org/10.5281/zenodo.3587843',archive_member=matches[0],width=im.width,height=im.height,sha256=hashlib.sha256(data).hexdigest());row.pop('error',None);count+=1
p.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8');print('TACO recovered',count,collections.Counter(r['status'] for r in rows))
coins=[r for r in json.loads((workspace_path('KOREA_WASTE_RULES_20260913_v1', 'sources/coin_cells/download_manifest.json')).read_text('utf-8')) if r['status'].startswith('downloaded')]
s=Image.new('RGB',(1400,900),'white');d=ImageDraw.Draw(s)
for i,r in enumerate(coins):
 im=Image.open(workspace_path('KOREA_WASTE_RULES_20260913_v1', r['local_path'])).convert('RGB');im.thumbnail((190,250));x=i%7*200;y=i//7*225;im.thumbnail((190,190));s.paste(im,(x,y));d.text((x,y+195),str(i),fill='black')
s.save(workspace_path('KOREA_WASTE_RULES_20260913_v1', 'reports/coin_review.jpg'));(workspace_path('KOREA_WASTE_RULES_20260913_v1', 'reports/coin_review_index.json')).write_text(json.dumps(coins,indent=2),encoding='utf-8')
