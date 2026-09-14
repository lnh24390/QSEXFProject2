"""배급처가 공개한 압축 자료를 무결성 검사와 진행 기록을 남기며 내려받는다."""
# 현재 작업 디렉토리와 상관없이 공용 작업공간 경로를 찾는다.
from pathlib import Path as _WorkspacePath
import sys as _workspace_sys
_workspace_root = next(p for p in _WorkspacePath(__file__).resolve().parents if (p / 'workspace_paths.py').is_file())
_workspace_sys.path.insert(0, str(_workspace_root))
from workspace_paths import workspace_path

from pathlib import Path
import requests,hashlib,time,json,zipfile,sys
ROOT=workspace_path('KOREA_WASTE_RULES_20260913_v1')
SOURCES={
 'household_batteries':('https://www.kaggle.com/api/v1/datasets/download/markcsizmadia/object-detection-batteries-dices-and-toy-cars','household_batteries.zip'),
 'garbage_v2':('https://www.kaggle.com/api/v1/datasets/download/sumn2u/garbage-classification-v2','garbage_v2.zip'),
 'warp':('https://www.kaggle.com/api/v1/datasets/download/parohod/warp-waste-recycling-plant-dataset','WaRP.zip'),
 'taco_archive':('https://zenodo.org/api/records/3587843/files/TACO.zip/content','TACO.zip')}
name=sys.argv[1];url,filename=SOURCES[name];out=workspace_path('KOREA_WASTE_RULES_20260913_v1', 'sources', name);out.mkdir(parents=True,exist_ok=True);p=out/filename;part=p.with_suffix('.part')
if not p.exists():
 with requests.get(url,stream=True,timeout=(20,90)) as res:
  res.raise_for_status();total=int(res.headers.get('Content-Length',0));received=0;last=time.monotonic()
  with part.open('wb') as f:
   for chunk in res.iter_content(4*1024*1024):
    f.write(chunk);received+=len(chunk)
    if received>8_000_000_000:raise RuntimeError('Unexpected archive exceeds 8GB')
    if time.monotonic()-last>20:print(name,round(received/1e6),'MB of',round(total/1e6),flush=True);last=time.monotonic()
  if total and total!=received:raise RuntimeError('Incomplete HTTP body')
 if not zipfile.is_zipfile(part):raise RuntimeError('Response was not a ZIP archive')
 part.replace(p)
with zipfile.ZipFile(p) as z:
 infos=[{'path':i.filename,'bytes':i.file_size} for i in z.infolist()]
 (out/'archive_inventory.json').write_text(json.dumps(infos,indent=2),encoding='utf-8')
digest=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
(out/'download_receipt.json').write_text(json.dumps({'url':url,'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':digest,'members':len(infos)},indent=2),encoding='utf-8')
print(name,'complete',p.stat().st_size,'bytes;',len(infos),'members',flush=True)
