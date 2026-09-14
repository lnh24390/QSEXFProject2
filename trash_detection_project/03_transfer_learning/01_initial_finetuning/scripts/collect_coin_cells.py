"""출처를 밝힌 코인·버튼 전지 사진을 모은다. Wikimedia 에 설명이 있는 user agent 로 요청한다."""
# 현재 작업 디렉토리와 상관없이 공용 작업공간 경로를 찾는다.
from pathlib import Path as _WorkspacePath
import sys as _workspace_sys
_workspace_root = next(p for p in _WorkspacePath(__file__).resolve().parents if (p / 'workspace_paths.py').is_file())
_workspace_sys.path.insert(0, str(_workspace_root))
from workspace_paths import workspace_path

from pathlib import Path
import requests,json,time,hashlib,io
from PIL import Image
from urllib.parse import urlsplit
ROOT=workspace_path('KOREA_WASTE_RULES_20260913_v1');OUT=workspace_path('KOREA_WASTE_RULES_20260913_v1', 'sources/coin_cells');OUT.mkdir(exist_ok=True)
headers={'User-Agent':'KoreanWasteDatasetResearch/1.0 (public dataset annotation; local research)'}
categories=['Button cells','LR44 batteries','LR41 batteries','LR54 batteries','CR2032 batteries']
rows={};queries=[]
for category in categories:
 res=requests.get('https://commons.wikimedia.org/w/api.php',params={'action':'query','format':'json','generator':'categorymembers','gcmtitle':'Category:'+category,'gcmtype':'file','gcmlimit':100,'prop':'imageinfo','iiprop':'url|extmetadata|size','iiurlwidth':1600},headers=headers,timeout=40)
 queries.append({'category':category,'http_status':res.status_code})
 if res.status_code in (403,429):break
 res.raise_for_status();pages=res.json().get('query',{}).get('pages',{})
 for k,page in pages.items():
  if 'imageinfo' not in page:continue
  ii=page['imageinfo'][0];meta=ii.get('extmetadata',{});get=lambda key:meta.get(key,{}).get('value','')
  license=get('LicenseShortName')
  if not any(s in license.lower() for s in ('cc by','cc0','public domain','cc-zero')):continue
  if not urlsplit(ii['url']).path.lower().endswith(('.jpg','.jpeg','.png')):continue
  rows[k]={'id':'coin_commons_'+k,'source':'Wikimedia_Commons','source_category':category,'title':page['title'],'source_page':ii['descriptionurl'],'download_url':ii.get('thumburl',ii['url']),'original_url':ii['url'],'license':license,'license_url':get('LicenseUrl'),'attribution':get('Artist'),'credit':get('Credit'),'description':get('ImageDescription'),'proposed_class':'battery','visual_review':'pending_actual_battery_extents','local_path':'raw/coin_cells/coin_commons_'+k+Path(urlsplit(ii['url']).path).suffix.lower()}
 print(category,len(pages),'unique',len(rows),flush=True);time.sleep(1)
(workspace_path('KOREA_WASTE_RULES_20260913_v1/sources/coin_cells', 'queries.json')).write_text(json.dumps(queries,indent=2),encoding='utf-8');done=[]
for i,row in enumerate(rows.values(),1):
 p=workspace_path('KOREA_WASTE_RULES_20260913_v1', row['local_path']);p.parent.mkdir(parents=True,exist_ok=True)
 try:
  if not p.exists():
   res=requests.get(row['download_url'],headers=headers,timeout=45)
   if res.status_code in (403,429):raise RuntimeError(f'rate_or_access_limit_{res.status_code}')
   res.raise_for_status();p.write_bytes(res.content)
  with Image.open(p) as im:im.verify()
  with Image.open(p) as im:row.update(width=im.width,height=im.height)
  row.update(status='downloaded_review_pending',sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest())
 except Exception as e:row.update(status='failed',error=str(e))
 done.append(row)
 if i%15==0:print('coin images',i,'/',len(rows),flush=True)
 (workspace_path('KOREA_WASTE_RULES_20260913_v1/sources/coin_cells', 'download_manifest.json')).write_text(json.dumps(done,ensure_ascii=False,indent=2),encoding='utf-8')
 if 'rate_or_access_limit' in row.get('error',''):break
 time.sleep(.4)
print('coin download completed',len(done),'attempts',flush=True)
