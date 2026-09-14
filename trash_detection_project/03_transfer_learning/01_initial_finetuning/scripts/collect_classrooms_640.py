# 현재 작업 디렉토리와 상관없이 공용 작업공간 경로를 찾는다.
from pathlib import Path as _WorkspacePath
import sys as _workspace_sys
_workspace_root = next(p for p in _WorkspacePath(__file__).resolve().parents if (p / 'workspace_paths.py').is_file())
_workspace_sys.path.insert(0, str(_workspace_root))
from workspace_paths import workspace_path

from pathlib import Path
import requests,json,io,time,hashlib
from PIL import Image
R=workspace_path('KOREA_WASTE_RULES_20260913_v1');out=workspace_path('KOREA_WASTE_RULES_20260913_v1', 'review_pending_640');out.mkdir(exist_ok=True)
titles=['File:Computer classroom.JPG','File:Classroom 2 TNMOC-CP.jpg','File:CComputer lab.jpg','File:Chihlee computer classroom 20091120.jpg','File:Acer desktop computers in computer classroom of Baozhong Junior High School 20121009.jpg']
headers={'User-Agent':'KoreanWasteDatasetResearch/1.0 (public dataset annotation; local research)'}
r=requests.get('https://commons.wikimedia.org/w/api.php',params={'action':'query','format':'json','titles':'|'.join(titles),'prop':'imageinfo','iiprop':'url|extmetadata|size','iiurlwidth':960},headers=headers,timeout=30);r.raise_for_status();rows=[]
for page in r.json()['query']['pages'].values():
 if 'imageinfo' not in page:continue
 ii=page['imageinfo'][0];meta=ii.get('extmetadata',{});get=lambda k:meta.get(k,{}).get('value','');row={'id':'room_commons_'+str(page['pageid']),'title':page['title'],'source_page':ii['descriptionurl'],'license':get('LicenseShortName'),'license_url':get('LicenseUrl'),'attribution':get('Artist'),'download_url':ii.get('thumburl',ii['url']),'original_url':ii['url']}
 res=requests.get(row['download_url'],headers=headers,timeout=30);row['http_status']=res.status_code
 if res.status_code in [403,429]:row['status']='access_or_rate_limit';rows.append(row);break
 res.raise_for_status();im=Image.open(io.BytesIO(res.content)).convert('RGB');w,h=im.size;rr=min(640/w,640/h);nw,nh=round(w*rr),round(h*rr);canvas=Image.new('RGB',(640,640),(114,114,114));canvas.paste(im.resize((nw,nh)),((640-nw)//2,(640-nh)//2));dest=workspace_path('KOREA_WASTE_RULES_20260913_v1/review_pending_640', row['id']+'.jpg');canvas.save(dest,quality=85,optimize=True);row.update(status='640_review_pending',candidate_path=str(dest.resolve()),sha256=hashlib.file_digest(dest.open('rb'),'sha256').hexdigest());rows.append(row);print(row['id'],row['title'],flush=True);time.sleep(1)
(workspace_path('KOREA_WASTE_RULES_20260913_v1', 'reports/classroom_commons_candidates.json')).write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
