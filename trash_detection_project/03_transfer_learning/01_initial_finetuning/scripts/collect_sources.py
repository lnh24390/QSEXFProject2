"""출처를 밝힌 공개 이미지를 모은다. 학습이나 외부 업로드는 하지 않는다."""
# 현재 작업 디렉토리와 상관없이 공용 작업공간 경로를 찾는다.
from pathlib import Path as _WorkspacePath
import sys as _workspace_sys
_workspace_root = next(p for p in _WorkspacePath(__file__).resolve().parents if (p / 'workspace_paths.py').is_file())
_workspace_sys.path.insert(0, str(_workspace_root))
from workspace_paths import workspace_path

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import argparse, hashlib, json, time, re, io, gzip
import requests
from urllib.parse import urlparse
from PIL import Image

ROOT=workspace_path('KOREA_WASTE_RULES_20260913_v1')
PROJECT=workspace_path('')
UA='KoreanWasteDatasetResearch/1.0 (public dataset annotation; local research)'
BLOCKED_HOSTS=set()
def savejson(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def get(url,**kwargs):
    host=urlparse(url).netloc
    if host in BLOCKED_HOSTS: raise RuntimeError('rate_limited_host_stopped_for_this_run')
    r=requests.get(url,headers={'User-Agent':UA},timeout=(15,60),**kwargs)
    if r.status_code==429: BLOCKED_HOSTS.add(host)
    r.raise_for_status(); return r
def fetch_image(row):
    p=workspace_path('KOREA_WASTE_RULES_20260913_v1', row['local_path']); p.parent.mkdir(parents=True,exist_ok=True)
    try:
        if p.exists(): content=p.read_bytes()
        else:
            r=get(row['download_url']); content=r.content
            if len(content)>25_000_000: raise ValueError('image_over_25MB')
        with Image.open(io.BytesIO(content)) as im:
            im.load(); width,height=im.size; fmt=im.format
        if min(width,height)<160: raise ValueError('too_small')
        if not p.exists(): p.write_bytes(content)
        row.update(status='downloaded',width=width,height=height,format=fmt,sha256=hashlib.sha256(content).hexdigest(),bytes=len(content))
    except Exception as e: row.update(status='failed',error=f'{type(e).__name__}: {e}')
    return row
def taco():
    ann=workspace_path('KOREA_WASTE_RULES_20260913_v1', 'sources/taco/annotations.json')
    if not ann.exists(): savejson(ann,get('https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json').json())
    d=json.loads(ann.read_text(encoding='utf-8'))
    savejson(workspace_path('KOREA_WASTE_RULES_20260913_v1', 'sources/taco/zenodo_record.json'),get('https://zenodo.org/api/records/3587843').json())
    existing=set()
    for split in ('train','val','test'):
        existing.update(p.stem for p in (workspace_path('', 'YOLO_7CLASS_10000_PER_CLASS_20260909/images', split)).glob('s3_taco_*.jpg'))
    rows=[]
    for im in d['images']:
        legacy=f"s3_taco_{im['id']:06d}"
        rows.append({'id':f"taco_{im['id']:06d}",'source':'TACO_official','source_image_id':im['id'],
            'download_url':im['flickr_url'],'source_page':'https://doi.org/10.5281/zenodo.3587843',
            'license':'CC-BY-4.0','license_url':'https://creativecommons.org/licenses/by/4.0/',
            'attribution':'Pedro F. Proenca and Pedro Simoes, TACO: Trash Annotations in Context (2020); original photo URL retained',
            'local_path':f"raw/taco/taco_{im['id']:06d}"+('.png' if im['flickr_url'].split('?')[0].endswith('.png') else '.jpg'),
            'legacy_stem':legacy if legacy in existing else None,'country':'unknown','korea_specific':False,
            'semantic_review':'source_annotation_not_local_visual_review'})
    todo=[r for r in rows if not r['legacy_stem']]
    completed=[]
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures=[ex.submit(fetch_image,r) for r in todo]
        for i,f in enumerate(as_completed(futures),1):
            completed.append(f.result())
            if i%25==0:
                savejson(workspace_path('KOREA_WASTE_RULES_20260913_v1', 'sources/taco/download_manifest.json'),sorted(completed,key=lambda x:x['id'])+[r for r in rows if r['legacy_stem']])
                print('TACO',i,'/',len(todo),dict(Counter(r['status'] for r in completed)),flush=True)
    completed += [dict(r,status='existing_legacy_reused') for r in rows if r['legacy_stem']]
    savejson(workspace_path('KOREA_WASTE_RULES_20260913_v1', 'sources/taco/download_manifest.json'),sorted(completed,key=lambda x:x['id']))
    print('TACO complete',dict(Counter(r['status'] for r in completed)),flush=True)

QUERIES=[
 ('glass','soju bottle'),('glass','Korean beer bottle'),('glass','broken glass bottle'),('glass','glass bottle green brown clear'),
 ('can','crushed beverage can'),('can','Korean beverage can'),('can','dented tin can'),
 ('plastic','crushed plastic bottle'),('plastic','Korea plastic bottle'),('plastic','Yakult bottle'),
 ('plastic','plastic takeout container'),('plastic','shampoo bottle'),
 ('paper','Korea cardboard waste'),('paper','crumpled paper'),('paper','milk carton Korea'),
 ('vinyl','ramyeon packaging'),('vinyl','plastic bag Korea'),('vinyl','snack packaging Korea'),
 ('battery','used AA batteries'),('battery','button cell battery'),('battery','damaged battery'),
 ('general','broken ceramic plate'),('general','used tissue paper'),('general','dirty pizza box'),
 ('general','toothbrush discarded'),('general','cigarette butts Korea'),
]
def commons(limit):
    recs={}; queries=[]
    for family,q in QUERIES:
        try:
            r=get('https://commons.wikimedia.org/w/api.php',params={'action':'query','format':'json','generator':'search',
                'gsrsearch':q+' filetype:bitmap','gsrnamespace':6,'gsrlimit':limit,'prop':'imageinfo',
                'iiprop':'url|extmetadata|size','iiurlwidth':1600}).json()
            pages=list(r.get('query',{}).get('pages',{}).values())
            queries.append({'query':q,'family':family,'n':len(pages)})
            for page in pages:
                if 'imageinfo' not in page:continue
                ii=page['imageinfo'][0]; meta=ii.get('extmetadata',{})
                mv=lambda k:meta.get(k,{}).get('value','')
                lic=mv('LicenseShortName'); url=ii.get('thumburl',ii['url'])
                if not any(k in lic.lower() for k in ('cc by','cc0','public domain','cc-zero')): continue
                ident=f"commons_{page['pageid']}"
                if ident in recs:
                    recs[ident]['search_queries'].append(q); continue
                recs[ident]={'id':ident,'source':'Wikimedia_Commons','title':page['title'],
                    'download_url':url,'original_url':ii['url'],'source_page':ii['descriptionurl'],
                    'license':lic,'license_url':mv('LicenseUrl'),'attribution':mv('Artist'),'credit':mv('Credit'),
                    'description':mv('ImageDescription'),'categories':mv('Categories'),
                    'search_queries':[q],'search_family_hint':family,
                    'local_path':f'raw/commons/{ident}.jpg','semantic_review':'unreviewed_search_candidate',
                    'country':'unverified','korea_specific':None}
            print('Commons query',q,len(pages),'unique',len(recs),flush=True)
        except Exception as e: queries.append({'query':q,'error':str(e)}); print('Commons query error',q,str(e),flush=True)
        time.sleep(.25)
    savejson(workspace_path('KOREA_WASTE_RULES_20260913_v1', 'sources/commons/queries.json'),queries)
    done=[]
    with ThreadPoolExecutor(max_workers=3) as ex:
        for i,r in enumerate(ex.map(fetch_image,recs.values()),1):
            done.append(r)
            if i%20==0: print('Commons downloaded',i,'/',len(recs),flush=True)
    savejson(workspace_path('KOREA_WASTE_RULES_20260913_v1', 'sources/commons/download_manifest.json'),done)
    print('Commons complete',dict(Counter(r['status'] for r in done)),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('source',choices=['taco','commons']);ap.add_argument('--limit',type=int,default=12)
    a=ap.parse_args(); taco() if a.source=='taco' else commons(a.limit)
