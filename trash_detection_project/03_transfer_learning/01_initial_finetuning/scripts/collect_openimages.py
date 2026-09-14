"""수건과 봉투 구분, 모니터 오탐용 음성 자료를 배급처 주석에서 골라 모은다."""
# 현재 작업 디렉토리와 상관없이 공용 작업공간 경로를 찾는다.
from pathlib import Path as _WorkspacePath
import sys as _workspace_sys
_workspace_root = next(p for p in _WorkspacePath(__file__).resolve().parents if (p / 'workspace_paths.py').is_file())
_workspace_sys.path.insert(0, str(_workspace_root))
from workspace_paths import workspace_path

from pathlib import Path
import requests,csv,json,time,hashlib,sys,io
from collections import defaultdict,Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from PIL import Image
ROOT=workspace_path('KOREA_WASTE_RULES_20260913_v1');OUT=workspace_path('KOREA_WASTE_RULES_20260913_v1', 'sources/openimages');OUT.mkdir(exist_ok=True)
TARGETS={'/m/0162_1':('Towel','general',500),'/m/05gqfk':('Plastic bag','vinyl',500),'/m/02w3r3':('Paper towel','paper',180),'/m/09gtd':('Toilet paper','paper',100),'/m/02522':('Computer monitor',None,250),'/m/01c648':('Laptop',None,80)}
splits=sys.argv[1:] or ['validation','test','train']
def fetch(row):
 p=workspace_path('KOREA_WASTE_RULES_20260913_v1', row['local_path']);p.parent.mkdir(parents=True,exist_ok=True)
 try:
  if not p.exists():
   r=requests.get(row['download_url'],timeout=(15,45));r.raise_for_status();p.write_bytes(r.content)
  with Image.open(p) as im:im.verify()
  with Image.open(p) as im:row.update(width=im.width,height=im.height)
  row.update(status='downloaded_annotation_review_pending',sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest())
 except Exception as e:row.update(status='download_failed',error=str(e))
 return row
for split in splits:
 filtered=workspace_path('KOREA_WASTE_RULES_20260913_v1/sources/openimages', f'{split}_target_boxes.json')
 if not filtered.exists():
  u='https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv' if split=='train' else f'https://storage.googleapis.com/openimages/v5/{split}-annotations-bbox.csv'
  keep=defaultdict(list);last=time.monotonic();n=0
  with requests.get(u,stream=True,timeout=(20,90)) as res:
   res.raise_for_status();res.raw.decode_content=True
   reader=csv.DictReader(line.decode('utf-8') for line in res.iter_lines(chunk_size=1024*1024))
   for row in reader:
    n+=1
    if row['LabelName'] in TARGETS:keep[row['ImageID']].append(row)
    if time.monotonic()-last>25:print(split,'annotation rows scanned',n,'target images',len(keep),flush=True);last=time.monotonic()
  filtered.write_text(json.dumps({'url':u,'scanned_rows':n,'images':keep},indent=2),encoding='utf-8')
 else:keep=json.loads(filtered.read_text('utf-8'))['images']
 counts=Counter();selected={}
 # 수집 순서가 아니라 이미지 해시로 섞어 매번 같은 결과가 나오게 한다.
 for iid,aa in sorted(keep.items(),key=lambda kv:hashlib.sha256(kv[0].encode()).hexdigest()):
  valid=[a for a in aa if a['IsGroupOf']=='0' and a['IsDepiction']=='0' and float(a['Confidence'])==1]
  if not valid:continue
  desired=[a['LabelName'] for a in valid if counts[a['LabelName']]<(TARGETS[a['LabelName']][2] if split=='train' else min(100,TARGETS[a['LabelName']][2]))]
  if not desired:continue
  selected[iid]=dict(id=f'oi_{iid}',source='Open_Images',source_split=split,source_image_id=iid,local_path=f'raw/openimages/{split}/{iid}.jpg',download_url=f'https://open-images-dataset.s3.amazonaws.com/{split}/{iid}.jpg',source_page='https://storage.googleapis.com/openimages/web/download_v7.html',image_license='CC-BY-2.0 as stated by Open Images; individual metadata retained',annotation_license='CC-BY-4.0',annotations=aa,proposed_classes=sorted({TARGETS[a['LabelName']][1] for a in valid if TARGETS[a['LabelName']][1]}),monitor_negative_candidate=any(a['LabelName'] in ('/m/02522','/m/01c648') for a in valid),visual_review='pending_all_relevant_objects_and_physical_vs_displayed_content',all_7class_objects_annotated=False)
  counts.update(set(desired))
 print(split,'selected',len(selected),dict(counts),flush=True)
 meta_path=workspace_path('KOREA_WASTE_RULES_20260913_v1/sources/openimages', f'{split}_selected_metadata.json')
 if not meta_path.exists():
  filename='train-images-boxable-with-rotation.csv' if split=='train' else f'{split}-images-with-rotation.csv'
  u=f'https://storage.googleapis.com/openimages/2018_04/{split}/{filename}'
  metadata={}
  with requests.get(u,stream=True,timeout=(20,90)) as res:
   res.raise_for_status();res.raw.decode_content=True
   for m in csv.DictReader(line.decode('utf-8') for line in res.iter_lines(chunk_size=1024*1024)):
    if m['ImageID'] in selected:metadata[m['ImageID']]=m
  meta_path.write_text(json.dumps({'url':u,'images':metadata},ensure_ascii=False,indent=2),encoding='utf-8')
 else:metadata=json.loads(meta_path.read_text('utf-8'))['images']
 for iid,row in selected.items():row['image_metadata']=metadata.get(iid,{});row['attribution_status']='source_metadata_present' if iid in metadata else 'missing_hold'
 done=[]
 with ThreadPoolExecutor(max_workers=4) as ex:
  for i,f in enumerate(as_completed([ex.submit(fetch,row) for row in selected.values()]),1):
   done.append(f.result())
   if i%50==0:print(split,'downloaded attempts',i,'/',len(selected),flush=True)
   if i%100==0:(workspace_path('KOREA_WASTE_RULES_20260913_v1/sources/openimages', f'{split}_download_manifest.json')).write_text(json.dumps(done,ensure_ascii=False,indent=2),encoding='utf-8')
 (workspace_path('KOREA_WASTE_RULES_20260913_v1/sources/openimages', f'{split}_download_manifest.json')).write_text(json.dumps(sorted(done,key=lambda r:r['id']),ensure_ascii=False,indent=2),encoding='utf-8')
 print(split,'complete',dict(Counter(r['status'] for r in done)),flush=True)
