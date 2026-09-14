# 현재 작업 디렉토리와 상관없이 공용 작업공간 경로를 찾는다.
from pathlib import Path as _WorkspacePath
import sys as _workspace_sys
_workspace_root = next(p for p in _WorkspacePath(__file__).resolve().parents if (p / 'workspace_paths.py').is_file())
_workspace_sys.path.insert(0, str(_workspace_root))
from workspace_paths import workspace_path

from pathlib import Path
import zipfile,json,hashlib,io,os,collections
from PIL import Image,ImageOps,ImageDraw
R=workspace_path('KOREA_WASTE_RULES_20260913_v1').resolve();rows=[]
with zipfile.ZipFile(workspace_path('KOREA_WASTE_RULES_20260913_v1', 'sources/household_batteries/household_batteries.zip')) as z:
 parts=collections.defaultdict(set)
 for n in z.namelist():
  if n.startswith('dataset/dataset/') and '/images/' in n:parts[hashlib.sha256(z.read(n)).hexdigest()].add(n.split('/')[2])
 seen=set()
 for n in z.namelist():
  if '/annotations/' not in n:continue
  ann=[x.split() for x in z.read(n).decode().splitlines() if x.strip()]
  if not any(a[0]=='battery' for a in ann):continue
  member=n.replace('/annotations/','/images/').replace('.txt','.jpg');data=z.read(member);sha=hashlib.sha256(data).hexdigest();splits=parts[sha]
  split='test' if 'test' in splits else 'val' if 'val' in splits else 'train'
  name='hb_'+sha[:20];raw=workspace_path('KOREA_WASTE_RULES_20260913_v1', 'raw/household_batteries', f'{name}.jpg');raw.parent.mkdir(exist_ok=True);raw.write_bytes(data)
  im=Image.open(io.BytesIO(data));w,h=im.size;boxes=[];why=[]
  if {a[0] for a in ann}!={'battery'}:why.append('other_source_classes_need_review')
  if sha in seen:why.append('duplicate_content')
  seen.add(sha)
  for a in ann:
   x1,y1,x2,y2=map(float,a[4:8])
   if not(0<=x1<x2<=w and 0<=y1<y2<=h):why.append('invalid_bbox')
   if a[0]=='battery':boxes.append([x1/w,y1/h,x2/w,y2/h])
  row={'id':name,'source':'Kaggle_markcsizmadia_batteries_dices_toy_cars','archive_member':member,'annotation_member':n,'source_page':'https://www.kaggle.com/datasets/markcsizmadia/object-detection-batteries-dices-and-toy-cars','license':'CC0 as declared by publisher','local_path':str(raw.relative_to(R)),'sha256':sha,'source_splits':sorted(splits),'split':split,'boxes_xyxy_normalized':boxes,'visual_review':'pending','status':'held' if why else 'source_bbox_validated','hold_reasons':why}
  rows.append(row)
  if why:continue
  for folder in ['supplement_household_bbox']+(['staged_bbox'] if split=='train' else []):
   out=workspace_path('KOREA_WASTE_RULES_20260913_v1', folder);ip=out/'images'/split/raw.name;lp=out/'labels'/split/(name+'.txt');ip.parent.mkdir(parents=True,exist_ok=True);lp.parent.mkdir(parents=True,exist_ok=True)
   if not ip.exists():os.link(raw,ip)
   lp.write_text(''.join(f'4 {(a+c)/2:.8f} {(b+d)/2:.8f} {c-a:.8f} {d-b:.8f}\n' for a,b,c,d in boxes))
(workspace_path('KOREA_WASTE_RULES_20260913_v1', 'annotations/household_battery_instances.json')).write_text(json.dumps(rows,indent=2),encoding='utf-8')
print(collections.Counter((r['split'],r['status']) for r in rows))
# 출처 검토용 모아보기 시트다. 승인 근거가 아니다.
for key,rr in [('household',rows[:40])]:
 sheet=Image.new('RGB',(1200,1000),'white');draw=ImageDraw.Draw(sheet)
 for i,r in enumerate(rr):
  im=Image.open(workspace_path('KOREA_WASTE_RULES_20260913_v1', r['local_path'])).convert('RGB');im.thumbnail((145,170));x=(i%8)*150;y=(i//8)*200;sheet.paste(im,(x,y));draw.text((x,y+175),str(i),fill='black')
 sheet.save(workspace_path('KOREA_WASTE_RULES_20260913_v1', f'reports/{key}_review.jpg'));(workspace_path('KOREA_WASTE_RULES_20260913_v1', f'reports/{key}_review_index.json')).write_text(json.dumps(rr,indent=2))
# 목적을 정해 고른 Open Images 미리보기.
oi=json.loads((workspace_path('KOREA_WASTE_RULES_20260913_v1', 'sources/openimages/train_download_manifest.json')).read_text('utf-8'))
for key,cls in [('towel','/m/0162_1'),('monitor','/m/02522')]:
 rr=[r for r in oi if any(a['LabelName']==cls for a in r['annotations']) and r['status'].startswith('downloaded')][:40]
 sheet=Image.new('RGB',(1500,1100),'white');draw=ImageDraw.Draw(sheet)
 for i,r in enumerate(rr):
  im=Image.open(workspace_path('KOREA_WASTE_RULES_20260913_v1', r['local_path'])).convert('RGB');im.thumbnail((180,180));x=(i%8)*187;y=(i//8)*220;sheet.paste(im,(x,y));draw.text((x,y+185),str(i),fill='black')
 sheet.save(workspace_path('KOREA_WASTE_RULES_20260913_v1', f'reports/{key}_review.jpg'));(workspace_path('KOREA_WASTE_RULES_20260913_v1', f'reports/{key}_review_index.json')).write_text(json.dumps(rr,indent=2),encoding='utf-8')
