"""배급처의 bbox 라벨을 가져오고, 원래 분할을 유지하며, 같은 잘라낸 이미지를 중복해서 세지 않는다."""
# 현재 작업 디렉토리와 상관없이 공용 작업공간 경로를 찾는다.
from pathlib import Path as _WorkspacePath
import sys as _workspace_sys
_workspace_root = next(p for p in _WorkspacePath(__file__).resolve().parents if (p / 'workspace_paths.py').is_file())
_workspace_sys.path.insert(0, str(_workspace_root))
from workspace_paths import workspace_path

from pathlib import Path
import json,zipfile,hashlib,os,math
from collections import Counter,defaultdict
from PIL import Image
ROOT=workspace_path('KOREA_WASTE_RULES_20260913_v1')
archive=workspace_path('KOREA_WASTE_RULES_20260913_v1', 'sources/warp/WaRP.zip');target=workspace_path('KOREA_WASTE_RULES_20260913_v1', 'raw/warp');target.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(archive) as z:
 for n in z.namelist():
  if not n.startswith(('Warp-D/','Warp-S/')) or n.endswith('/') or Path(n).suffix.lower() not in ('.jpg','.jpeg','.png','.txt'):continue
  dest=(workspace_path('KOREA_WASTE_RULES_20260913_v1/raw/warp', n)).resolve()
  if target.resolve() not in dest.parents:raise ValueError('Unsafe archive path')
  dest.parent.mkdir(parents=True,exist_ok=True)
  if not dest.exists():dest.write_bytes(z.read(n))
names=(workspace_path('KOREA_WASTE_RULES_20260913_v1/raw/warp', 'Warp-D/classes.txt')).read_text().splitlines()
mapping={i:'plastic' if n.startswith(('bottle-','detergent-')) or n=='canister' else 'glass' if n.startswith('glass-') else 'can' if n=='cans' else 'paper' if n.endswith('cardboard') else None for i,n in enumerate(names)}
classes=['paper','plastic','can','glass','battery','vinyl','general'];out=workspace_path('KOREA_WASTE_RULES_20260913_v1', 'supplement_warp_bbox');rows=[];hashes=defaultdict(set);stats=Counter();errors=[]
for split in ('train','test'):
 for ip in sorted((workspace_path('KOREA_WASTE_RULES_20260913_v1/raw/warp', 'Warp-D', split, 'images')).glob('*')):
  if ip.suffix.lower() not in ('.jpg','.jpeg','.png'):continue
  lp=workspace_path('KOREA_WASTE_RULES_20260913_v1/raw/warp', 'Warp-D', split, 'labels', ip.stem+'.txt');lines=[];instances=[];issues=[]
  with Image.open(ip) as im:im.verify()
  with Image.open(ip) as im:W,H=im.size
  for lineno,line in enumerate(lp.read_text().splitlines(),1):
   if not line.strip():continue
   v=list(map(float,line.split()));ci=int(v[0]);c=v[1:];name=mapping.get(ci)
   if not name or len(v)!=5 or v[0]!=ci or not all(math.isfinite(x) and 0<=x<=1 for x in c) or c[2]<=0 or c[3]<=0:issues.append(f'invalid_label_{lineno}');continue
   x,y,w,h=c
   if min(x-w/2,y-h/2)<-0.00001 or max(x+w/2,y+h/2)>1.00001:issues.append(f'bbox_outside_image_{lineno}')
   lines.append(str(classes.index(name))+' '+' '.join(f'{x:.8f}' for x in c));instances.append({'source_class':names[ci],'class_name':name,'bbox_xywh_normalized':c})
  sha=hashlib.file_digest(ip.open('rb'),'sha256').hexdigest();hashes[sha].add(split)
  ident='warp_'+hashlib.sha256(ip.stem.encode()).hexdigest()[:20]
  row={'id':ident,'source':'WaRP-D','source_image':str(ip.relative_to(ROOT)),'source_label':str(lp.relative_to(ROOT)),'source_split':split,'split':split,'width':W,'height':H,'sha256':sha,'source_page':'https://github.com/AIRI-Institute/WaRP','license':'CC0-1.0 per publisher Kaggle metadata','license_metadata':'sources/warp/metadata.json','country':'not_Korea_specific','label_method':'publisher_bbox_material_mapping','visual_review':'pending','instances':instances,'issues':issues}
  if not issues:
   for base in (out,workspace_path('KOREA_WASTE_RULES_20260913_v1', 'staged_bbox')):
    # 새 출처의 평가용 자료는 물려받은 모델의 기존 val/test 와 분리한다.
    if base.name=='staged_bbox' and split!='train':continue
    dest=base/'images'/split/(ident+ip.suffix);lab=base/'labels'/split/(ident+'.txt');dest.parent.mkdir(parents=True,exist_ok=True);lab.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():os.link(ip,dest)
    lab.write_text('\n'.join(lines)+'\n',encoding='utf-8')
   row['status']='source_labels_validated_visual_review_pending';stats[split+'_images']+=1;stats['instances']+=len(lines)
  else:row['status']='held_for_review';errors.append({'id':ident,'issues':issues})
  rows.append(row)
# 배급처 분할에 완전히 같은 파일이 있다. 겹치는 해시는 모두 평가용으로 뺀다.
shared={h for h,s in hashes.items() if len(s)>1};excluded=0
for row in rows:
 if row['source_split']=='train' and row['sha256'] in shared and not row['issues']:
  row['status']='excluded_train_duplicate_of_source_holdout';excluded+=1;stats['train_images']-=1;stats['instances']-=len(row['instances'])
  for base in (out,workspace_path('KOREA_WASTE_RULES_20260913_v1', 'staged_bbox')):
   for kind,ext in [('images',Path(row['source_image']).suffix),('labels','.txt')]:
    p=(base/kind/'train'/(row['id']+ext)).resolve()
    assert ROOT.resolve() in p.parents
    if p.exists():p.unlink()
(workspace_path('KOREA_WASTE_RULES_20260913_v1', 'annotations/warp_instances.json')).write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
(workspace_path('KOREA_WASTE_RULES_20260913_v1', 'sources/warp/class_mapping.json')).write_text(json.dumps([{'source_id':i,'source_class':n,'class_name':mapping[i]} for i,n in enumerate(names)],indent=2),encoding='utf-8')
report={'counts':dict(stats),'held_images':len(errors),'issues':errors,'original_exact_hash_cross_source_split':len(shared),'excluded_train_duplicates':excluded,'exported_exact_hash_cross_split':0,'unique_image_hashes':len(hashes),'new_independent_training_images_not_claimed_for_crop_exports':True,'segmentation_note':'Warp-C crops excluded from added image count. Warp-S retained raw only; no bbox-to-mask fabrication.'}
(workspace_path('KOREA_WASTE_RULES_20260913_v1', 'reports/warp_import.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
yaml='\n'.join(f'  {i}: {n}' for i,n in enumerate(classes))
(workspace_path('KOREA_WASTE_RULES_20260913_v1/supplement_warp_bbox', 'SOURCE_LABELS_REVIEW_PENDING.yaml')).write_text(f'path: {out.as_posix()}\ntrain: images/train\nval: images/test\nnames:\n{yaml}\n',encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
