"""라벨을 독립 사본으로 만들고, 과제별 원래 클래스 ID·분할을 유지하며, 불확실한 항목을 기록한다."""
# 현재 작업 디렉토리와 상관없이 공용 작업공간 경로를 찾는다.
from pathlib import Path as _WorkspacePath
import sys as _workspace_sys
_workspace_root = next(p for p in _WorkspacePath(__file__).resolve().parents if (p / 'workspace_paths.py').is_file())
_workspace_sys.path.insert(0, str(_workspace_root))
from workspace_paths import workspace_path

from pathlib import Path
from collections import Counter, defaultdict
import json, re, os, hashlib, math, gzip, sys
from PIL import Image, ImageOps
ROOT=workspace_path('KOREA_WASTE_RULES_20260913_v1'); PROJECT=workspace_path('')
NAMES={'bbox':['paper','plastic','can','glass','battery','vinyl','general'], 'seg':['general','can','plastic','paper','glass','vinyl','battery']}
OLD={'bbox':workspace_path('', 'YOLO_7CLASS_10000_PER_CLASS_20260909'),'seg':workspace_path('', 'YOLO_SEG_7CLASS_10000_PER_CLASS_20260910_yolo_seg')}
def dump(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def key(p):return re.sub(r'^images__(train|val|test)__','',p.stem)
def link(src,dst):
 dst.parent.mkdir(parents=True,exist_ok=True)
 if not dst.exists():os.link(src,dst)
def write(p,s):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s,encoding='utf-8')
def bbox(coords,task):
 if task=='bbox':return coords
 xs=coords[::2];ys=coords[1::2];return [(min(xs)+max(xs))/2,(min(ys)+max(ys))/2,max(xs)-min(xs),max(ys)-min(ys)]
def source_box(ann,im,letterbox):
 x,y,w,h=ann['bbox'];W,H=im['width'],im['height']
 if letterbox:
  m=max(W,H);return [(x+w/2+(m-W)/2)/m,(y+h/2+(m-H)/2)/m,w/m,h/m]
 return [(x+w/2)/W,(y+h/2)/H,w/W,h/H]
# 사용자 확인 사항: 재질·품목을 인식하고, 지역별 배출 분류는 정하지 않는다.
# 모호한 복합재질, 음식물, 라벨 없는 쓰레기는 아직 정리하지 못했다.
MAP={1:'battery',4:'plastic',5:'plastic',6:'glass',7:'plastic',8:'can',9:'glass',10:'can',11:'can',12:'can',13:'paper',14:'paper',15:'paper',16:'paper',17:'paper',18:'paper',19:'paper',20:'paper',21:'plastic',22:'plastic',23:'glass',24:'plastic',26:'glass',27:'plastic',28:'can',29:'plastic',30:'paper',31:'paper',32:'paper',33:'paper',34:'paper',35:'paper',36:'vinyl',37:'plastic',38:'vinyl',39:'vinyl',40:'vinyl',41:'vinyl',42:'vinyl',43:'plastic',44:'plastic',45:'plastic',46:'plastic',47:'plastic',48:'vinyl',49:'plastic',50:'can',53:'general',55:'plastic',56:'paper',57:'plastic',59:'general'}
EXCEPTIONS=MAP
def attributes(cat):
 c=MAP.get(cat);material=('glass' if cat in (6,9,23,26) else 'paper' if cat==31 else c)
 return {'recognition_group':c,'material_hint':material,'damage':'broken' if cat==9 else 'unknown','contamination':'unknown','color':'unknown','reflection':'unknown','korea_specific':None,'visual_review':'pending'}
def main():
 d=json.loads((workspace_path('KOREA_WASTE_RULES_20260913_v1', 'sources/taco/annotations.json')).read_text('utf-8')); ims={i['id']:i for i in d['images']};anns=defaultdict(list)
 for a in d['annotations']:anns[a['image_id']].append(a)
 cats={c['id']:c['name'] for c in d['categories']}; changes=[]; stats={};source_counts=Counter(); errors=[]
 groupmeta={}
 with gzip.open(OLD['bbox']/'selection_manifest.jsonl.gz','rt',encoding='utf-8') as f:
  for line in f:
   r=json.loads(line);groupmeta[key(Path(r['image']))]=r
 if '--new-only' in sys.argv:
  stats=json.loads((workspace_path('KOREA_WASTE_RULES_20260913_v1', 'reports/build_summary.json')).read_text('utf-8'))['legacy']
  changes=json.loads((workspace_path('KOREA_WASTE_RULES_20260913_v1', 'annotations/legacy_rule_changes.json')).read_text('utf-8'))
 for task,old in ({} if '--new-only' in sys.argv else OLD).items():
  out=workspace_path('KOREA_WASTE_RULES_20260913_v1', 'staged_'+task);summary=Counter();groups=defaultdict(set)
  with (workspace_path('KOREA_WASTE_RULES_20260913_v1', f'legacy_{task}_manifest.jsonl')).open('w',encoding='utf-8') as manifest:
   for split in ('train','val','test'):
    for ip in sorted((old/'images'/split).iterdir()):
     if ip.suffix.lower() not in ('.jpg','.jpeg','.png'):continue
     st=key(ip);lp=old/'labels'/split/(ip.stem+'.txt');text=lp.read_text('utf-8');new=[];instances=[]
     tid=int(st.rsplit('_',1)[-1]) if st.startswith('s3_taco_') else None;used=set()
     for idx,line in enumerate(text.splitlines()):
      if not line.strip():continue
      v=list(map(float,line.split()));cid=int(v[0]);co=v[1:]
      valid=v[0]==cid and 0<=cid<7 and all(math.isfinite(x) and 0<=x<=1 for x in co) and (len(co)==4 if task=='bbox' else len(co)>=6 and len(co)%2==0)
      if not valid:errors.append({'task':task,'file':str(lp),'line':idx+1});continue
      oldname=NAMES[task][cid];matched=None;err=None
      if tid in ims:
       target=bbox(co,task);scores=sorted((max(abs(x-y) for x,y in zip(target,source_box(a,ims[tid],True))),a['id'],a) for a in anns[tid] if a['id'] not in used)
       if scores and scores[0][0]<0.004 and (len(scores)==1 or scores[1][0]-scores[0][0]>0.001):
        err,aid,matched=scores[0];used.add(aid);summary['taco_geometry_matches']+=1
      name=oldname;meta={'line':idx+1,'original_class':oldname,'class_name':name,'visual_review':'pending'}
      if matched:
       cat=matched['category_id'];meta.update(source_category=cats[cat],source_annotation_id=matched['id'],geometry_max_abs_error=err,attributes=attributes(cat),proposed_class=MAP.get(cat))
       if cat in EXCEPTIONS:
        name=EXCEPTIONS[cat];cid=NAMES[task].index(name);meta['class_name']=name
        if name!=oldname:changes.append({'task':task,'split':split,'image':ip.name,'line':idx+1,'from':oldname,'to':name,'source_annotation_id':matched['id'],'source_category':cats[cat],'method':'source_category_and_geometry_material_item_mapping','visual_review':'pending'});summary['changed_instances']+=1
       if cat not in MAP:meta['review_reason']='unmapped_or_out_of_scope_item'
      new.append(str(cid)+' '+' '.join(f'{x:.8f}' for x in co));instances.append(meta)
     link(ip,out/'images'/split/ip.name);write(out/'labels'/split/(ip.stem+'.txt'),'\n'.join(new)+('\n' if new else ''))
     gm=groupmeta.get(st,{});group=gm.get('split_group',st);groups[str(group)].add(split)
     r={'image':str((out/'images'/split/ip.name).relative_to(ROOT)),'source_image':str(ip),'source_label':str(lp),'split':split,'split_group':group,'source_family':gm.get('family','COCO_background' if st.startswith('bg_coco_') else 'unknown'),'generated_augmentation':gm.get('generated_augmentation',False),'status':'format_checked_semantic_review_pending','instances':instances}
     manifest.write(json.dumps(r,ensure_ascii=False)+'\n');summary[f'{split}_images']+=1;summary['instances']+=len(new)
     if summary.total()%15000==0:print(task,dict(summary),flush=True)
  summary['split_group_leaks']=sum(len(s)>1 for s in groups.values());stats[task]=dict(summary)
  names='\n'.join(f'  {i}: {n}' for i,n in enumerate(NAMES[task]))
  write(out/'STAGED_NOT_FULLY_REVIEWED.yaml',f'# Provisional: semantic review incomplete. Original task-specific IDs retained.\npath: {out.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n')
 downloaded=json.loads((workspace_path('KOREA_WASTE_RULES_20260913_v1', 'sources/taco/download_manifest.json')).read_text('utf-8'));newrows=[];seen=set();accepted_names=set()
 visual_path=workspace_path('KOREA_WASTE_RULES_20260913_v1', 'annotations/taco_visual_sample_review.json')
 visual_holds={r['id']:r['status'] for r in json.loads(visual_path.read_text('utf-8')) if r['status'].startswith('hold_')} if visual_path.exists() else {}
 similar_path=workspace_path('KOREA_WASTE_RULES_20260913_v1', 'reports/taco_similarity_candidates.json')
 similar={r['id'] for r in json.loads(similar_path.read_text('utf-8'))['matches']} if similar_path.exists() else set()
 for r in downloaded:
  if r['status']!='downloaded':continue
  tid=r['source_image_id'];im=ims[tid];aa=anns[tid];raw=workspace_path('KOREA_WASTE_RULES_20260913_v1', r['local_path']);why=[]
  if r['id'] in visual_holds:why.append(visual_holds[r['id']])
  if r['id'] in similar:why.append('legacy_near_duplicate_candidate_requires_review')
  if r['sha256'] in seen:why.append('exact_duplicate_new_source')
  seen.add(r['sha256'])
  needs_orientation=False
  if (r['width'],r['height'])!=(im['width'],im['height']):
   with Image.open(raw) as pil:
    if ImageOps.exif_transpose(pil).size==(im['width'],im['height']):needs_orientation=True
    else:why.append('source_dimension_mismatch')
  if any(a['category_id'] not in MAP for a in aa):why.append('unmapped_or_out_of_scope_object_in_image')
  if any(len(a.get('segmentation',[]))!=1 for a in aa):why.append('multipart_or_missing_polygon_requires_review')
  for a in aa:
   flat=[v/(im['width'] if j%2==0 else im['height']) for poly in a.get('segmentation',[]) for j,v in enumerate(poly)]
   if not all(math.isfinite(x) and 0<=x<=1 for x in source_box(a,im,False)+flat):why.append('source_coordinates_out_of_bounds');break
  split='train' # Novel external source reserved for training; no fabricated independent test split.
  row=dict(r,split=split,instances=[dict(source_annotation_id=a['id'],source_category=cats[a['category_id']],proposed_class=MAP.get(a['category_id']),bbox_xywh=a['bbox'],attributes=attributes(a['category_id'])) for a in aa],hold_reasons=why,status='held_for_review' if why else 'source_label_mapped_visual_review_pending')
  if not why:
   if needs_orientation:
    prepared=workspace_path('KOREA_WASTE_RULES_20260913_v1', 'prepared/taco', f"{r['id']}.jpg");prepared.parent.mkdir(parents=True,exist_ok=True)
    if not prepared.exists():
     with Image.open(raw) as pil:ImageOps.exif_transpose(pil).convert('RGB').save(prepared,quality=95)
    raw=prepared;row['preprocessing']='EXIF_orientation_normalized_JPEG95_original_preserved'
   row['training_image_path']=str(raw.relative_to(ROOT));row['annotation_coordinate_frame']='source_annotated_upright'
   accepted_names.add(f"new_{r['id']}")
   for task in OLD:
    out=workspace_path('KOREA_WASTE_RULES_20260913_v1', 'staged_'+task);name=f"new_{r['id']}";lines=[]
    for a in aa:
     cid=NAMES[task].index(MAP[a['category_id']])
     co=source_box(a,im,False) if task=='bbox' else [v/(im['width'] if j%2==0 else im['height']) for j,v in enumerate(a['segmentation'][0])]
     if not all(math.isfinite(x) and 0<=x<=1 for x in co):raise ValueError(f'Invalid source coords {a["id"]}')
     lines.append(str(cid)+' '+' '.join(f'{x:.8f}' for x in co))
    link(raw,out/'images'/split/(name+raw.suffix));write(out/'labels'/split/(name+'.txt'),'\n'.join(lines)+'\n')
   source_counts['mapped_images']+=1;source_counts['mapped_instances']+=len(aa)
  else:source_counts['held_images']+=1
  newrows.append(row)
 # 중단된 빌드가 남긴 오래된 신규 출처 생성물만 지운다.
 for task in OLD:
  for kind in ('images','labels'):
   target=(workspace_path('KOREA_WASTE_RULES_20260913_v1', 'staged_'+task, kind, 'train')).resolve()
   for p in target.glob('new_taco_*'):
    if p.stem not in accepted_names:
     assert p.resolve().parent==target and ROOT.resolve() in p.resolve().parents
     p.unlink()
 dump(workspace_path('KOREA_WASTE_RULES_20260913_v1', 'annotations/new_taco_instances.json'),newrows);dump(workspace_path('KOREA_WASTE_RULES_20260913_v1', 'annotations/legacy_rule_changes.json'),changes)
 dump(workspace_path('KOREA_WASTE_RULES_20260913_v1', 'policy/taco_category_mapping.json'),[{'id':i,'source_category':n,'proposed_class':MAP.get(i),'legacy_change_allowed':i in EXCEPTIONS,'attributes':attributes(i)} for i,n in cats.items()])
 dump(workspace_path('KOREA_WASTE_RULES_20260913_v1', 'reports/build_summary.json'),{'label_policy':'material_item_not_disposal_route','legacy':stats,'new_taco':dict(source_counts),'errors':errors,'release_status':'STAGED_SEMANTIC_REVIEW_INCOMPLETE','new_source_split':'train_only','legacy_images_are_hardlinks':'Do not edit image bytes in place; labels are independent files.'})
 print(json.dumps({'legacy':stats,'new':dict(source_counts),'errors':len(errors)},indent=2),flush=True)
if __name__=='__main__':main()
