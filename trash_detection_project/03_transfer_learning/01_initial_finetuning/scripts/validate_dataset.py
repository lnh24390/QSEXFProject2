"""내보낸 라벨, 원본 보존, 허용된 클래스 수정, 분할 유지를 검증한다."""
from pathlib import Path
import sys
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p/'workspace_paths.py').is_file())))
from workspace_paths import workspace_path
from collections import Counter,defaultdict
import json,math,os
from PIL import Image
from build_dataset import ROOT,OLD,NAMES,MAP
errors=[];report={};changes=json.loads((ROOT/'annotations/legacy_rule_changes.json').read_text('utf-8'))
allowed={(x['task'],x['split'],x['image'],x['line']):x for x in changes}
for task,old in OLD.items():
 out=ROOT/('staged_'+task);stats=Counter();class_counts=Counter()
 for split in ('train','val','test'):
  imgs={p.stem:p for p in (out/'images'/split).iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png')};labs={p.stem:p for p in (out/'labels'/split).glob('*.txt')}
  if imgs.keys()!=labs.keys():errors.append([task,split,'image_label_pair_mismatch'])
  stats[split+'_images']=len(imgs)
  for stem,ip in imgs.items():
   lp=labs[stem];lines=[list(map(float,x.split())) for x in lp.read_text('utf-8').splitlines() if x.strip()]
   for v in lines:
    cls=int(v[0]);c=v[1:]
    if cls!=v[0] or not 0<=cls<7 or not all(math.isfinite(a) and 0<=a<=1 for a in c):errors.append([str(lp),'invalid_class_or_coordinate'])
    if task=='bbox':
     if len(c)!=4 or c[2]<=0 or c[3]<=0:errors.append([str(lp),'invalid_bbox'])
    elif len(c)<6 or len(c)%2:errors.append([str(lp),'invalid_polygon'])
    class_counts[NAMES[task][cls]]+=1
   stats['instances']+=len(lines)
   if not stem.startswith(('new_','warp_','oi_','gv2_','rvm_','hb_','coin_')):
    src=old/'images'/split/ip.name;sl=old/'labels'/split/lp.name
    if not src.exists() or not sl.exists():errors.append([str(ip),'legacy_split_or_filename_changed']);continue
    if not os.path.samefile(src,ip):errors.append([str(ip),'expected_legacy_image_hardlink'])
    if os.path.samefile(sl,lp):errors.append([str(lp),'labels_must_be_independent'])
    orig=[list(map(float,x.split())) for x in sl.read_text('utf-8').splitlines() if x.strip()]
    if len(orig)!=len(lines):errors.append([str(lp),'legacy_instance_count_changed']);continue
    for i,(a,b) in enumerate(zip(orig,lines),1):
     if len(a)!=len(b) or max([abs(x-y) for x,y in zip(a[1:],b[1:])]+[0])>1e-7:errors.append([str(lp),i,'legacy_geometry_changed'])
     if a[0]!=b[0]:
      change=allowed.get((task,split,ip.name,i))
      if not change or NAMES[task][int(b[0])]!=change['to']:errors.append([str(lp),i,'unlogged_class_change'])
   else:
    with Image.open(ip) as im:im.verify()
    stats['new_images']+=1
 report[task]={'counts':dict(stats),'class_instances':dict(class_counts)}
# 사용자가 분명히 정한 의미 규칙이다.
assert MAP[9]=='glass' and MAP[23]=='glass'
assert MAP[36]=='vinyl' and MAP[38]=='vinyl' and MAP[5]=='plastic'
assert MAP[31]=='paper' and MAP[1]=='battery'
report.update(errors=errors,passed=not errors,semantic_invariants='material/item preserved; no contamination or regional bin mapping',limitation='Structural/source checks are not a full visual material review. Perceptual screening covers known TACO only.')
workspace_path('KOREA_WASTE_RULES_20260913_v1/reports/validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2));raise SystemExit(0 if not errors else 1)
