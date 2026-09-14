# 현재 작업 디렉토리와 상관없이 공용 작업공간 경로를 찾는다.
from pathlib import Path as _WorkspacePath
import sys as _workspace_sys
_workspace_root = next(p for p in _WorkspacePath(__file__).resolve().parents if (p / 'workspace_paths.py').is_file())
_workspace_sys.path.insert(0, str(_workspace_root))
from workspace_paths import workspace_path

from pathlib import Path
import json,os,math,collections
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import yaml
R=workspace_path('KOREA_WASTE_RULES_20260913_v1')
NAMES={'bbox':['paper','plastic','can','glass','battery','vinyl','general'],'seg':['general','can','plastic','paper','glass','vinyl','battery']}
def convert(item):
 task,split,ip=item;out=workspace_path('KOREA_WASTE_RULES_20260913_v1', f'training_640_{task}');dst=out/'images'/split/(ip.stem+'.jpg');lp=workspace_path('KOREA_WASTE_RULES_20260913_v1', f'staged_{task}', 'labels', split, ip.stem+'.txt');dl=out/'labels'/split/lp.name
 with Image.open(ip) as im:
  w,h=im.size;r=min(640/w,640/h);nw,nh=round(w*r),round(h*r);left=(640-nw)//2;top=(640-nh)//2
  canvas=Image.new('RGB',(640,640),(114,114,114));canvas.paste(im.convert('RGB').resize((nw,nh),Image.Resampling.BILINEAR),(left,top))
  temp=dst.with_suffix('.tmp.jpg');canvas.save(temp,quality=85,optimize=True,subsampling=2);temp.replace(dst)
 lines=[]
 for line in lp.read_text('utf-8').splitlines():
  if not line.strip():continue
  v=list(map(float,line.split()));cls=int(v[0]);c=v[1:]
  if task=='bbox':x,y,bw,bh=c;c=[(x*nw+left)/640,(y*nh+top)/640,bw*nw/640,bh*nh/640]
  else:c=[(a*(nw if i%2==0 else nh)+(left if i%2==0 else top))/640 for i,a in enumerate(c)]
  assert all(math.isfinite(a) and 0<=a<=1 for a in c)
  lines.append(str(cls)+' '+' '.join(f'{a:.8f}' for a in c))
 dl.write_text('\n'.join(lines)+ ('\n' if lines else ''),encoding='utf-8')
 with Image.open(dst) as im:
  assert im.size==(640,640)
  assert not im.getexif() and not any(k in im.info for k in ['exif','icc_profile','xmp','comment'])
  assert all(m not in ['APP1','APP2','APP13','COM'] for m,_ in getattr(im,'applist',[]))
 return {'task':task,'split':split,'source_image':str(ip.relative_to(R)),'image':str(dst.relative_to(R)),'source_size':[w,h],'resize_size':[nw,nh],'padding_left_top':[left,top],'instances':len(lines),'image_640_verified':True,'metadata_removed_verified':True,'source_bytes':ip.stat().st_size,'output_bytes':dst.stat().st_size}
def main():
 jobs=[]
 for task in NAMES:
  for split in ['train','val','test']:
   out=workspace_path('KOREA_WASTE_RULES_20260913_v1', f'training_640_{task}')
   for kind in ['images','labels']:(out/kind/split).mkdir(parents=True,exist_ok=True)
   jobs.extend((task,split,p) for p in (workspace_path('KOREA_WASTE_RULES_20260913_v1', f'staged_{task}', 'images', split)).iterdir() if p.suffix.lower() in ['.jpg','.jpeg','.png'])
  (out/'STAGED_NOT_FULLY_REVIEWED.yaml').write_text(yaml.safe_dump({'path':str(out),'train':'images/train','val':'images/val','test':'images/test','names':dict(enumerate(NAMES[task]))},sort_keys=False),encoding='utf-8')
 counts=collections.Counter();resized=collections.Counter()
 with (workspace_path('KOREA_WASTE_RULES_20260913_v1', 'reports/resize_640_manifest.jsonl')).open('w',encoding='utf-8') as f,ThreadPoolExecutor(max_workers=6) as pool:
  for i,row in enumerate(pool.map(convert,jobs),1):
   f.write(json.dumps(row)+'\n');counts[row['task']+'_'+row['split']]+=1;counts['source_bytes']+=row['source_bytes'];counts['output_bytes']+=row['output_bytes']
   if row['source_size']!=[640,640]:resized[row['task']]+=1
   if i%10000==0:print('640 conversion verified',i,'/',len(jobs),flush=True)
 result={'counts':dict(counts),'resized_images':dict(resized),'all_images_verified_640':True,'metadata_removed_verified':True,'jpeg_quality':85,'optimize':True,'labels_transformed_with_actual_rounded_resize_dimensions':True,'originals_untouched':True,'semantic_review_incomplete':True}
 (workspace_path('KOREA_WASTE_RULES_20260913_v1', 'reports/resize_640_validation.json')).write_text(json.dumps(result,indent=2));print(result,flush=True)
if __name__=='__main__':main()
