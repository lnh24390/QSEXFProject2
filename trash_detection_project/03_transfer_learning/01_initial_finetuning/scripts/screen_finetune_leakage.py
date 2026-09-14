"""새 train 과 기존 평가 자료의 유사도를 보수적으로 재어 겹치는 것을 격리한다."""
# 현재 작업 디렉토리와 상관없이 공용 작업공간 경로를 찾는다.
from pathlib import Path as _WorkspacePath
import sys as _workspace_sys
_workspace_root = next(p for p in _WorkspacePath(__file__).resolve().parents if (p / 'workspace_paths.py').is_file())
_workspace_sys.path.insert(0, str(_workspace_root))
from workspace_paths import workspace_path

from pathlib import Path
import json,re,hashlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
import cv2
R=workspace_path('KOREA_WASTE_RULES_20260913_v1')
def key(p):return re.sub(r'^images__(train|val|test)__','',p.stem)
def fingerprint(item):
 k,p=item
 with Image.open(p) as im:
  a=np.asarray(im.convert('L').resize((32,32)),dtype=np.float32);low=cv2.dct(a)[:8,:8].reshape(-1);ph=0
  for b in low>np.median(low[1:]):ph=(ph<<1)|int(b)
  a=np.asarray(im.convert('L').resize((9,8)));dh=0
  for b in (a[:,1:]>a[:,:-1]).reshape(-1):dh=(dh<<1)|int(b)
  rgb=np.asarray(im.convert('RGB').resize((1,1)))[0,0].tolist();digest=hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()
 return k,{'ph':ph,'dh':dh,'rgb':rgb,'sha':digest}
def main():
 paths={};evals={};news={}
 for t in ['bbox','seg']:
  base=workspace_path('KOREA_WASTE_RULES_20260913_v1', f'training_640_{t}');ev={key(p):p for s in ['val','test'] for p in (base/'images'/s).glob('*.jpg')};nr={key(p):p for p in (base/'images/train').glob('*.jpg') if p.stem.startswith(('new_','warp_','hb_','oi_','coin_','room_'))};paths.update(ev);paths.update(nr);evals[t]=ev;news[t]=nr
 fp={}
 with ThreadPoolExecutor(max_workers=6) as ex:
  for i,(k,v) in enumerate(ex.map(fingerprint,paths.items()),1):
   fp[k]=v
   if i%5000==0:print('similarity fingerprints',i,'/',len(paths),flush=True)
 lut=np.array([i.bit_count() for i in range(256)],dtype=np.uint8);result={}
 for t in ['bbox','seg']:
  ks=list(evals[t]);ph=np.array([fp[k]['ph'] for k in ks],dtype=np.uint64);dh=np.array([fp[k]['dh'] for k in ks],dtype=np.uint64);rgb=np.array([fp[k]['rgb'] for k in ks]);hashes={fp[k]['sha']:k for k in ks};hits=[]
  for k in news[t]:
   f=fp[k];pd=lut[np.bitwise_xor(ph,np.uint64(f['ph'])).view(np.uint8).reshape(-1,8)].sum(1);idx=np.where(pd<=4)[0]
   for j in idx:
    dd=(int(dh[j])^f['dh']).bit_count();cd=float(np.abs(rgb[j]-f['rgb']).mean())
    if dd<=4 and cd<=8:hits.append({'new_id':k,'evaluation_id':ks[j],'phash_distance':int(pd[j]),'dhash_distance':dd,'mean_color_distance':cd,'exact_decoded_pixels':f['sha']==fp[ks[j]]['sha']})
   if f['sha'] in hashes and not any(h['new_id']==k for h in hits):hits.append({'new_id':k,'evaluation_id':hashes[f['sha']],'exact_decoded_pixels':True})
  result[t]={'new_images_screened':len(news[t]),'evaluation_images_screened':len(ks),'quarantined_new_ids':sorted({h['new_id'] for h in hits}),'candidate_matches':hits,'limitation':'Conservative perceptual candidate exclusion; not exhaustive semantic near-duplicate proof. Quarantine does not claim duplicate identity.'};print(t,'quarantined',len(result[t]['quarantined_new_ids']),flush=True)
 (workspace_path('KOREA_WASTE_RULES_20260913_v1', 'reports/finetune_leakage_screen.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
if __name__=='__main__':main()
