"""데이터셋·결과를 읽기만 하며 점검한다. 자기 JSON 근거 파일만 기록한다."""
# 현재 작업 디렉토리와 상관없이 공용 작업공간 경로를 찾는다.
from pathlib import Path as _WorkspacePath
import sys as _workspace_sys
_workspace_root = next(p for p in _WorkspacePath(__file__).resolve().parents if (p / 'workspace_paths.py').is_file())
_workspace_sys.path.insert(0, str(_workspace_root))
from workspace_paths import workspace_path

from pathlib import Path
from collections import Counter, defaultdict
import csv, gzip, json, math, re, statistics, hashlib

BASE = workspace_path('')
SPLITS = ('train', 'val', 'test')
ROOTS = {'bbox': workspace_path('', 'YOLO_7CLASS_10000_PER_CLASS_20260909'),
         'seg': workspace_path('', 'YOLO_SEG_7CLASS_10000_PER_CLASS_20260910_yolo_seg')}
NAMES = {'bbox': ['paper','plastic','can','glass','battery','vinyl','general'],
         'seg': ['general','can','plastic','paper','glass','vinyl','battery']}
def stem(s):
    return re.sub(r'^images__(train|val|test)__', '', Path(s).stem)
def readcsv(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return [{k.strip(): v.strip() for k,v in r.items()} for r in csv.DictReader(f)]
manifest = {}
with gzip.open(ROOTS['bbox']/'selection_manifest.jsonl.gz','rt',encoding='utf-8') as f:
    for line in f:
        r=json.loads(line); manifest[stem(r['image'])]=r
report = {'manifest_records':len(manifest), 'datasets':{}, 'runs':{}, 'sampling':{}, 'lists':{}}
where, labels = {}, {}
for fam, root in ROOTS.items():
    where[fam], labels[fam] = {}, {}
    ds = {'splits':{}, 'errors':Counter(), 'error_examples':[], 'duplicate_stems':0}
    groups=defaultdict(set); hashes=defaultdict(set); old=0
    for split in SPLITS:
        imgs={p.stem:p for p in (root/'images'/split).iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png')}
        labs={p.stem:p for p in (root/'labels'/split).glob('*.txt')}
        stats={'images':len(imgs), 'labels':len(labs), 'pair_mismatch':len(set(imgs)^set(labs)),
               'backgrounds':0,'empty_non_bg':0,'instances':Counter(),'class_images':Counter(),
               'four_point':Counter(),'axis_aligned_four_point':Counter(),'source_images':Counter(),
               'small_under_1pct':Counter(),'tiny_under_03pct':Counter(),
               'source_instances':defaultdict(Counter),'augmented':0,'groups':0}
        sg=set()
        for st in imgs:
            key=stem(st)
            if key in where[fam]: ds['duplicate_stems']+=1
            where[fam][key]=split
            m=manifest.get(key)
            source=m['family'] if m else ('COCO_background' if st.startswith('bg_coco_') else 'UNKNOWN')
            stats['source_images'][source]+=1
            if m:
                groups[m['split_group']].add(split); sg.add(m['split_group'])
                hashes[m['image_sha256']].add(split)
                stats['augmented']+=bool(m.get('generated_augmentation'))
                old+=m['split']!=split
            if st not in labs: continue
            lines=[x.split() for x in labs[st].read_text(encoding='utf-8').splitlines() if x.strip()]
            bg=st.startswith('bg_coco_')
            stats['backgrounds']+=bg
            stats['empty_non_bg']+=not lines and not bg
            if bg and lines: ds['errors']['nonempty_bg']+=1
            seen=set(); parsed=[]
            for i, fields in enumerate(lines,1):
                try:
                    c=int(fields[0]); coords=list(map(float,fields[1:]))
                    assert 0<=c<7, 'class_range'
                    assert all(math.isfinite(v) for v in coords), 'nonfinite'
                    assert all(0<=v<=1 for v in coords), 'coordinate_range'
                    assert (len(coords)==4 if fam=='bbox' else len(coords)>=6 and len(coords)%2==0), 'coordinate_count'
                    if fam=='bbox':
                        assert coords[2]>0 and coords[3]>0, 'nonpositive_size'
                        stats['small_under_1pct'][NAMES[fam][c]]+=coords[2]*coords[3]<0.01
                        stats['tiny_under_03pct'][NAMES[fam][c]]+=coords[2]*coords[3]<0.003
                    else:
                        pts=list(zip(coords[::2],coords[1::2]))
                        area=abs(sum(x*pts[(j+1)%len(pts)][1]-y*pts[(j+1)%len(pts)][0] for j,(x,y) in enumerate(pts)))/2
                        assert area>1e-12, 'zero_polygon_area'
                        if len(pts)==4:
                            stats['four_point'][NAMES[fam][c]]+=1
                            if len(set(coords[::2]))==2 and len(set(coords[1::2]))==2 and len(set(pts))==4:
                                stats['axis_aligned_four_point'][NAMES[fam][c]]+=1
                    name=NAMES[fam][c]; seen.add(name); stats['instances'][name]+=1
                    stats['source_instances'][source][name]+=1; parsed.append((name,coords))
                except (ValueError,AssertionError) as e:
                    ds['errors'][str(e) or type(e).__name__]+=1
                    if len(ds['error_examples'])<12: ds['error_examples'].append(f'{fam}/{split}/{st}:{i}: {e}')
            stats['class_images'].update(seen); labels[fam][key]=parsed
        stats['groups']=len(sg); ds['splits'][split]=stats
        print(f'{fam}/{split}: {len(imgs)} images, {sum(stats["instances"].values())} instances',flush=True)
    ds['cross_split_groups']=sum(len(s)>1 for s in groups.values())
    ds['cross_split_manifest_hashes']=sum(len(s)>1 for s in hashes.values())
    ds['manifest_split_diff']=old
    report['datasets'][fam]=ds
common=set(where['bbox'])&set(where['seg'])
report['cross_dataset']={'common_stems':len(common), 'split_matrix':dict(Counter(where['bbox'][s]+'->'+where['seg'][s] for s in common)),
    'class_multiset_mismatch_images':sum(Counter(c for c,_ in labels['bbox'][s])!=Counter(c for c,_ in labels['seg'][s]) for s in common),
    'both_test_labeled':sum(where['bbox'][s]==where['seg'][s]=='test' and not s.startswith('bg_coco_') for s in common)}
report['cross_dataset']['class_mismatch_examples']=[{'stem':s,'bbox':dict(Counter(c for c,_ in labels['bbox'][s])),
    'seg':dict(Counter(c for c,_ in labels['seg'][s]))} for s in sorted(common)
    if Counter(c for c,_ in labels['bbox'][s])!=Counter(c for c,_ in labels['seg'][s])][:10]
for fam in ROOTS:
    for split in SPLITS:
        p=workspace_path('', 'best_train/data_nobg', f'{fam}_{split}.txt')
        paths=[Path(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
        actual={stem(p) for p in paths}; expected={s for s,v in where[fam].items() if v==split and not s.startswith('bg_coco_')}
        report['lists'][f'{fam}_{split}']={'n':len(paths),'duplicates':len(paths)-len(actual),'mismatch':len(actual^expected),'missing_paths':sum(not p.is_file() for p in paths)}
for p in (workspace_path('', 'best_train/runs')).glob('*/results.csv'):
    rs=readcsv(p)
    if not rs: continue
    b='metrics/mAP50-95(B)'; m='metrics/mAP50-95(M)'
    best=max(rs,key=lambda r:float(r[b])+float(r.get(m,0)))
    bestbox=max(rs,key=lambda r:float(r[b]))
    epochs=[int(float(r['epoch'])) for r in rs]; times=[float(r['time']) for r in rs]
    report['runs'][p.parent.name]={'rows':len(rs),'last_epoch':epochs[-1], 'duplicate_epochs':len(epochs)-len(set(epochs)),
      'best_fitness_epoch':best['epoch'],'best_box_epoch':bestbox['epoch'],'best_box':bestbox[b],
      'box_at_best_fitness':best[b], 'mask_at_best_fitness':best.get(m), 'time_last_s':times[-1],
      'time_decreases':sum(y<x for x,y in zip(times,times[1:])),
      'summary_exists':(p.parent/'summary.txt').exists(),'best_exists':(p.parent/'weights/best.pt').exists(),
      'loss_columns':[k for k in rs[0] if k.startswith('train/')],
      'first_losses':{k:v for k,v in rs[0].items() if k.startswith('train/')}}
for p in (workspace_path('', 'best_train/runs/imgsz_scan')).glob('*_val.txt'):
    fam='seg' if p.name.startswith('seg') else 'bbox'
    paths=[Path(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
    sample=Counter(); classimgs=Counter()
    for x in paths:
        st=stem(x); sample[manifest[st]['family'] if st in manifest else 'COCO_background']+=1
        classimgs.update(set(c for c,_ in labels[fam][st]))
    full=sorted((ROOTS[fam]/'images/val').glob('*.jpg'))
    if 'nobg' in p.name: full=[x for x in full if not x.name.startswith('bg_coco_')]
    selected=set(x.name for x in paths); last=max(i for i,x in enumerate(full) if x.name in selected)
    omitted=Counter(manifest[stem(x)]['family'] if stem(x) in manifest else 'COCO_background' for x in full[last+1:])
    report['sampling'][p.stem]={'n':len(paths),'source_images':sample,'class_images':classimgs,'full_n':len(full),
      'last_sample_index':last,'unreachable_tail':len(full)-last-1,'tail_sources':omitted}
out=Path(__file__).resolve().with_suffix('.json')
out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('Evidence:',out,flush=True)
