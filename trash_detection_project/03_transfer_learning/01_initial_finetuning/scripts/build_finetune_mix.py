"""공통 train 을 먼저 쓰는 복습 선택. 과제별 원래 클래스 ID 를 쓰고 평가 분할은 건드리지 않는다."""
# 현재 작업 디렉토리와 상관없이 공용 작업공간 경로를 찾는다.
from pathlib import Path as _WorkspacePath
import sys as _workspace_sys
_workspace_root = next(p for p in _WorkspacePath(__file__).resolve().parents if (p / 'workspace_paths.py').is_file())
_workspace_sys.path.insert(0, str(_workspace_root))
from workspace_paths import workspace_path

from pathlib import Path
import json,gzip,re,hashlib,collections,random,yaml
R=workspace_path('KOREA_WASTE_RULES_20260913_v1');OUT=workspace_path('KOREA_WASTE_RULES_20260913_v1', 'finetune');OUT.mkdir(exist_ok=True)
NAMES={'bbox':['paper','plastic','can','glass','battery','vinyl','general'],'seg':['general','can','plastic','paper','glass','vinyl','battery']}
def key(p):return re.sub(r'^images__(train|val|test)__','',p.stem)
def order(k):return hashlib.sha256(('replay73'+k).encode()).hexdigest()
def main():
 maps={t:{s:{key(p):p for p in (workspace_path('KOREA_WASTE_RULES_20260913_v1', f'training_640_{t}', 'images', s)).glob('*.jpg')} for s in ['train','val','test']} for t in NAMES}
 oldmaps={t:{s:{key(p) for p in (workspace_path('.', 'YOLO_7CLASS_10000_PER_CLASS_20260909' if t=='bbox' else 'YOLO_SEG_7CLASS_10000_PER_CLASS_20260910_yolo_seg', 'images', s)).iterdir() if p.suffix.lower() in ['.jpg','.png','.jpeg']} for s in ['train','val','test']} for t in NAMES}
 common=oldmaps['bbox']['train']&oldmaps['seg']['train'];meta={}
 with gzip.open(workspace_path('.', 'YOLO_7CLASS_10000_PER_CLASS_20260909/selection_manifest.jsonl.gz'),'rt',encoding='utf-8') as f:
  for line in f:
   r=json.loads(line);k=key(Path(r['image']));meta[k]={'classes':{NAMES['bbox'][int(c)] for c,v in r['objects'].items() if v},'family':r['family'],'generated':r.get('generated_augmentation',False),'group':r.get('split_group',k)}
 new={t:sorted(set(maps[t]['train'])-oldmaps[t]['train'],key=order) for t in NAMES}
 # 격리 후보는 출처 후보로 남긴다. 조용히 배경으로 라벨하지 않는다.
 blocked=json.loads((workspace_path('KOREA_WASTE_RULES_20260913_v1', 'reports/finetune_leakage_screen.json')).read_text('utf-8')) if (workspace_path('KOREA_WASTE_RULES_20260913_v1', 'reports/finetune_leakage_screen.json')).exists() else {}
 for t in NAMES:new[t]=[k for k in new[t] if k not in set(blocked.get(t,{}).get('quarantined_new_ids',[]))]
 target=2*sum(bool((workspace_path('KOREA_WASTE_RULES_20260913_v1', 'training_640_bbox/labels/train', maps['bbox']['train'][k].stem+'.txt')).read_text('utf-8').strip()) for k in new['bbox']);quota=target//7
 chosen=set();groups=collections.Counter();counts=collections.Counter();families=collections.Counter();pool={k for k in common if k in meta and meta[k]['classes']}
 def add(k):
  chosen.add(k);groups[meta[k]['group']]+=1;families[meta[k]['family']]+=1
  counts.update(meta[k]['classes'])
 for cls in sorted(NAMES['bbox'],key=lambda c:sum(c in meta[k]['classes'] for k in pool)):
  candidates=sorted((k for k in pool if cls in meta[k]['classes']),key=lambda k:(meta[k]['generated'],order(k)))
  # 실제/생성 구분 안에서 출처 계열을 번갈아 고른다.
  buckets=collections.defaultdict(list)
  for k in candidates:buckets[(meta[k]['generated'],meta[k]['family'])].append(k)
  for generated_tier in [False,True]:
   while counts[cls]<quota:
    added=False
    for b in sorted(b for b in buckets if b[0]==generated_tier):
     while buckets[b]:
      k=buckets[b].pop()
      if k in chosen or groups[meta[k]['group']]>=2:continue
      add(k);added=True;break
     if counts[cls]>=quota:break
    if not added:break
 remaining=sorted(pool-chosen,key=lambda k:(meta[k]['generated'],sum(counts[c] for c in meta[k]['classes'])/len(meta[k]['classes']),order(k)))
 for k in remaining:
  if len(chosen)>=target:break
  if groups[meta[k]['group']]<2:add(k)
 bgs=sorted([k for k in common if k.startswith('bg_coco_')],key=order)[:max(300,round((len(chosen)+len(new['bbox']))*.07))]
 reports={};all_selected={}
 for t in NAMES:
  native=[];have=collections.Counter(c for k in chosen for c in meta[k]['classes'])
  # 공통 복습 할당량을 못 채운 클래스에만 과제 전용 train 을 쓴다.
  for cls in NAMES['bbox']:
   for k in sorted(oldmaps[t]['train']-common,key=order):
    if have[cls]>=quota:break
    if k not in meta or cls not in meta[k]['classes'] or k in native:continue
    native.append(k);have.update(meta[k]['classes'])
  selected=[(k,'common_original') for k in sorted(chosen,key=order)]+[(k,'task_specific_original') for k in native]+[(k,'common_background') for k in bgs]+[(k,'new_source') for k in new[t]]
  # 세그 추가분이 적다. 새 이미지 파일을 만들지 않고 최대 세 번까지만, 기록을 남기고 반복한다.
  if t=='seg':selected += [(k,'new_source_repeat') for k in new[t]]*2
  def actual_classes(k):
   p=maps[t]['train'][k];lp=p.parent.parent.parent/'labels/train'/(p.stem+'.txt')
   return {NAMES[t][int(line.split()[0])] for line in lp.read_text('utf-8').splitlines() if line.strip()}
  general_real=[k for k in sorted(chosen,key=order) if not meta[k]['generated'] and 'general' in actual_classes(k)][:400]
  selected += [(k,'real_general_replay_repeat') for k in general_real]
  if t=='bbox':
   general_new=[k for k in new[t] if 'general' in actual_classes(k)][:100]
   packaging=[k for k in new[t] if k.startswith(('new_taco_','warp_')) and actual_classes(k)&{'paper','vinyl'}][:200]
   hard_bg=[k for k in new[t] if not actual_classes(k)]
   selected += [(k,'new_general_repeat') for k in general_new]*2
   selected += [(k,'packaging_repeat') for k in packaging]
   selected += [(k,'new_background_repeat') for k in hard_bg]*2
  random.Random(73).shuffle(selected);records=[];obj=collections.Counter();class_images=collections.Counter();membership=set()
  for k,kind in selected:
   assert k in maps[t]['train'] and k not in oldmaps[t]['val'] and k not in oldmaps[t]['test']
   p=maps[t]['train'][k];lp=p.parent.parent.parent/'labels/train'/(p.stem+'.txt');cs=[]
   for line in lp.read_text('utf-8').splitlines():
    if line.strip():cs.append(NAMES[t][int(line.split()[0])])
   obj.update(cs);class_images.update(set(cs));membership.add(k);records.append({'id':k,'path':str(p),'kind':kind,'classes':sorted(set(cs))})
  (workspace_path('KOREA_WASTE_RULES_20260913_v1/finetune', f'{t}_train.txt')).write_text('\n'.join(r['path'] for r in records)+'\n',encoding='utf-8')
  for split in ['val','test']:(workspace_path('KOREA_WASTE_RULES_20260913_v1/finetune', f'{t}_{split}.txt')).write_text('\n'.join(str(maps[t][split][k]) for k in sorted(oldmaps[t][split]))+'\n',encoding='utf-8')
  config={'path':str(workspace_path('KOREA_WASTE_RULES_20260913_v1', f'training_640_{t}')),'train':str(workspace_path('KOREA_WASTE_RULES_20260913_v1/finetune', f'{t}_train.txt')),'val':str(workspace_path('KOREA_WASTE_RULES_20260913_v1/finetune', f'{t}_val.txt')),'test':str(workspace_path('KOREA_WASTE_RULES_20260913_v1/finetune', f'{t}_test.txt')),'names':dict(enumerate(NAMES[t]))}
  (workspace_path('KOREA_WASTE_RULES_20260913_v1/finetune', f'{t}.yaml')).write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8');(workspace_path('KOREA_WASTE_RULES_20260913_v1/finetune', f'{t}_selection.json')).write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8');all_selected[t]=membership
  reports[t]={'common_train_pool':len(common),'target_old_labeled':target,'common_original_images':len(chosen),'task_specific_original_images':len(native),'common_background_images':len(bgs),'new_unique_images':len(new[t]),'epoch_entries':len(records),'unique_train_images':len(membership),'source_entry_counts':dict(collections.Counter(r['kind'] for r in records)),'background_exposures_per_epoch':sum(not r['classes'] for r in records),'background_unique_images':len({r['id'] for r in records if not r['classes']}), 'general_real_repeat_count':len(general_real),'class_image_exposures_per_epoch':dict(class_images),'class_object_exposures_per_epoch':dict(obj),'original_generated_images':sum(meta[k]['generated'] for k in chosen|set(native)),'own_val_test_ids_in_train':0,'val_images':len(oldmaps[t]['val']),'test_images':len(oldmaps[t]['test'])}
 reports['common_selected_unique_images']=len(all_selected['bbox']&all_selected['seg']);reports['note']='Training only. Seg new sources repeated 3x total for sampling, not counted as independent new images. General real photos and packaging/hard backgrounds receive documented bounded extra exposure. Class IDs remain task-specific. Source-label and structural validation are not exhaustive visual review.'
 (workspace_path('KOREA_WASTE_RULES_20260913_v1', 'reports/finetune_mix.json')).write_text(json.dumps(reports,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(reports,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
