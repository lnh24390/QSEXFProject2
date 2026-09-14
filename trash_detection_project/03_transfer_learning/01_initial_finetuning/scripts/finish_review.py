"""직접 눈으로 확인한 내용만 기록한다. 검색 결과를 자동으로 승인하지 않는다."""
# 현재 작업 디렉토리와 상관없이 공용 작업공간 경로를 찾는다.
from pathlib import Path as _WorkspacePath
import sys as _workspace_sys
_workspace_root = next(p for p in _WorkspacePath(__file__).resolve().parents if (p / 'workspace_paths.py').is_file())
_workspace_sys.path.insert(0, str(_workspace_root))
from workspace_paths import workspace_path

from pathlib import Path
import json, os
from PIL import Image, ImageDraw
ROOT=workspace_path('KOREA_WASTE_RULES_20260913_v1')
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
rows=json.loads((workspace_path('KOREA_WASTE_RULES_20260913_v1', 'reports/commons_review_index.json')).read_text('utf-8'))
# 전체 이미지를 눈으로 보고 직접 친 박스를 정규화한 값이다.
specs={0:('plastic',[.145,.008,.835,.954],'한국 소주 플라스틱 소형병; 설명에 plastic flask 명시. 사용 중 제품 보조 사례.'),
13:('can',[.096,.052,.932,.939],'금속 왕관 병뚜껑. 태극기 무늬를 국내 촬영 근거로 사용하지 않음.'),
31:('can',[.243,.237,.716,.781],'도로 위 압착된 음료캔. 은색 바닥과 주황색 몸체가 한 객체.'),
63:('plastic',[.221,.075,.822,.923],'불투명 요구르트 용기. 미개봉 제품 보조 사례.'),
66:('paper',[.135,.093,.953,1.0],'휴지 롤과 연결된 종이. 하단 프레임에 일부 잘림.'),
70:('paper',[.025,.080,.899,.885],'휴지 롤과 연결된 종이 한 객체.')}
anns=[];review=[]
reject={8,9,10,11,12,17,19,20,21,22,23,24,25,26,27,28,29,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,50,53,54,58,61,62,67,68,69}
for i,r in enumerate(rows):
 state='bbox_visually_annotated' if i in specs else 'excluded_from_current_selection' if i in reject else 'candidate_needs_full_object_annotation'
 review.append({'id':r['id'],'review_level':'contact_sheet_relevance_screening','status':state,'note':'검색어를 정답으로 사용하지 않음. 제외는 이 수집 목적의 선별 결과이며 재질 정답 판정이 아님.'})
 if i not in specs:continue
 name,box,note=specs[i];x1,y1,x2,y2=box;cls=['paper','plastic','can','glass','battery','vinyl','general'].index(name)
 ann={'id':r['id'],'source':r,'class_name':name,'bbox_xyxy_normalized':box,'review_method':'assistant_full_image_visual_box_annotation','note':note,'split':'train','segmentation':'not_annotated','bbox_precision':'manual_visual_approximation','korea_product_confirmed':True if i==0 else None,'korea_capture_confirmed':None}
 anns.append(ann)
 for folder in ('reviewed_bbox','staged_bbox'):
  target=workspace_path('KOREA_WASTE_RULES_20260913_v1', folder);ip=target/'images/train'/f"new_{r['id']}.jpg";lp=target/'labels/train'/f"new_{r['id']}.txt"
  ip.parent.mkdir(parents=True,exist_ok=True);lp.parent.mkdir(parents=True,exist_ok=True)
  if not ip.exists():os.link(workspace_path('KOREA_WASTE_RULES_20260913_v1', r['local_path']),ip)
  lp.write_text(f'{cls} {(x1+x2)/2:.8f} {(y1+y2)/2:.8f} {x2-x1:.8f} {y2-y1:.8f}\n',encoding='utf-8')
dump(workspace_path('KOREA_WASTE_RULES_20260913_v1', 'annotations/commons_manual_bbox.json'),anns);dump(workspace_path('KOREA_WASTE_RULES_20260913_v1', 'annotations/commons_screening.json'),review)
picked=json.loads((workspace_path('KOREA_WASTE_RULES_20260913_v1', 'reports/taco_review_index.json')).read_text('utf-8'));observations=[]
accepted={0,1,3,4,13,16,17,18,19,20}
for i,r in enumerate(picked):
 status='sample_class_and_bbox_visual_check' if i in accepted else 'hold_incomplete_objects' if i==14 else 'needs_full_resolution_review'
 observations.append({'id':r['id'],'source_annotation_id':r['ann']['id'],'status':status,'review_level':'contact_sheet','note':'마스크 경계의 정밀 검수는 하지 않음.' if i in accepted else '신발 외에 추가 작은 쓰레기로 의심되는 물체가 보임. 이미지 전체 객체 재검수.' if i==14 else '작은 객체 또는 모호한 품목으로 축소본만으로 확정하지 않음.'})
dump(workspace_path('KOREA_WASTE_RULES_20260913_v1', 'annotations/taco_visual_sample_review.json'),observations)
write=workspace_path('KOREA_WASTE_RULES_20260913_v1', 'reviewed_bbox/TRAIN_ONLY_NOT_AN_EVALUATION_SET.txt');write.write_text('6장의 직접 박스 라벨링 표본. 전체 7클래스 학습·독립 평가용 데이터셋이 아닙니다. 원본 출처는 annotations/commons_manual_bbox.json에 있습니다.\n',encoding='utf-8')
# 로컬 검수 시트에는 직접 라벨링한 6장만 들어 있다.
sheet=Image.new('RGB',(1200,800),'white');dr=ImageDraw.Draw(sheet)
for j,a in enumerate(anns):
 im=Image.open(workspace_path('KOREA_WASTE_RULES_20260913_v1', a['source']['local_path'])).convert('RGB');im.thumbnail((390,350));xx=(j%3)*400;yy=(j//3)*400;sheet.paste(im,(xx,yy));x1,y1,x2,y2=a['bbox_xyxy_normalized'];dr.rectangle((xx+x1*im.width,yy+y1*im.height,xx+x2*im.width,yy+y2*im.height),outline='red',width=2);dr.text((xx+4,yy+355),a['id']+' '+a['class_name'],fill='black')
sheet.save(workspace_path('KOREA_WASTE_RULES_20260913_v1', 'reports/manual_bbox_qa.jpg'))
print('Commons manually annotated:',len(anns),'TACO visual sample:',len(observations))
