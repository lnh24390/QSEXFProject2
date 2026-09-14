# 현재 작업 디렉토리와 상관없이 공용 작업공간 경로를 찾는다.
from pathlib import Path as _WorkspacePath
import sys as _workspace_sys
_workspace_root = next(p for p in _WorkspacePath(__file__).resolve().parents if (p / 'workspace_paths.py').is_file())
_workspace_sys.path.insert(0, str(_workspace_root))
from workspace_paths import workspace_path
_workspace_sys.path.insert(0, str(workspace_path("02_training")))
from model_analysis.schema import summary
from model_analysis.evaluation import evaluate_model

from pathlib import Path
import datetime,json,hashlib
R=workspace_path('STAGED_FINETUNING_20260913')
OLD=workspace_path('.', 'KOREA_WASTE_RULES_20260913_v1')
STAGES=['stage2','stage3','final']
def now():return datetime.datetime.now().isoformat(timespec='seconds')
def read(path):return json.loads(Path(path).read_text('utf-8'))
def write(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def event(kind,**values):
 with (workspace_path('STAGED_FINETUNING_20260913', 'reports/events.jsonl')).open('a',encoding='utf-8') as f:f.write(json.dumps(dict(time=now(),event=kind,**values),ensure_ascii=False)+'\n')
def state(**values):
 p=workspace_path('STAGED_FINETUNING_20260913', 'reports/state.json');d=read(p) if p.exists() else {};d.update(values,updated_at=now());write(p,d)
def score(metrics,task):return metrics['box']['map']+(metrics['seg']['map'] if task=='seg' else 0)
def promotion(reference,candidate,task):
 assert reference['names']==candidate['names'],'Cannot compare different native class orders'
 kinds=['box']+(['seg'] if task=='seg' else [])
 checks={'fitness_improved':score(candidate,task)>score(reference,task)+.0001}
 for kind in kinds:
  checks[kind+'_overall_not_lower']=candidate[kind]['map']>=reference[kind]['map']-.0001
  checks[kind+'_class_drop_within_1pp']=all(b>=a-.01 for a,b in zip(reference[kind]['per_class_map'],candidate[kind]['per_class_map'],strict=True))
 return {'accepted':all(checks.values()),'checks':checks,'note':'Validation rule fixed in advance; not a statistical significance claim.'}
