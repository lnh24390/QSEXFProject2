# 현재 작업 디렉토리와 상관없이 공용 작업공간 경로를 찾는다.
from pathlib import Path as _WorkspacePath
import sys as _workspace_sys
_workspace_root = next(p for p in _WorkspacePath(__file__).resolve().parents if (p / 'workspace_paths.py').is_file())
_workspace_sys.path.insert(0, str(_workspace_root))
from workspace_paths import workspace_path

from pathlib import Path
import subprocess,sys,json
R=workspace_path('KOREA_WASTE_RULES_20260913_v1')
for task,mode in [('bbox','offline'),('bbox','none'),('seg','native'),('seg','offline'),('seg','none')]:
 print('Start',task,mode,flush=True)
 with (workspace_path('KOREA_WASTE_RULES_20260913_v1', f'reports/benchmark_{task}_{mode}.log')).open('w',encoding='utf-8') as log:
  result=subprocess.run([sys.executable,str(workspace_path('KOREA_WASTE_RULES_20260913_v1', 'scripts/benchmark_mosaic.py')),mode,task],stdout=log,stderr=subprocess.STDOUT)
 if result.returncode:raise SystemExit(f'Benchmark failed: {task} {mode}; inspect log')
 print('Complete',task,mode,flush=True)
