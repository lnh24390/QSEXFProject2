"""데이터셋·학습·전이학습·결과의 공용 폴더 설정.

workspace_layout.json 의 키는 원래 프로젝트 기준의 자원을 가리킨다.
파일을 열기 전에 전체 경로를 해석하므로, 링크나 모델 파일 복제 없이
한 프로젝트의 코드와 산출물을 서로 다른 범주에 둘 수 있다.
"""
from pathlib import Path, PurePosixPath
import json
import re

WORKSPACE_ROOT = Path(__file__).resolve().parent
LAYOUT = json.loads((WORKSPACE_ROOT / 'workspace_layout.json').read_text(encoding='utf-8'))
_RULES = sorted(LAYOUT.items(), key=lambda item: len(item[0]), reverse=True)


def _legacy_names(raw):
    """기존 체크포인트의 날짜형 실험명을 현재 논리 키로 정규화한다."""
    if not any(token in raw for token in ('RETRAIN_V2_', 'STAGED_TRANSFER_', 'analysis')) and not raw.endswith('.md'):
        return raw
    parts = raw.split('/')
    reports = {
        '분석_Report_20260913.md': '프로젝트_분석_20260913.md',
        '새로운_전략_20260913.md': '학습_개선전략_20260913.md',
        '진행경과_및_데이터셋_구축보고서_20260913.md': '데이터셋_구축경과_20260913.md',
    }
    for i, part in enumerate(parts):
        if re.fullmatch(r'(?:[A-Za-z]+_)?RETRAIN_V2_20260913', part):
            parts[i] = 'MODEL_SCALING_20260913'
        elif re.fullmatch(r'(?:[A-Za-z]+_)?STAGED_TRANSFER_20260913', part):
            parts[i] = 'STAGED_FINETUNING_20260913'
        elif re.fullmatch(r'[A-Za-z]{3}_analysis', part):
            parts[i] = 'project_analysis'
        elif re.sub(r'^[A-Z]{3}_', '', part) in reports:
            parts[i] = reports[re.sub(r'^[A-Z]{3}_', '', part)]
    # 과거 루트 지침 파일에 대한 호환. 존재하는 파일은 원래 이름을 유지한다.
    if len(parts) == 1 and re.fullmatch(r'[A-Z]{6}\.md', parts[0]) and parts[0] != 'README.md':
        if not (WORKSPACE_ROOT / parts[0]).exists():
            parts[0] = 'PROJECT_GUIDE.md'
    return '/'.join(parts)


def workspace_path(*parts) -> Path:
    """프로젝트 자원이나 과거 경로에 대해 현재의 절대 경로를 돌려준다."""
    raw = '/'.join(str(p).replace('\\', '/') for p in parts if str(p))
    old_root = 'C:/Users/USER/Desktop/bbox_data'
    for prefix in (WORKSPACE_ROOT.as_posix(), old_root):
        if raw.lower() == prefix.lower():
            raw = ''
            break
        if raw.lower().startswith(prefix.lower() + '/'):
            raw = raw[len(prefix) + 1:]
            break
    # 직접 지정한 외부 경로(예: 읽기 전용 원본 백업)는 그대로 둔다.
    if Path(raw).is_absolute():
        return Path(raw)
    raw = str(PurePosixPath(raw)) if raw else ''
    if raw == '.':
        raw = ''
    raw = _legacy_names(raw)
    for old, new in _RULES:
        if raw.lower() == old.lower() or raw.lower().startswith(old.lower() + '/'):
            return WORKSPACE_ROOT / (new + raw[len(old):])
    return WORKSPACE_ROOT / raw


DATASETS = WORKSPACE_ROOT / '01_datasets'
TRAINING = WORKSPACE_ROOT / '02_training'
TRANSFER_LEARNING = WORKSPACE_ROOT / '03_transfer_learning'
RESULTS = WORKSPACE_ROOT / '04_results'
MODEL_ANALYSIS_CODE = workspace_path('model_analysis_code')
MODEL_ANALYSIS_RESULTS = workspace_path('model_analysis_results')
EXTERNAL_MODELS = workspace_path('external_models')
