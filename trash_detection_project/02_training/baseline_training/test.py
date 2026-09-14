"""기존 진입점을 유지하면서 공통 평가 형식으로 저장한다."""
from pathlib import Path
import argparse
import json
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'workspace_paths.py').is_file())
sys.path[:0] = [str(ROOT), str(ROOT/'02_training')]
from workspace_paths import workspace_path
from model_analysis.evaluation import evaluate_model, ensure_gpu_idle
from train import TASKS


def main():
    parser = argparse.ArgumentParser(description='초기 학습 모델 또는 외부 모델의 공통 test 평가')
    parser.add_argument('task', nargs='?', default='bbox', choices=list(TASKS))
    parser.add_argument('--model', help='선택 사항: 외부 .pt 경로')
    parser.add_argument('--data', help='선택 사항: 별도 평가 데이터 YAML')
    args = parser.parse_args()
    family = 'seg' if args.task.startswith('seg') else 'bbox'
    catalog = json.loads(workspace_path('model_analysis_code', 'models.json').read_text('utf-8'))
    weights = args.model or workspace_path('best_train/runs', TASKS[args.task]['name'], 'weights/best.pt')
    data = args.data or catalog['datasets'][family]
    ensure_gpu_idle()
    result = evaluate_model(weights, data, task=family, name=Path(weights).stem if args.model else TASKS[args.task]['name'],
                            source='external' if args.model else 'baseline_training')
    print(f'평가 결과: {result.analysis_record_path}')


if __name__ == '__main__':
    main()
