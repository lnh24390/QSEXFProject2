"""등록된 체크포인트와 외부 체크포인트를 같은 절차로 평가한다."""
from pathlib import Path
import argparse
import json
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'workspace_paths.py').is_file())
sys.path[:0] = [str(ROOT), str(ROOT/'02_training')]
from workspace_paths import workspace_path
from model_analysis.evaluation import evaluate_model, dataset_snapshot, class_mapping, ensure_gpu_idle, validate_task_labels
from model_analysis.schema import compare_records, refresh_index
from model_analysis.history import import_history


def main(argv=None):
    parser = argparse.ArgumentParser(description='기존/외부 YOLO 모델 공통 분석')
    subs = parser.add_subparsers(dest='command', required=True)
    subs.add_parser('list', help='등록된 모델과 평가 데이터 확인')
    subs.add_parser('import-history', help='기존 측정 기록을 공통 형식으로 정리 (GPU 사용 없음)')
    evaluate = subs.add_parser('evaluate', help='모델을 순차 평가하고 같은 조건끼리 비교')
    evaluate.add_argument('--models', nargs='+', required=True, help='등록 모델 ID 또는 외부 .pt 절대 경로')
    evaluate.add_argument('--task', choices=['bbox', 'seg'], required=True)
    evaluate.add_argument('--split', choices=['val', 'test'], default='test')
    evaluate.add_argument('--data', help='선택 사항: 별도의 라벨된 평가 데이터 YAML')
    evaluate.add_argument('--class-map', help='외부 클래스 번호/이름을 데이터 클래스 이름에 대응하는 JSON')
    evaluate.add_argument('--device', default='0')
    evaluate.add_argument('--batch', type=int, default=16)
    evaluate.add_argument('--workers', type=int, default=4)
    evaluate.add_argument('--check-only', action='store_true', help='가중치·클래스·데이터 연결만 확인')
    comparison = subs.add_parser('compare')
    comparison.add_argument('--records', nargs='+', required=True)
    args = parser.parse_args(argv)
    catalog = json.loads((Path(__file__).parent/'models.json').read_text('utf-8'))
    if args.command == 'list':
        for key, entry in catalog['models'].items():
            print(f"{key}: {entry['task']} / {workspace_path(entry['path'])}")
        return
    if args.command == 'import-history':
        print(import_history()); refresh_index(workspace_path('model_analysis_results')); return
    if args.command == 'compare':
        print(compare_records(args.records, workspace_path('model_analysis_results')/'comparisons'))
        refresh_index(workspace_path('model_analysis_results')); return
    if not args.check_only:
        ensure_gpu_idle()
    data = workspace_path(args.data or catalog['datasets'][args.task])
    aliases = json.loads(Path(args.class_map).read_text('utf-8')) if args.class_map else None
    from ultralytics import YOLO
    entries = []
    for identifier in args.models:
        registered = catalog['models'].get(identifier)
        entry = registered or dict(path=identifier, task=args.task, source='external')
        if entry['task'] != args.task:
            raise ValueError(f'{identifier}: 다른 과제 모델은 같은 비교에 넣을 수 없습니다.')
        path = workspace_path(entry['path'])
        if not path.is_file() or path.suffix.lower() != '.pt':
            raise ValueError(f'.pt 파일이 없습니다: {path}')
        entries.append((identifier if registered else path.stem, path, entry))
    # 첫 GPU 평가를 시작하기 전에 모든 모델을 미리 점검한다.
    dataset, manifest = dataset_snapshot(data, args.split)
    for name, path, entry in entries:
        model = YOLO(str(path))
        expected = 'segment' if args.task == 'seg' else 'detect'
        if model.task != expected:
            raise ValueError(f'{path}: {model.task} 모델을 {args.task}로 평가할 수 없습니다.')
        validate_task_labels(manifest, model.task)
        mapping = class_mapping(model.names, dataset['names'], aliases)
        print(f"확인: {name}, {model.task}, {dataset['count']}장, 클래스 대응 {mapping}")
        del model
    if args.check_only:
        return
    paths = []
    for name, path, entry in entries:
        result = evaluate_model(path, data, split=args.split, name=name, task=args.task,
            source=entry.get('source', 'external'), class_map=aliases,
            device=args.device, batch=args.batch, workers=args.workers)
        paths.append(result.analysis_record_path)
        print(f'저장: {result.analysis_record_path}')
        del result
        import torch
        torch.cuda.empty_cache()
    if len(paths) > 1:
        print(f"비교: {compare_records(paths, workspace_path('model_analysis_results')/'comparisons')}")
    refresh_index(workspace_path('model_analysis_results'))


if __name__ == '__main__':
    main()
