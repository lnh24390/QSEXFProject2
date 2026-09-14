"""기존 측정값을 가져오기만 한다. 새로 평가한 것처럼 기록하지 않는다."""
from pathlib import Path
import csv
import json
from .schema import sha, digest, now, unique_dir, write_json, normalize_legacy, save_record, compare_records
from .evaluation import workspace_path


def training_summary(run):
    run = Path(run)
    with (run/'results.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(stream)]
    if not rows:
        return None
    is_seg = 'metrics/mAP50-95(M)' in rows[0]
    def score(row):
        return float(row['metrics/mAP50-95(B)'])+(float(row['metrics/mAP50-95(M)']) if is_seg else 0)
    best = max(rows, key=score)
    # 시간이 초기화된 지점과 첫 행은 안정 상태의 에포크 시간으로 볼 수 없다.
    durations = []
    resets = 0
    for previous, current in zip(rows, rows[1:]):
        delta = float(current.get('time', 0))-float(previous.get('time', 0))
        if delta > 0:
            durations.append(delta)
        else:
            resets += 1
    return dict(schema_version=1, run=str(run), source_csv_sha256=sha(run/'results.csv'),
                recorded_epochs=len(rows), last_epoch=int(float(rows[-1]['epoch'])),
                best_epoch=int(float(best['epoch'])), best_fitness=score(best),
                selection_rule='box_map + mask_map' if is_seg else 'box_map',
                best_epoch_metrics={k: float(v) for k, v in best.items() if k.startswith('metrics/')},
                last_losses={k: float(v) for k, v in rows[-1].items() if k.startswith(('train/', 'val/'))},
                mean_epoch_seconds_excluding_start_or_reset=sum(durations)/len(durations) if durations else None,
                timer_resets=resets, source='results.csv',
                note='학습 중 검증 기록입니다. test 평가 및 최종 best.pt 재검증과 구분합니다. '
                     '재시작 구간의 첫 시간이 양수인 경우 준비 시간이 평균에 포함될 수 있습니다.')


def import_history():
    folder = unique_dir(workspace_path('model_analysis_results')/'historical', 'existing_results')
    written = []
    comparisons = []
    for task in ('bbox', 'seg'):
        source = workspace_path('STAGED_FINETUNING_20260913', f'reports/{task}_final_comparison.json')
        values = json.loads(source.read_text('utf-8'))
        first = json.loads(workspace_path('KOREA_WASTE_RULES_20260913_v1', f'reports/{task}_transfer_comparison.json').read_text('utf-8'))
        models = [('original', first['original_checkpoint']), ('first_transfer', first['new_checkpoint']), ('final_selected', values['selected_checkpoint'])]
        for split, keys in [('test', ['original_legacy_test', 'first_transfer_legacy_test', 'new_test']),
                            ('val', ['original_validation', None, 'selected_validation'])]:
            group = []
            for (name, checkpoint), key in zip(models, keys, strict=True):
                if key is None:
                    continue
                checkpoint = workspace_path(checkpoint)
                dest = folder/task/split/name
                dest.mkdir(parents=True)
                record = dict(schema_version=1, created_at=now(), origin='historical_import', task=task,
                    model=dict(name=name, path=str(checkpoint), sha256=sha(checkpoint)),
                    dataset=dict(split=split, fingerprint=None, identity=f'{task}_legacy_{split}', count=None),
                    settings=None, summary=normalize_legacy(values[key]),
                    provenance=dict(report=str(source), report_sha256=sha(source), field=key),
                    comparison_key=digest(dict(task=task, split=split, historical_report=sha(source))),
                    limitations=['기존 저장 결과를 표준 형식으로 옮겼습니다. 새로 평가한 수치가 아닙니다.',
                                 '과거 평가 당시의 전체 실행 조건과 데이터 내용 해시는 없어 새 평가와 자동 비교하지 않습니다.',
                                 '검출·분할 테스트 구성은 다릅니다. 새 혼동 사례의 개선을 입증하지 않습니다.',
                                 '과거 기록에 없는 클래스별 정밀도·재현율은 미기록으로 표시합니다.'])
                save_record(dest, record)
                group.append(dest/'result.json'); written.append(str(dest/'result.json'))
            comparisons.append(str(compare_records(group, folder/'comparisons')))
    training = []
    for project in ('baseline_training', '01_initial_finetuning', '02_model_scaling', '03_staged_finetuning'):
        for source in sorted((workspace_path('04_results')/project/'runs').glob('*/results.csv')):
            value = training_summary(source.parent)
            if value:
                path = folder/'training'/project/source.parent.name/'summary.json'
                write_json(path, value); training.append(str(path))
    write_json(folder/'index.json', dict(evaluations=written, comparisons=comparisons, training_summaries=training))
    (folder/'README.md').write_text('# 기존 결과 통합\n\n'
        'bbox/seg: 과제별 평가 결과. comparisons: 같은 과제·분할의 비교. training: CSV 기준 학습 요약.\n\n'
        '이 폴더는 기존 측정값을 정리한 것입니다. 새 평가 결과는 ../../evaluations에 있습니다.\n', encoding='utf-8')
    return folder


def save_training_summary(run, source):
    value = training_summary(run)
    if value is None:
        return None
    folder = unique_dir(workspace_path('model_analysis_results')/'training'/source, Path(run).name)
    write_json(folder/'summary.json', value)
    return folder
