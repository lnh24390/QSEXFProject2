"""클래스 이름을 붙인, 버전이 있는 지표. 측정하지 못한 값은 null 로 둔다."""
from pathlib import Path
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from uuid import uuid4


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def unique_dir(parent, label):
    import re
    label = re.sub(r'[^\w.-]+', '_', label).strip(' ._')[:80] or 'model'
    path = Path(parent) / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+label+'_'+uuid4().hex[:8])
    path.mkdir(parents=True, exist_ok=False)
    return path


def names_dict(names):
    value = dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}
    if sorted(value) != list(range(len(value))) or len(set(value.values())) != len(value):
        raise ValueError('클래스 번호는 0부터 연속이어야 하고 클래스 이름이 중복되면 안 됩니다.')
    return value


def number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def summary(result):
    """기존 키를 그대로 두고, 이름 기준의 클래스별 지표를 모두 더한다."""
    names = names_dict(result.names)
    speed = {k: number(v) for k, v in result.speed.items()}
    data = {'metrics': {k: number(v) for k, v in result.results_dict.items()},
            'names': {str(k): v for k, v in names.items()},
            'speed_ms': speed, 'speed_ms_per_image': speed}
    for kind in ('box', 'seg'):
        metric = getattr(result, kind, None)
        if metric is None:
            continue
        per = {name: {'precision': None, 'recall': None, 'map50': None, 'map': None} for name in names.values()}
        # metric.maps 는 없는 클래스를 전체 평균으로 채운다. 그 값을 측정된 AP 로 보고하지 않는다.
        maps = [None] * len(names)
        for i, cls in enumerate(metric.ap_class_index):
            cls = int(cls)
            per[names[cls]] = dict(precision=number(metric.p[i]), recall=number(metric.r[i]),
                                   map50=number(metric.ap50[i]), map=number(metric.ap[i]))
            maps[cls] = number(metric.ap[i])
        data[kind] = dict(map=number(metric.map), map50=number(metric.map50),
                          precision=number(metric.mp), recall=number(metric.mr),
                          per_class_map=maps, per_class=per)
    return data


def normalize_legacy(value):
    names = names_dict(value['names'])
    data = {'names': {str(k): v for k, v in names.items()}, 'metrics': value.get('metrics', {}),
            'speed_ms': value.get('speed_ms', value.get('speed_ms_per_image', {}))}
    for kind, suffix in [('box', 'B'), ('seg', 'M')]:
        if kind not in value:
            continue
        metric = value[kind]
        data[kind] = dict(map=metric['map'], map50=metric.get('map50'),
            precision=data['metrics'].get(f'metrics/precision({suffix})'),
            recall=data['metrics'].get(f'metrics/recall({suffix})'),
            per_class={name: dict(map=metric['per_class_map'][i], map50=None, precision=None, recall=None)
                       for i, name in names.items()})
    return data


def save_record(folder, record):
    write_json(folder/'result.json', record)
    with (folder/'metrics.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model', 'task', 'split', 'metric', 'class', 'precision', 'recall', 'map50', 'map50_95'])
        for kind in ('box', 'seg'):
            metric = record['summary'].get(kind)
            if metric is None:
                continue
            for name, values in [('all', metric), *metric['per_class'].items()]:
                writer.writerow([record['model']['name'], record['task'], record['dataset']['split'], kind, name,
                                 *[values.get(k) for k in ('precision', 'recall', 'map50', 'map')]])
    lines = [f"# 모델 평가: {record['model']['name']}", '',
             f"과제: {record['task']} / 분할: {record['dataset']['split']} / 출처: {record['origin']}", '',
             '수치는 0~1 단위입니다. 빈 값은 측정 기록이 없다는 뜻입니다.', '',
             '| 지표 | mAP50–95 | mAP50 | 정밀도 | 재현율 |', '|---|---:|---:|---:|---:|']
    for kind in ('box', 'seg'):
        metric = record['summary'].get(kind)
        if metric:
            cells = ['미기록' if metric.get(k) is None else f'{metric[k]:.4f}' for k in ('map', 'map50', 'precision', 'recall')]
            lines.append('| '+('박스' if kind == 'box' else '마스크')+' | '+' | '.join(cells)+' |')
    lines += ['', *['- '+v for v in record.get('limitations', [])], '',
              '전체 클래스별 수치와 평가 조건은 result.json 및 metrics.csv에 있습니다.']
    (folder/'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def compare_records(paths, output_root):
    records = [json.loads(Path(path).read_text('utf-8')) for path in paths]
    if len(records) < 2:
        raise ValueError('비교에는 평가 결과가 2개 이상 필요합니다.')
    keys = {record.get('comparison_key') for record in records}
    if len(keys) != 1 or None in keys:
        raise ValueError('과제·이미지/라벨·분할·평가 조건이 다릅니다. 같은 조건으로 다시 평가하세요.')
    folder = unique_dir(output_root, records[0]['task']+'_comparison')
    lines = ['# 동일 조건 모델 비교', '', 'mAP50–95는 % 단위, 차이는 첫 모델 대비 %p입니다.', '',
             '| 모델 | 박스 | 박스 차이 | 마스크 | 마스크 차이 |', '|---|---:|---:|---:|---:|']
    rows = []
    for record in records:
        cells = [record['model']['name']]
        for kind in ('box', 'seg'):
            current = record['summary'].get(kind)
            base = records[0]['summary'].get(kind)
            cells += [f"{current['map']*100:.2f}", f"{(current['map']-base['map'])*100:+.2f}"] if current and base else ['—', '—']
            if current:
                for cls, values in [('all', current), *current['per_class'].items()]:
                    ref = base if cls == 'all' else base['per_class'].get(cls, {})
                    delta = None if values.get('map') is None or ref.get('map') is None else (values['map']-ref['map'])*100
                    rows.append([record['model']['name'], kind, cls, values.get('map'), delta])
        lines.append('| '+' | '.join(cells)+' |')
    limitations = sorted({note for record in records for note in record.get('limitations', [])})
    for kind in ('box', 'seg'):
        if kind not in records[0]['summary']:
            continue
        lines += ['', ('박스' if kind == 'box' else '마스크')+' 클래스별 mAP50–95 (%)', '',
                  '| 클래스 | '+' | '.join(r['model']['name'] for r in records)+' |',
                  '|---|'+'---:|'*len(records)]
        for cls in records[0]['summary'][kind]['per_class']:
            values = [r['summary'][kind]['per_class'].get(cls, {}).get('map') for r in records]
            lines.append('| '+cls+' | '+' | '.join('미기록' if v is None else f'{v*100:.2f}' for v in values)+' |')
    lines += ['', *['- '+note for note in limitations], '',
              '검출과 분할은 별도로 비교합니다. 속도는 전용 벤치마크가 아니므로 순위를 매기지 않습니다.']
    (folder/'comparison.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write_json(folder/'comparison.json', {'schema_version': 1, 'records': [str(Path(p).resolve()) for p in paths],
                                         'comparison_key': records[0]['comparison_key'], 'limitations': limitations})
    with (folder/'comparison.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream); writer.writerow(['model', 'metric', 'class', 'map50_95', 'delta_pp']); writer.writerows(rows)
    return folder


def refresh_index(root):
    """다시 만들 수 있는 것은 이 탐색용 목록뿐이다. 측정 기록 자체는 바꾸지 않는다."""
    root = Path(root)
    lines = ['# 모델 분석 결과 목록', '', '새 측정과 과거 기록을 구분합니다. 각 과제의 비교표를 확인하세요.', '',
             '## 새 평가 비교', '']
    for path in sorted((root/'comparisons').glob('*/comparison.md'), reverse=True):
        lines.append(f'- [{path.parent.name}]({path.relative_to(root).as_posix()})')
    lines += ['', '## 새 측정', '', '| 모델 | 과제 | 분할 | 이미지 수 | 결과 |', '|---|---|---|---:|---|']
    for path in sorted((root/'evaluations').glob('*/*/*/result.json')):
        record = json.loads(path.read_text('utf-8'))
        if record['model'].get('source') == 'verification' or record.get('verification_only'):
            continue
        target = path.with_name('summary.md').relative_to(root).as_posix()
        lines.append(f"| {record['model']['name']} | {record['task']} | {record['dataset']['split']} | {record['dataset']['count']} | [보기]({target}) |")
    lines += ['', '## 과거 기록 정리', '']
    for path in sorted((root/'historical').glob('*/README.md'), reverse=True):
        lines.append(f'- [{path.parent.name}]({path.relative_to(root).as_posix()})')
    lines += ['', '## 종합 분석', '']
    for path in sorted((root/'reviews').glob('*/summary.md'), reverse=True):
        title = path.read_text('utf-8').splitlines()[0].lstrip('# ')
        lines.append(f'- [{title}]({path.relative_to(root).as_posix()})')
    (root/'index.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
