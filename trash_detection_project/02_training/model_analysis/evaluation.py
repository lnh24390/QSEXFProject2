"""내부 모델과 외부 Ultralytics .pt 모델에 같은 평가 규약을 적용한다."""
from pathlib import Path
import os
import sys
import json
import time
from contextlib import contextmanager
import yaml

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'workspace_paths.py').exists())
sys.path.insert(0, str(ROOT))
from workspace_paths import workspace_path
from .schema import names_dict, sha, digest, now, unique_dir, write_json, summary, save_record, refresh_index

DEFAULTS = dict(imgsz=640, batch=16, workers=4, device=0, conf=.001, iou=.7, max_det=300,
                quantize=32, augment=False, rect=True, single_cls=False, agnostic_nms=False,
                classes=None, plots=False, save_json=False, save_txt=False, verbose=False,
                seed=0, deterministic=True, fraction=1.0, cache=False, compile=False,
                overlap_mask=True, mask_ratio=4, retina_masks=False, dnn=False)


def class_mapping(model_names, data_names, aliases=None):
    """완전한 일대일 대응만 인정한다. COCO 나 숫자 라벨의 의미를 추측하지 않는다."""
    model_names, data_names = names_dict(model_names), names_dict(data_names)
    aliases = aliases or {}
    translated = {i: aliases.get(str(i), aliases.get(name, name)) for i, name in model_names.items()}
    if len(set(translated.values())) != len(translated) or set(translated.values()) != set(data_names.values()):
        raise ValueError(f'모델 클래스가 평가 클래스와 다릅니다. model={model_names}, data={data_names}. '
                         '같은 의미의 모든 클래스를 1:1로 지정한 --class-map JSON이 필요합니다.')
    return {i: next(k for k, name in data_names.items() if name == target) for i, target in translated.items()}


def dataset_snapshot(data, split):
    """Ultralytics 의 경로 해석 방식을 따르고, 실제 이미지·라벨 바이트로 지문을 만든다."""
    from ultralytics.data.utils import check_det_dataset, img2label_paths
    from ultralytics.data.base import BaseDataset
    path = workspace_path(data)
    config = check_det_dataset(str(path), autodownload=False)
    if not config.get(split):
        raise ValueError(f'{split} 분할이 없습니다: {path}')
    reader = BaseDataset.__new__(BaseDataset)
    reader.prefix = ''; reader.fraction = 1.0
    files = reader.get_img_files(config[split])
    if len(files) != len(set(files)):
        raise ValueError('평가 목록에 중복 이미지 경로가 있습니다.')
    rows = []
    for image, label in zip(files, img2label_paths(files), strict=True):
        if not Path(label).is_file():
            raise ValueError(f'라벨이 없습니다. 배경도 빈 .txt 파일이 필요합니다: {label}')
        rows.append(dict(image=str(Path(image).resolve()), label=str(Path(label).resolve()),
                         image_sha256=sha(image), label_sha256=sha(label)))
    if not rows:
        raise ValueError('평가 이미지가 없습니다.')
    fingerprint = digest({'names': config['names'],
                          'samples': sorted((r['image_sha256'], r['label_sha256']) for r in rows)})
    return dict(yaml=str(path), split=split, names=names_dict(config['names']),
                count=len(rows), fingerprint=fingerprint), rows


def remapped_validator(task, mapping, names):
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.models.yolo.segment import SegmentationValidator
    import torch
    base = SegmentationValidator if task == 'segment' else DetectionValidator

    class MappedValidator(base):
        def build_dataset(self, img_path, mode='val', batch=None):
            from copy import copy
            from .dataset import EvaluationDataset
            return EvaluationDataset(img_path=img_path, imgsz=self.args.imgsz, batch_size=batch,
                augment=False, hyp=copy(self.args), rect=self.args.rect, cache=False,
                single_cls=False, stride=self.stride, pad=.5, prefix='evaluation: ',
                task=self.args.task, classes=None, data=self.data, fraction=1.0)

        def init_metrics(self, model):
            super().init_metrics(model)
            if len(self.dataloader.dataset) != self.data['_analysis_count']:
                raise ValueError('평가 중 이미지가 누락되었습니다. 손상된 이미지·라벨을 확인하세요.')
            self.names = names
            self.metrics.names = names
            self.confusion_matrix.names = names

        def postprocess(self, preds):
            output = super().postprocess(preds)
            for pred in output:
                lookup = torch.tensor([mapping[i] for i in range(len(mapping))], device=pred['cls'].device)
                pred['cls'] = lookup[pred['cls'].long()].to(pred['cls'].dtype)
            return output
    return MappedValidator


def validate_task_labels(manifest, task):
    import math
    for row in manifest:
        for line in Path(row['label']).read_text('utf-8-sig').splitlines():
            if not line.strip():
                continue
            values = list(map(float, line.split()))
            valid_shape = len(values) == 5 if task == 'detect' else len(values) >= 7 and len(values) % 2 == 1
            if not valid_shape or not all(math.isfinite(v) for v in values):
                raise ValueError(f'{task} 과제에 맞지 않는 라벨: {row["label"]}')


@contextmanager
def evaluation_lock():
    """평가용 공용 잠금. 두 학습 폴더에서 부르는 경우까지 함께 막는다."""
    parent = workspace_path('model_analysis_results')
    parent.mkdir(parents=True, exist_ok=True)
    lock = parent/'evaluation.lock'
    try:
        stream = lock.open('x', encoding='utf-8')
    except FileExistsError as exc:
        raise RuntimeError(f'다른 평가가 실행 중이거나 중단된 잠금이 남았습니다: {lock}') from exc
    try:
        with stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        lock.unlink(missing_ok=True)


def ensure_gpu_idle():
    """단독 실행 명령은 학습이 끝날 때까지 기다린다. 파이프라인 안의 평가는 자기 차례를 가진다."""
    import psutil
    targets = {'train.py', 'train_best.py', 'train_sequential.py', 'train_pipeline.py', 'pipeline.py',
               'benchmark_sizes.py', 'benchmark_mosaic.py', 'bench_run.py'}
    current = psutil.Process(os.getpid())
    own = {current.pid, *[p.pid for p in current.parents()]}
    busy = []
    for process in psutil.process_iter(['pid', 'cmdline']):
        if process.pid in own:
            continue
        try:
            if any(Path(arg).name.lower() in targets for arg in (process.info['cmdline'] or [])):
                busy.append(process.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if busy:
        raise RuntimeError(f'학습 또는 벤치마크가 진행 중입니다. 완료 후 평가하세요: PID {busy}')


def evaluate_model(weights, data, *, split='test', name=None, source='external', task=None,
                   class_map=None, model=None, **options):
    """평소의 지표 객체를 돌려주고, 조건을 모두 적은 평가 기록을 함께 저장한다."""
    import torch
    import ultralytics
    from ultralytics import YOLO
    checkpoint = workspace_path(weights)
    if not checkpoint.is_file() or checkpoint.suffix.lower() != '.pt':
        raise ValueError(f'존재하는 Ultralytics .pt 파일을 지정하세요: {checkpoint}')
    if split not in ('val', 'test'):
        raise ValueError('분석은 val 또는 test 분할만 지원합니다.')
    net = model if model is not None else YOLO(str(checkpoint))
    if net.task not in ('detect', 'segment'):
        raise ValueError(f'지원하지 않는 모델 과제: {net.task}')
    family = 'seg' if net.task == 'segment' else 'bbox'
    if task and task not in (family, net.task):
        raise ValueError(f'모델 과제 {net.task}와 요청 과제 {task}가 다릅니다.')
    original_hash = sha(checkpoint)
    with evaluation_lock():
        dataset, manifest = dataset_snapshot(data, split)
        validate_task_labels(manifest, net.task)
        model_names = names_dict(net.names)
        mapping = class_mapping(model_names, dataset['names'], class_map)
        settings = dict(DEFAULTS)
        unknown = set(options) - set(DEFAULTS)
        if unknown:
            raise ValueError(f'공통 평가에 정의되지 않은 옵션: {sorted(unknown)}')
        settings.update(options)
        if str(settings['device']).lower() != 'cpu' and not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU를 사용할 수 없습니다.')
        folder = unique_dir(workspace_path('model_analysis_results')/'evaluations'/family/source, name or checkpoint.stem)
        write_json(folder/'dataset_manifest.json', manifest)
        # 경로를 모두 푼 로컬 YAML 을 쓰면 작업 디렉토리에 따라 해석이 달라지지 않고 자동 다운로드도 막힌다.
        from ultralytics.data.utils import check_det_dataset
        resolved = check_det_dataset(str(workspace_path(data)), autodownload=False)
        resolved.pop('download', None)
        resolved = json.loads(json.dumps(resolved, default=str))
        resolved.update(_analysis_fingerprint=dataset['fingerprint'], _analysis_cache_dir=str(folder),
                        _analysis_count=dataset['count'])
        resolved_yaml = folder/'data.yaml'
        resolved_yaml.write_text(yaml.safe_dump(resolved, allow_unicode=True), encoding='utf-8')
        started = time.perf_counter()
        try:
            validator_type = remapped_validator(net.task, mapping, dataset['names'])
            captured = []
            def validator_factory(**kwargs):
                validator = validator_type(**kwargs)
                captured.append(validator)
                return validator
            result = net.val(data=str(resolved_yaml), split=split,
                validator=validator_factory,
                project=str(folder), name='artifacts', exist_ok=False, **settings)
            validator = captured[0]
            elapsed = time.perf_counter()-started
            if sha(checkpoint) != original_hash:
                raise RuntimeError('평가 중 가중치 파일이 변경되었습니다.')
            result.names = dataset['names']
            actual = {key: getattr(validator.args, key) for key in settings}
            identity = {key: actual[key] for key in actual if key not in ('plots', 'save_json', 'save_txt', 'verbose', 'workers')}
            identity.update(ultralytics=ultralytics.__version__, torch=torch.__version__)
            record = dict(schema_version=1, created_at=now(), origin='measured', task=family,
                model=dict(name=name or checkpoint.stem, path=str(checkpoint), sha256=original_hash,
                           source=source, native_names=model_names, class_mapping=mapping),
                dataset=dataset, settings=actual,
                environment=dict(ultralytics=ultralytics.__version__, torch=torch.__version__,
                    device=str(validator.device), gpu=torch.cuda.get_device_name(validator.device) if validator.device.type=='cuda' else None),
                wall_seconds=elapsed, summary=summary(result),
                comparison_key=digest(dict(task=family, dataset=dataset['fingerprint'], split=split, settings=identity)),
                limitations=['기존 평가 자료를 다시 본 결과로, 한국 현장·새로운 혼동 사례의 독립 평가를 대신하지 않습니다.',
                             '속도는 이번 검증 실행의 이미지당 시간이며 전용 추론 벤치마크가 아닙니다.'])
            if source == 'external':
                record['limitations'].append('외부 모델의 학습 데이터와 평가 데이터 중복 여부는 확인되지 않았습니다.')
            save_record(folder, record)
            refresh_index(workspace_path('model_analysis_results'))
            result.analysis_record_path = str(folder/'result.json')
            return result
        except Exception as exc:
            write_json(folder/'failure.json', dict(at=now(), error=str(exc), model=str(checkpoint), dataset=dataset, settings=settings))
            raise
