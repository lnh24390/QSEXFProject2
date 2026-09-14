"""라벨이 비어 있음을 확인한 이미지에서 오탐을 측정한다. 이미지를 복사하지 않는다."""
from pathlib import Path
import json
from collections import Counter
from .evaluation import evaluation_lock, ensure_gpu_idle
from .schema import now, sha, write_json


def evaluate_background(weights, evaluation_record, output, name):
    from ultralytics import YOLO
    record_path = Path(evaluation_record)
    record = json.loads(record_path.read_text('utf-8'))
    manifest = json.loads(record_path.with_name('dataset_manifest.json').read_text('utf-8'))
    images = [row for row in manifest if not Path(row['label']).read_text('utf-8-sig').strip()]
    if not images:
        raise ValueError('빈 정답 라벨을 가진 배경 이미지가 없습니다.')
    for row in images:
        if sha(row['image']) != row['image_sha256'] or sha(row['label']) != row['label_sha256']:
            raise ValueError('이전 평가 이후 데이터가 변경되었습니다.')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    dest = output/(name+'_background.json')
    if dest.exists():
        raise FileExistsError(dest)
    listing = output/(name+'_background_images.txt')
    listing.write_text('\n'.join(row['image'] for row in images)+'\n', encoding='utf-8')
    before = sha(weights)
    ensure_gpu_idle()
    with evaluation_lock():
        model = YOLO(str(weights))
        thresholds = (.25, .5)
        counts = {str(t): dict(images_with_fp=0, detections=0, per_class=Counter()) for t in thresholds}
        details = []
        for result in model.predict(source=str(listing), imgsz=640, batch=16, device=0, quantize=32,
                conf=.25, iou=.7, max_det=300, augment=False, rect=True, stream=True,
                save=False, verbose=False):
            boxes = result.boxes
            predictions = [] if boxes is None else [dict(class_name=model.names[int(cls)], confidence=float(conf))
                for cls, conf in zip(boxes.cls.cpu().tolist(), boxes.conf.cpu().tolist(), strict=True)]
            for threshold in thresholds:
                selected = [row for row in predictions if row['confidence'] >= threshold]
                current = counts[str(threshold)]
                current['images_with_fp'] += bool(selected)
                current['detections'] += len(selected)
                current['per_class'].update(row['class_name'] for row in selected)
            details.append(dict(image=result.path, predictions=predictions))
        if len(details) != len(images) or before != sha(weights):
            raise RuntimeError('이미지 수 또는 모델 해시 검증에 실패했습니다.')
        for value in counts.values():
            value['image_fp_rate'] = value['images_with_fp']/len(images)
            value['per_class'] = dict(value['per_class'])
        result = dict(schema_version=1, at=now(), model=name, checkpoint=str(weights), sha256=before,
            task=record['task'], source_evaluation=str(record_path), image_count=len(images),
            image_sha256=sorted(row['image_sha256'] for row in images), thresholds=counts,
            settings=dict(imgsz=640, batch=16, quantize=32, iou=.7, max_det=300, device=0),
            images=details, note='현재 공통 test의 빈 라벨 배경입니다. 새 교실·컴퓨터실 현장 평가를 대신하지 않습니다.')
        write_json(dest, result)
        del model
        import torch
        torch.cuda.empty_cache()
        return result
