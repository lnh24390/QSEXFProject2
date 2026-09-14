"""체크포인트의 폴더 정보를 메모리에서 다시 매핑한다. 가중치 파일은 바꾸지 않는다."""
from pathlib import Path
from workspace_paths import workspace_path
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.segment import SegmentationTrainer


class RelocatedResumePaths:
    def check_resume(self, overrides):
        for key in ('data', 'model', 'resume', 'project', 'save_dir', 'pretrained'):
            value = getattr(self.args, key, None)
            if isinstance(value, (str, Path)) and value:
                setattr(self.args, key, str(workspace_path(value)))
        super().check_resume(overrides)
        for key in ('data', 'model', 'resume', 'project', 'save_dir', 'pretrained'):
            value = getattr(self.args, key, None)
            if isinstance(value, (str, Path)) and value:
                setattr(self.args, key, str(workspace_path(value)))


class RelocatedDetectionTrainer(RelocatedResumePaths, DetectionTrainer):
    pass


class RelocatedSegmentationTrainer(RelocatedResumePaths, SegmentationTrainer):
    pass


def trainer_for(task):
    return RelocatedSegmentationTrainer if task in ('seg', 'segment') else RelocatedDetectionTrainer
