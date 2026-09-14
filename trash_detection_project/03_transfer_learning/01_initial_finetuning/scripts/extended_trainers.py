"""학습 상태를 복원하면서 에포크 상한만 명시적으로 바꿀 수 있게 한다."""
from pathlib import Path
import sys
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p/'workspace_paths.py').is_file())))
from workspace_trainers import RelocatedDetectionTrainer as DetectionTrainer
from workspace_trainers import RelocatedSegmentationTrainer as SegmentationTrainer

class ExtendEpochs:
    def check_resume(self, overrides):
        super().check_resume(overrides)
        if self.resume and 'epochs' in overrides:
            self.args.epochs = int(overrides['epochs'])

class ExtendedDetectionTrainer(ExtendEpochs, DetectionTrainer):
    pass

class ExtendedSegmentationTrainer(ExtendEpochs, SegmentationTrainer):
    pass
