"""Training job description: a JSON file the GUI writes and the CLI reads."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from . import registry


@dataclass
class TrainJob:
    model_key: str                     # registry key, e.g. "yolo11-seg"
    variant: str = ""                  # checkpoint / HF id
    dataset_dir: str = ""              # exported dataset root
    dataset_format: str = "yolo"       # yolo | yolo-det | coco | masks
    classes: List[str] = field(default_factory=list)
    project_dir: str = "runs"          # where run folders are created
    run_name: str = ""
    device: str = "0"                  # "0" | "cpu" | "0,1"
    hyper: Dict = field(default_factory=dict)
    resume: str = ""                   # checkpoint to resume from
    created: float = field(default_factory=time.time)

    # ---- helpers ----------------------------------------------------------
    @property
    def spec(self):
        return registry.get(self.model_key)

    @property
    def run_dir(self) -> Path:
        """runs/<model_key>/<run_name>/ -- job.json and the results together.

        The model_key level is what lets several runs of the same model be
        compared side by side. Every trainer derives its output directory
        from here; none of them rebuilds the path itself, or job.json would
        drift into a different folder than the weights it describes.
        """
        return Path(self.project_dir) / self.model_key / self.run_name

    def fill_defaults(self) -> "TrainJob":
        spec = self.spec
        if spec is None:
            raise ValueError(f"알 수 없는 모델: {self.model_key}")
        if not self.variant:
            self.variant = spec.default_variant
        if not self.dataset_format:
            self.dataset_format = spec.dataset_format
        merged = dict(spec.defaults)
        merged.update(self.hyper or {})
        self.hyper = merged
        if not self.run_name:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            self.run_name = f"{self.model_key}_{stamp}"
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Optional[Path] = None) -> Path:
        path = Path(path) if path else (self.run_dir / "job.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                        encoding="utf-8")
        return path

    @staticmethod
    def load(path: Path) -> "TrainJob":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        valid = set(TrainJob.__dataclass_fields__)
        return TrainJob(**{k: v for k, v in d.items() if k in valid})

    def describe(self) -> str:
        s = self.spec
        name = s.display if s else self.model_key
        h = self.hyper
        return (f"{name} / {self.variant}\n"
                f"데이터셋: {self.dataset_dir} ({self.dataset_format})\n"
                f"클래스 {len(self.classes)}개 · epochs {h.get('epochs')} · "
                f"imgsz {h.get('imgsz')} · batch {h.get('batch')} · device {self.device}")
