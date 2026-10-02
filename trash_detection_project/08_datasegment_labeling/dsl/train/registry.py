"""Catalog of trainable models.

Adding a model = adding an entry here (plus a trainer function if the
framework is new). The training panel builds itself from this table.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TrainerSpec:
    key: str
    display: str
    framework: str            # ultralytics | torchvision | hf
    task: str                 # instance | semantic | detect
    dataset_format: str       # yolo | coco | masks
    variants: List[str]       # checkpoint names / HF ids, small -> large
    default_variant: str = ""
    requires: List[str] = field(default_factory=list)
    defaults: Dict = field(default_factory=dict)
    auto_download: bool = True     # framework fetches the checkpoint itself
    notes: str = ""

    def __post_init__(self):
        if not self.default_variant and self.variants:
            self.default_variant = self.variants[0]


_YOLO_DEFAULTS = {"epochs": 100, "imgsz": 640, "batch": 8, "lr0": 0.01,
                  "patience": 30, "workers": 4, "optimizer": "auto",
                  "amp": True, "cache": False}

_SPECS: List[TrainerSpec] = [
    # ---- Ultralytics instance segmentation --------------------------------
    TrainerSpec(
        "yolo26-seg", "YOLO26 Segmentation", "ultralytics", "instance", "yolo",
        ["yolo26n-seg.pt", "yolo26s-seg.pt", "yolo26m-seg.pt",
         "yolo26l-seg.pt", "yolo26x-seg.pt"],
        requires=["ultralytics"], defaults=dict(_YOLO_DEFAULTS),
        notes="최신 세대. ultralytics 가 가중치를 자동으로 내려받습니다. "
              "설치된 ultralytics 가 YOLO26을 모르면 오류 메시지에 업그레이드 안내가 나옵니다."),
    TrainerSpec(
        # Ultralytics publishes YOLO12 as *detection only* -- there is no
        # yolo12*-seg.pt in the assets release (checked v8.4.0, 2026-09-07),
        # and the file prefix is `yolo12`, not `yolov12`.
        "yolo12", "YOLO12 Detection", "ultralytics", "detect", "yolo-det",
        ["yolo12n.pt", "yolo12s.pt", "yolo12m.pt", "yolo12l.pt", "yolo12x.pt"],
        requires=["ultralytics"], defaults=dict(_YOLO_DEFAULTS),
        notes="어텐션 중심 구조. 세그먼트 체크포인트는 공개되지 않아 검출 전용입니다. "
              "라벨은 세그먼트 폴리곤의 바운딩 박스로 만듭니다. "
              "세그먼트가 필요하면 YOLO11/YOLO26 을 쓰세요."),
    TrainerSpec(
        "yolo11-seg", "YOLO11 Segmentation (안정)", "ultralytics", "instance", "yolo",
        ["yolo11n-seg.pt", "yolo11s-seg.pt", "yolo11m-seg.pt",
         "yolo11l-seg.pt", "yolo11x-seg.pt"],
        default_variant="yolo11s-seg.pt",
        requires=["ultralytics"], defaults=dict(_YOLO_DEFAULTS),
        notes="가장 검증된 기본값. 처음이라면 여기서 시작하세요."),
    TrainerSpec(
        "yolov9-seg", "YOLOv9 Segmentation", "ultralytics", "instance", "yolo",
        ["yolov9c-seg.pt", "yolov9e-seg.pt"],
        requires=["ultralytics"], defaults=dict(_YOLO_DEFAULTS)),
    TrainerSpec(
        "yolov8-seg", "YOLOv8 Segmentation", "ultralytics", "instance", "yolo",
        ["yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt",
         "yolov8l-seg.pt", "yolov8x-seg.pt"],
        requires=["ultralytics"], defaults=dict(_YOLO_DEFAULTS)),
    TrainerSpec(
        "rtdetr", "RT-DETR (Baidu, 박스 검출)", "ultralytics", "detect", "yolo-det",
        ["rtdetr-l.pt", "rtdetr-x.pt"],
        requires=["ultralytics"],
        defaults={**_YOLO_DEFAULTS, "batch": 4, "lr0": 0.0001},
        notes="RT-DETR은 검출 전용입니다. 내보내기에서 세그먼트 폴리곤의 "
              "바운딩 박스를 뽑아 검출 라벨로 만듭니다."),

    # ---- torchvision -------------------------------------------------------
    TrainerSpec(
        "maskrcnn", "Mask R-CNN (torchvision)", "torchvision", "instance", "coco",
        ["maskrcnn_resnet50_fpn_v2", "maskrcnn_resnet50_fpn"],
        requires=["torchvision", "pycocotools"],
        defaults={"epochs": 30, "imgsz": 800, "batch": 2, "lr0": 0.005,
                  "momentum": 0.9, "weight_decay": 0.0005, "workers": 2,
                  "amp": True, "trainable_layers": 3},
        notes="인스턴스 마스크 품질이 높지만 느립니다. 배치 2~4 권장."),

    # ---- HuggingFace semantic segmentation --------------------------------
    TrainerSpec(
        "segformer", "SegFormer (시맨틱)", "hf", "semantic", "masks",
        ["nvidia/mit-b0", "nvidia/mit-b1", "nvidia/mit-b2",
         "nvidia/mit-b3", "nvidia/mit-b4", "nvidia/mit-b5"],
        default_variant="nvidia/mit-b2",
        requires=["transformers"],
        defaults={"epochs": 50, "imgsz": 512, "batch": 4, "lr0": 6e-5,
                  "weight_decay": 0.01, "workers": 2, "amp": True},
        notes="픽셀 단위 클래스 분할. 인스턴스 구분은 하지 않습니다."),
    TrainerSpec(
        "convnext-upernet", "ConvNeXt + UPerNet (시맨틱)", "hf", "semantic", "masks",
        ["openmmlab/upernet-convnext-tiny", "openmmlab/upernet-convnext-small",
         "openmmlab/upernet-convnext-base", "openmmlab/upernet-convnext-large"],
        requires=["transformers"],
        defaults={"epochs": 50, "imgsz": 512, "batch": 2, "lr0": 1e-4,
                  "weight_decay": 0.05, "workers": 2, "amp": True},
        notes="ConvNeXt 백본 + UPerNet 헤드. SegFormer보다 무겁습니다."),
]

REGISTRY: Dict[str, TrainerSpec] = {s.key: s for s in _SPECS}

TASK_LABELS = {"instance": "인스턴스 세그멘테이션",
               "semantic": "시맨틱 세그멘테이션",
               "detect": "객체 검출"}


def get(key: str) -> Optional[TrainerSpec]:
    return REGISTRY.get(key)


def by_task(task: str) -> List[TrainerSpec]:
    return [s for s in _SPECS if s.task == task]


def all_specs() -> List[TrainerSpec]:
    return list(_SPECS)


def missing_packages(spec: TrainerSpec) -> List[str]:
    import importlib.util
    alias = {"pycocotools": "pycocotools", "transformers": "transformers",
             "ultralytics": "ultralytics", "torchvision": "torchvision"}
    out = []
    for req in spec.requires:
        base = req.split(">=")[0].split("==")[0].strip()
        if importlib.util.find_spec(alias.get(base, base.replace("-", "_"))) is None:
            out.append(req)
    return out


def recommend_batch(spec: TrainerSpec, vram_gb: float, imgsz: int = 640) -> int:
    """Rough batch size that fits the detected VRAM."""
    if vram_gb <= 0:
        return 1
    base = {"ultralytics": 3.0, "torchvision": 6.0, "hf": 5.0}[spec.framework]
    scale = (imgsz / 640.0) ** 2
    est = max(1, int((vram_gb * 0.8) / (base * scale)))
    return min(est, 32)
