"""Optional heavy dependencies: what is missing, and how to install it.

None of these ship inside the frozen build (see build.spec). torch alone is
~4GB on disk with its CUDA runtime, which would dwarf the 200MB application
and force every user to download it whether or not they train anything. So
the executable stays small and installs nothing; when a feature needs one of
these, the app says exactly what to run.

This module is the single place those commands are written down. Backends and
trainers raise their own errors at the point of use, but they all quote the
commands from here so the wording cannot drift.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class OptionalDep:
    module: str          # import name, not the pip name
    display: str
    install: str         # the exact command to run
    size_mb: int         # rough installed size, so the user can judge
    enables: str

    @property
    def installed(self) -> bool:
        try:
            return importlib.util.find_spec(self.module) is not None
        except (ImportError, ValueError):
            # A half-written install leaves a package directory that
            # find_spec accepts but import does not; treat that as absent.
            return False


#: Ordered cheapest-first: the first entry alone makes SAM usable.
DEPS: List[OptionalDep] = [
    OptionalDep(
        "onnxruntime", "ONNX Runtime",
        "uv pip install onnxruntime", 60,
        "torch 없이 SAM 2.1 세그먼트 (가장 가벼운 선택)"),
    OptionalDep(
        "psutil", "psutil",
        "uv pip install psutil", 5,
        "정확한 RAM 측정"),
    OptionalDep(
        "torch", "PyTorch (CUDA 12.6)",
        "uv pip install torch torchvision "
        "--index-url https://download.pytorch.org/whl/cu126", 4000,
        "GPU 가속, 모든 학습, SAM 1/2/3"),
    OptionalDep(
        "ultralytics", "Ultralytics",
        "uv pip install ultralytics", 120,
        "YOLO 계열 학습·RT-DETR·MobileSAM (torch 필요)"),
    OptionalDep(
        "transformers", "HuggingFace Transformers",
        "uv pip install \"transformers>=4.57\" huggingface_hub", 400,
        "SAM 3/3.1, SegFormer, ConvNeXt-UPerNet (torch 필요)"),
    OptionalDep(
        "sam2", "SAM 2 (공식)",
        "uv pip install git+https://github.com/facebookresearch/sam2.git", 30,
        "SAM 2 / 2.1 정식 백엔드 (torch 필요)"),
    OptionalDep(
        "segment_anything", "Segment Anything (공식)",
        "uv pip install git+https://github.com/facebookresearch/"
        "segment-anything.git", 20,
        "SAM 1 백엔드 (torch 필요)"),
    OptionalDep(
        "pycocotools", "pycocotools",
        "uv pip install pycocotools", 10,
        "Mask R-CNN 평가, 압축 RLE 읽기"),
]

BY_MODULE: Dict[str, OptionalDep] = {d.module: d for d in DEPS}


def install_command(module: str) -> str:
    """The command that installs `module`, or a plain pip fallback."""
    dep = BY_MODULE.get(module)
    return dep.install if dep else f"uv pip install {module}"


def missing() -> List[OptionalDep]:
    return [d for d in DEPS if not d.installed]


def report() -> str:
    """Human-readable status of every optional dependency."""
    lines = []
    for d in DEPS:
        mark = "설치됨" if d.installed else "없음  "
        lines.append(f"[{mark}] {d.display}  (~{d.size_mb}MB)\n"
                     f"          {d.enables}")
        if not d.installed:
            lines.append(f"          → {d.install}")
    return "\n".join(lines)


def need(module: str, why: str = "") -> str:
    """The message shown when a feature is blocked by a missing package."""
    dep = BY_MODULE.get(module)
    name = dep.display if dep else module
    size = f" (~{dep.size_mb}MB)" if dep else ""
    head = f"'{name}'{size} 이(가) 필요합니다."
    if why:
        head += f"\n{why}"
    return (f"{head}\n\n    {install_command(module)}\n\n"
            "설치는 앱 바깥에서 한 번만 하면 됩니다. 설치 후 다시 시도하세요.")
