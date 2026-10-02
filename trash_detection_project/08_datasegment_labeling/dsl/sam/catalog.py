"""Catalog of downloadable SAM checkpoints (SAM 1 / 2 / 2.1 / 3 / 3.1).

Nothing is bundled with the app: every entry is fetched on demand into
`weights/` and matched against the machine it will run on (see
`dsl.sam.downloader.probe_hardware` / `recommend`).

Quantisation policy
-------------------
There is no separate file per precision -- none is published upstream, and
none is needed. Precision is applied to the checkpoint as downloaded, at load
time, on the user's own machine (`SamBackend.apply_precision`):
  fp32 -> as downloaded
  fp16 -> `model.half()` on CUDA, before `.to(device)`
  bf16 -> `model.to(torch.bfloat16)` on CUDA (SAM 3 only; SAM 1/2 return masks
          through numpy, which has no bfloat16)
  int8 -> `torch.ao.quantization.quantize_dynamic` on CPU (Linear layers)
Ultralytics-packaged models own their forward pass, so fp16 goes through
`predict(half=True)` instead of a cast.

Measured on an RTX 4060 (8GB), 1280x720 image, one point prompt:
  sam1_vit_b   fp32 358MB weights / 2770MB peak  ->  fp16 179MB / 1395MB, 2.3x faster
  sam1_vit_h   fp32 2446MB / 5731MB              ->  fp16 1223MB / 2845MB
  sam2.1_large fp32 856MB / 1445MB               ->  fp16 428MB / 1001MB
  sam1_vit_b   cpu fp32 358MB params             ->  int8 18MB params
Mask scores were unchanged for SAM 1/2/2.1; MobileSAM fp16 is the one
exception (0.962 -> 0.693), so it is left on fp32/int8 by default.

`ModelSpec.quant_modes` lists what a given checkpoint supports, and
`recommend()` picks a precision that fits the detected VRAM/RAM.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

FBAI = "https://dl.fbaipublicfiles.com/segment_anything"
FBAI2 = "https://dl.fbaipublicfiles.com/segment_anything_2"
ULTRA = "https://github.com/ultralytics/assets/releases/download/v8.3.0"

# What a finished download looks like on disk.
_WEIGHT_PATTERNS = ("*.safetensors", "*.bin", "*.pt", "*.pth", "*.onnx")
_MIN_WEIGHT_BYTES = 1024


def _is_real_file(p: Path) -> bool:
    """A present, non-placeholder file (guards against 0-byte / stub files)."""
    return p.is_file() and p.stat().st_size > _MIN_WEIGHT_BYTES


@dataclass
class ModelSpec:
    key: str                       # unique id used everywhere
    family: str                    # sam1 | sam2 | sam2.1 | sam3 | sam3.1
    display: str                   # UI label
    backend: str                   # which SamBackend class loads it
    filename: str                  # local file name inside weights/
    url: str = ""                  # direct download
    repo_id: str = ""              # HuggingFace repo (used when url is empty)
    hf_file: str = ""              # file inside the HF repo ("" = snapshot)
    hf_patterns: List[str] = field(default_factory=list)  # snapshot filter
    size_mb: int = 0
    params_m: int = 0
    min_vram_gb: float = 0.0       # comfortable fp16 VRAM
    cpu_ok: bool = True            # usable at tolerable speed without a GPU
    config: str = ""               # sam2 hydra config name
    requires: List[str] = field(default_factory=list)   # pip packages
    quant_modes: List[str] = field(default_factory=lambda: ["fp32", "fp16"])
    text_prompt: bool = False      # supports concept/text prompting (SAM 3)
    experimental: bool = False
    notes: str = ""

    def local_path(self, weights_dir: Path) -> Path:
        """weights/<프레임워크>/<계열>/<파일>.

        `onnx` 는 계열이 아니라 배포 형식이지만, 내용물이 전부 SAM 변환본이라
        weights/sam/ 아래에 둔다. YOLO 체크포인트는 weights/yolo/pretrained/
        에 모인다 (trainer_ultralytics._resolve_weights 참고).
        """
        return Path(weights_dir) / "sam" / self.family / self.filename

    def is_installed(self, weights_dir: Path) -> bool:
        p = self.local_path(weights_dir)
        # HF snapshot folder: a directory's own st_size says nothing about its
        # contents, so look for real weight files inside it.
        if p.is_dir():
            if self.hf_file:
                return _is_real_file(p / self.hf_file)
            return any(_is_real_file(f)
                       for pat in _WEIGHT_PATTERNS for f in p.rglob(pat))
        return _is_real_file(p)


# ---------------------------------------------------------------------------
# SAM 1
# ---------------------------------------------------------------------------
_SAM1 = [
    ModelSpec("sam1_vit_b", "sam1", "SAM 1 ViT-B (기본)", "sam1",
              "sam_vit_b_01ec64.pth", url=f"{FBAI}/sam_vit_b_01ec64.pth",
              size_mb=375, params_m=91, min_vram_gb=2.5,
              requires=["segment-anything"],
              quant_modes=["fp32", "fp16", "int8"],
              notes="가볍고 안정적. 6GB 이하 GPU 권장."),
    ModelSpec("sam1_vit_l", "sam1", "SAM 1 ViT-L", "sam1",
              "sam_vit_l_0b3195.pth", url=f"{FBAI}/sam_vit_l_0b3195.pth",
              size_mb=1249, params_m=308, min_vram_gb=5.0,
              requires=["segment-anything"],
              quant_modes=["fp32", "fp16", "int8"]),
    ModelSpec("sam1_vit_h", "sam1", "SAM 1 ViT-H (최고 품질)", "sam1",
              "sam_vit_h_4b8939.pth", url=f"{FBAI}/sam_vit_h_4b8939.pth",
              size_mb=2564, params_m=636, min_vram_gb=8.0, cpu_ok=False,
              requires=["segment-anything"],
              quant_modes=["fp32", "fp16"],
              notes="가장 정확하지만 임베딩 계산이 느립니다."),
    ModelSpec("mobile_sam", "sam1", "MobileSAM (초경량, CPU용)", "ultralytics",
              "mobile_sam.pt", url=f"{ULTRA}/mobile_sam.pt",
              size_mb=40, params_m=10, min_vram_gb=1.0,
              requires=["ultralytics"],
              # fp16 measurably degrades this one (score 0.962 -> 0.693 on the
              # same prompt), and at 40MB it saves nothing worth the loss.
              quant_modes=["fp32", "int8"],
              notes="GPU가 없거나 VRAM 4GB 미만일 때 첫 번째 선택."),
]

# ---------------------------------------------------------------------------
# SAM 2 / 2.1  (2.1 is the newer checkpoint set of the same architecture)
# ---------------------------------------------------------------------------
_SAM2 = [
    ModelSpec("sam2_hiera_tiny", "sam2", "SAM 2 Hiera-T", "sam2",
              "sam2_hiera_tiny.pt", url=f"{FBAI2}/072824/sam2_hiera_tiny.pt",
              size_mb=149, params_m=39, min_vram_gb=2.0,
              config="configs/sam2/sam2_hiera_t.yaml", requires=["sam2"]),
    ModelSpec("sam2_hiera_small", "sam2", "SAM 2 Hiera-S", "sam2",
              "sam2_hiera_small.pt", url=f"{FBAI2}/072824/sam2_hiera_small.pt",
              size_mb=176, params_m=46, min_vram_gb=2.5,
              config="configs/sam2/sam2_hiera_s.yaml", requires=["sam2"]),
    ModelSpec("sam2_hiera_base_plus", "sam2", "SAM 2 Hiera-B+", "sam2",
              "sam2_hiera_base_plus.pt",
              url=f"{FBAI2}/072824/sam2_hiera_base_plus.pt",
              size_mb=308, params_m=81, min_vram_gb=4.0,
              config="configs/sam2/sam2_hiera_b+.yaml", requires=["sam2"]),
    ModelSpec("sam2_hiera_large", "sam2", "SAM 2 Hiera-L", "sam2",
              "sam2_hiera_large.pt", url=f"{FBAI2}/072824/sam2_hiera_large.pt",
              size_mb=856, params_m=224, min_vram_gb=6.0, cpu_ok=False,
              config="configs/sam2/sam2_hiera_l.yaml", requires=["sam2"]),
]

_SAM21 = [
    ModelSpec("sam2.1_hiera_tiny", "sam2.1", "SAM 2.1 Hiera-T", "sam2",
              "sam2.1_hiera_tiny.pt", url=f"{FBAI2}/092824/sam2.1_hiera_tiny.pt",
              size_mb=149, params_m=39, min_vram_gb=2.0,
              config="configs/sam2.1/sam2.1_hiera_t.yaml", requires=["sam2"],
              notes="저사양 GPU 기본 추천."),
    ModelSpec("sam2.1_hiera_small", "sam2.1", "SAM 2.1 Hiera-S (권장)", "sam2",
              "sam2.1_hiera_small.pt", url=f"{FBAI2}/092824/sam2.1_hiera_small.pt",
              size_mb=176, params_m=46, min_vram_gb=2.5,
              config="configs/sam2.1/sam2.1_hiera_s.yaml", requires=["sam2"],
              notes="속도/품질 균형이 가장 좋아 기본값으로 씁니다."),
    ModelSpec("sam2.1_hiera_base_plus", "sam2.1", "SAM 2.1 Hiera-B+", "sam2",
              "sam2.1_hiera_base_plus.pt",
              url=f"{FBAI2}/092824/sam2.1_hiera_base_plus.pt",
              size_mb=308, params_m=81, min_vram_gb=4.0,
              config="configs/sam2.1/sam2.1_hiera_b+.yaml", requires=["sam2"]),
    ModelSpec("sam2.1_hiera_large", "sam2.1", "SAM 2.1 Hiera-L (고품질)", "sam2",
              "sam2.1_hiera_large.pt", url=f"{FBAI2}/092824/sam2.1_hiera_large.pt",
              size_mb=856, params_m=224, min_vram_gb=6.0, cpu_ok=False,
              config="configs/sam2.1/sam2.1_hiera_l.yaml", requires=["sam2"]),
    # Ultralytics-packaged mirrors: no sam2 repo install needed.
    ModelSpec("sam2.1_b_ultra", "sam2.1", "SAM 2.1 B (Ultralytics 패키지)",
              "ultralytics", "sam2.1_b.pt", url=f"{ULTRA}/sam2.1_b.pt",
              size_mb=162, params_m=81, min_vram_gb=4.0,
              requires=["ultralytics"],
              notes="sam2 저장소 설치 없이 바로 사용 가능."),
    ModelSpec("sam2.1_t_ultra", "sam2.1", "SAM 2.1 T (Ultralytics 패키지)",
              "ultralytics", "sam2.1_t.pt", url=f"{ULTRA}/sam2.1_t.pt",
              size_mb=79, params_m=39, min_vram_gb=2.0,
              requires=["ultralytics"]),
]

# ---------------------------------------------------------------------------
# ONNX Runtime exports. Unlike the PyTorch entries these DO ship one file per
# precision, so `quant_modes` here selects a file instead of a cast. No torch
# needed, which makes them the practical choice on a CPU-only machine.
# ---------------------------------------------------------------------------
_ONNX_PATTERNS = ["*.json"]
for _b in ("vision_encoder", "prompt_encoder_mask_decoder"):
    for _s in ("", "_fp16", "_int8", "_q4"):
        _ONNX_PATTERNS += [f"onnx/{_b}{_s}.onnx", f"onnx/{_b}{_s}.onnx_data"]

_ONNX = [
    ModelSpec("sam2.1_tiny_onnx", "onnx", "SAM 2.1 Hiera-T (ONNX)", "onnx",
              "sam2.1_tiny_onnx", repo_id="onnx-community/sam2.1-hiera-tiny-ONNX",
              size_mb=332, params_m=39, min_vram_gb=0.0, cpu_ok=True,
              requires=["onnxruntime"],
              quant_modes=["fp32", "fp16", "int8", "q4"],
              hf_patterns=_ONNX_PATTERNS,
              notes="torch 없이 도는 ONNX Runtime 백엔드. 정밀도는 파일로 고릅니다."),
    ModelSpec("sam2.1_small_onnx", "onnx", "SAM 2.1 Hiera-S (ONNX)", "onnx",
              "sam2.1_small_onnx", repo_id="onnx-community/sam2.1-hiera-small-ONNX",
              size_mb=384, params_m=46, min_vram_gb=0.0, cpu_ok=True,
              requires=["onnxruntime"],
              quant_modes=["fp32", "fp16", "int8", "q4"],
              hf_patterns=_ONNX_PATTERNS),
]

# ---------------------------------------------------------------------------
# SAM 3 / 3.1 -- distributed through HuggingFace.
# Repo ids are overridable in the model manager because Meta has moved these
# between orgs before; if a download 404s, edit the id there.
# ---------------------------------------------------------------------------
_SAM3 = [
    ModelSpec("sam3", "sam3", "SAM 3 (텍스트/개념 프롬프트 지원)", "sam3",
              "sam3", repo_id="facebook/sam3",
              size_mb=3500, params_m=848, min_vram_gb=8.0, cpu_ok=False,
              requires=["transformers>=4.57", "huggingface_hub"],
              quant_modes=["fp32", "fp16", "bf16"],
              text_prompt=True,
              notes="클릭·박스 프롬프트에 더해 'car' 같은 텍스트로 전체 "
                    "인스턴스를 한 번에 잡습니다. 클릭은 내장 트래커가 처리합니다."),
    # NOTE: there is no `facebook/sam3-tracker` repository on HuggingFace
    # (checked 2026-09-07: only facebook/sam3 and facebook/sam3.1 exist), and
    # Sam3Backend has no frame-propagation code path. A tracker entry is worth
    # adding back only together with a backend that actually implements it.
    ModelSpec("sam3.1", "sam3.1", "SAM 3.1 (최신)", "sam3",
              "sam3.1", repo_id="facebook/sam3.1",
              size_mb=3500, params_m=848, min_vram_gb=8.0, cpu_ok=False,
              requires=["transformers>=4.57", "huggingface_hub"],
              quant_modes=["fp32", "fp16", "bf16"],
              text_prompt=True, experimental=True,
              notes="⚠ 현재 저장소에는 Meta 원본 체크포인트(sam3.1_multiplex.pt)만 "
                    "있고 transformers 형식(model.safetensors)이 없어 아직 "
                    "불러올 수 없습니다. 변환본 공개 전까지는 SAM 3 를 쓰세요."),
]

CATALOG: Dict[str, ModelSpec] = {
    m.key: m for m in (_SAM1 + _SAM2 + _SAM21 + _SAM3 + _ONNX)
}

FAMILY_ORDER = ["sam1", "sam2", "sam2.1", "sam3", "sam3.1", "onnx"]
FAMILY_LABELS = {
    "sam1": "SAM 1",
    "sam2": "SAM 2",
    "sam2.1": "SAM 2.1",
    "sam3": "SAM 3",
    "sam3.1": "SAM 3.1",
    "onnx": "ONNX (torch 불필요)",
}


def by_family(family: str) -> List[ModelSpec]:
    return [m for m in CATALOG.values() if m.family == family]


def get(key: str) -> Optional[ModelSpec]:
    return CATALOG.get(key)


def installed_models(weights_dir: Path) -> List[ModelSpec]:
    return [m for m in CATALOG.values() if m.is_installed(weights_dir)]


def user_overrides_path(weights_dir: Path) -> Path:
    return Path(weights_dir) / "catalog_overrides.json"


def apply_overrides(weights_dir: Path) -> None:
    """Let users fix a URL / repo_id without touching the source."""
    import json
    p = user_overrides_path(weights_dir)
    if not p.exists():
        return
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    for key, fields in data.items():
        spec = CATALOG.get(key)
        if not spec:
            continue
        for k, v in fields.items():
            if hasattr(spec, k):
                setattr(spec, k, v)
