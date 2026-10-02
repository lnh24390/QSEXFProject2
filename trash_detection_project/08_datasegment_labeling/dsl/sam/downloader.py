"""Hardware probing + on-demand checkpoint download.

Also runnable standalone:

    python -m dsl.sam.downloader --probe
    python -m dsl.sam.downloader --list
    python -m dsl.sam.downloader --recommend
    python -m dsl.sam.downloader --install sam2.1_hiera_small
    python -m dsl.sam.downloader --install-recommended
"""
from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from . import catalog
from .catalog import CATALOG, ModelSpec

ProgressFn = Optional[Callable[[int, int, float], None]]   # done, total, bytes/s


# ---------------------------------------------------------------------------
# hardware probe
# ---------------------------------------------------------------------------
@dataclass
class Hardware:
    has_cuda: bool = False
    gpu_name: str = ""
    vram_gb: float = 0.0
    gpu_count: int = 0
    cpu_count: int = 0
    ram_gb: float = 0.0
    free_disk_gb: float = 0.0
    torch_version: str = ""
    device: str = "cpu"
    torch_error: str = ""      # torch imported but was unusable; see probe_hardware

    def summary(self) -> str:
        if self.has_cuda:
            gpu = f"{self.gpu_name} ({self.vram_gb:.1f}GB VRAM)"
        else:
            gpu = "GPU 없음 (CPU 모드)"
        base = (f"{gpu} · CPU {self.cpu_count}코어 · RAM {self.ram_gb:.1f}GB · "
                f"여유 디스크 {self.free_disk_gb:.0f}GB")
        # ASCII only: this string reaches the Windows console, whose Korean
        # default codepage (cp949) cannot encode most symbol characters.
        return f"{base} · [!] torch 오류: {self.torch_error}" if self.torch_error else base


def _system_ram_gb() -> float:
    try:
        import psutil                                   # optional
        return psutil.virtual_memory().total / 1024 ** 3
    except ImportError:
        pass
    if sys.platform == "win32":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return st.ullTotalPhys / 1024 ** 3
        return 0.0
    try:
        return (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / 1024 ** 3
    except (ValueError, OSError, AttributeError):
        return 0.0


def probe_hardware(weights_dir: Path | str = "weights") -> Hardware:
    hw = Hardware(cpu_count=os.cpu_count() or 1, ram_gb=_system_ram_gb())
    try:
        total, used, free = shutil.disk_usage(Path(weights_dir).anchor or ".")
        hw.free_disk_gb = free / 1024 ** 3
    except OSError:
        pass
    try:
        import torch
        hw.torch_version = torch.__version__
        if torch.cuda.is_available():
            hw.has_cuda = True
            hw.gpu_count = torch.cuda.device_count()
            hw.gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            hw.vram_gb = props.total_memory / 1024 ** 3
            hw.device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            hw.device = "mps"
            hw.gpu_name = "Apple MPS"
    except ImportError:
        pass                    # torch is optional -- nothing to report
    except Exception as e:
        # A torch that imports but does not work: a half-written install, a
        # missing CUDA DLL (OSError WinError 126 is the common Windows case),
        # a driver/runtime mismatch. This probe runs in MainWindow.__init__,
        # so letting it escape would stop the app from starting at all --
        # and DEVELOPMENT.md section 2 requires manual labelling to work with no
        # working ML stack. Record it instead; the hardware panel shows it.
        hw.torch_error = f"{type(e).__name__}: {e}"
        hw.has_cuda = False
        hw.device = "cpu"
    return hw


#: The one place the CUDA wheel index is written down.
CUDA_INSTALL_CMD = ("uv pip install torch torchvision "
                    "--index-url https://download.pytorch.org/whl/cu126")
ONNX_INSTALL_CMD = "uv pip install onnxruntime"


def detect_nvidia_gpu() -> str:
    """GPU name straight from the driver, without importing torch.

    Asking torch would conflate "no GPU" with "torch cannot see the GPU",
    which are the two cases the guidance below has to tell apart.
    """
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=8,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return ""                       # driver/tool absent -> no NVIDIA GPU
    if r.returncode != 0:
        return ""
    lines = [ln.strip() for ln in (r.stdout or "").splitlines() if ln.strip()]
    return lines[0] if lines else ""


def cuda_guidance(hw: Optional[Hardware] = None) -> str:
    """Actionable text about GPU acceleration; "" when there is nothing to do.

    CUDA is optional (DEVELOPMENT.md 2.2): a machine with no NVIDIA GPU is not
    misconfigured, so it is told nothing. Guidance is only produced when a
    GPU exists but torch is not using it.
    """
    hw = hw or probe_hardware()
    if hw.has_cuda:
        return ""                       # already accelerated
    gpu = detect_nvidia_gpu()
    if not gpu:
        return ""                       # CPU-only machine: CPU mode is correct
    if hw.torch_error:
        return (f"NVIDIA GPU({gpu})가 있지만 torch를 불러오지 못했습니다.\n"
                f"  {hw.torch_error}\n\n"
                f"CUDA 빌드로 다시 설치하세요:\n    {CUDA_INSTALL_CMD}")
    if not hw.torch_version:
        return (f"NVIDIA GPU({gpu})가 감지됐습니다. torch가 없어 CPU로 동작합니다.\n\n"
                f"GPU 가속(학습 포함):\n    {CUDA_INSTALL_CMD}\n"
                f"torch 없이 SAM만:\n    {ONNX_INSTALL_CMD}")
    return (f"NVIDIA GPU({gpu})가 있지만 설치된 torch {hw.torch_version} 는 "
            f"CUDA를 쓰지 못합니다.\n(CPU 전용 빌드이거나 드라이버가 맞지 않습니다)\n\n"
            f"CUDA 빌드로 다시 설치하세요:\n    {CUDA_INSTALL_CMD}")


# ---------------------------------------------------------------------------
# recommendation
# ---------------------------------------------------------------------------
@dataclass
class Recommendation:
    spec: ModelSpec
    precision: str
    reason: str


def recommend(hw: Optional[Hardware] = None,
              allow_experimental: bool = False) -> List[Recommendation]:
    """Rank catalog entries for this machine, best first."""
    hw = hw or probe_hardware()
    out: List[Recommendation] = []

    if not hw.has_cuda and hw.device != "mps":
        # CPU only: encoder speed is the bottleneck.
        for key, why in (("mobile_sam", "GPU가 없어 CPU에서도 1초 내에 도는 모델"),
                         ("sam2.1_hiera_tiny", "CPU에서 쓸 만한 SAM 2.1 최소 모델"),
                         ("sam1_vit_b", "CPU 폴백. 이미지당 수 초 걸립니다")):
            spec = CATALOG.get(key)
            if spec:
                prec = "int8" if "int8" in spec.quant_modes else "fp32"
                out.append(Recommendation(spec, prec, why))
        return out

    v = hw.vram_gb
    if v >= 16:
        order = [("sam3.1", "fp16", "VRAM 16GB+ : 텍스트 프롬프트까지 되는 최신 모델"),
                 ("sam3", "fp16", "SAM 3 개념 프롬프트 사용 가능"),
                 ("sam2.1_hiera_large", "fp16", "클릭 라벨링 품질 최상"),
                 ("sam2.1_hiera_base_plus", "fp16", "속도 우선일 때")]
    elif v >= 10:
        order = [("sam2.1_hiera_large", "fp16", "VRAM 10GB+ 에 적합한 최고 품질"),
                 ("sam3", "fp16", "여유가 있으면 텍스트 프롬프트도 가능"),
                 ("sam2.1_hiera_base_plus", "fp16", "더 빠른 대안"),
                 ("sam1_vit_h", "fp16", "SAM 1 최고 품질")]
    elif v >= 6:
        order = [("sam2.1_hiera_base_plus", "fp16", "VRAM 6~10GB 에 가장 균형적"),
                 ("sam2.1_hiera_small", "fp16", "더 가볍고 빠름"),
                 ("sam1_vit_l", "fp16", "SAM 1 계열 대안")]
    elif v >= 4:
        order = [("sam2.1_hiera_small", "fp16", "VRAM 4~6GB 기본 추천"),
                 ("sam2.1_hiera_tiny", "fp16", "여유가 없을 때"),
                 ("sam1_vit_b", "fp16", "SAM 1 계열 대안")]
    else:
        order = [("sam2.1_hiera_tiny", "fp16", "VRAM 4GB 미만"),
                 ("mobile_sam", "fp16", "가장 가벼운 선택"),
                 ("sam1_vit_b", "fp16", "여유가 있으면")]

    for key, prec, why in order:
        spec = CATALOG.get(key)
        if not spec:
            continue
        if spec.experimental and not allow_experimental:
            why += " (실험적)"
        if spec.min_vram_gb > v + 0.5:
            continue
        if prec not in spec.quant_modes:
            prec = spec.quant_modes[0]
        out.append(Recommendation(spec, prec, why))
    return out


def best_precision(spec: ModelSpec, hw: Optional[Hardware] = None) -> str:
    hw = hw or probe_hardware()
    if not hw.has_cuda:
        return "int8" if "int8" in spec.quant_modes else "fp32"
    if hw.vram_gb < spec.min_vram_gb * 2 and "fp16" in spec.quant_modes:
        return "fp16"
    return "fp16" if "fp16" in spec.quant_modes else "fp32"


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------
class DownloadError(RuntimeError):
    pass


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def download_url(url: str, dest: Path, progress: ProgressFn = None,
                 chunk: int = 1 << 20, cancel=None) -> Path:
    """Resumable download to `dest`, staged through dest.part."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0

    req = urllib.request.Request(url, headers={"User-Agent": "DataSegmentLabeling/1.0"})
    if have:
        req.add_header("Range", f"bytes={have}-")

    try:
        resp = urllib.request.urlopen(req, timeout=30)
    except urllib.error.HTTPError as e:
        if e.code == 416 and have:              # already complete
            part.replace(dest)
            return dest
        if e.code in (403, 404):
            raise DownloadError(
                f"다운로드 실패 ({e.code}). URL이 바뀌었을 수 있습니다: {url}") from e
        raise DownloadError(f"다운로드 실패 ({e.code}): {url}") from e
    except urllib.error.URLError as e:
        raise DownloadError(f"네트워크 오류: {e.reason}") from e

    with resp:
        if resp.status == 200:
            have = 0                            # server ignored Range
        total = int(resp.headers.get("Content-Length", 0)) + have
        mode = "ab" if have else "wb"
        t0 = time.time()
        done = have
        with open(part, mode) as f:
            while True:
                if cancel is not None and cancel():
                    raise DownloadError("사용자가 다운로드를 취소했습니다.")
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                if progress:
                    dt = max(time.time() - t0, 1e-6)
                    progress(done, total, (done - have) / dt)

    if part.stat().st_size == 0:
        part.unlink(missing_ok=True)
        raise DownloadError("빈 파일이 내려왔습니다.")
    part.replace(dest)
    return dest


def download_hf(repo_id: str, dest_dir: Path, hf_file: str = "",
                progress: ProgressFn = None,
                patterns: list[str] | None = None) -> Path:
    """Fetch a HuggingFace repo (SAM 3 / 3.1) via huggingface_hub."""
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as e:
        raise DownloadError(
            "huggingface_hub 가 필요합니다:  uv pip install huggingface_hub") from e

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        if hf_file:
            p = hf_hub_download(repo_id=repo_id, filename=hf_file,
                                local_dir=str(dest_dir))
        else:
            # Both weight formats are pulled: facebook/sam3 ships
            # model.safetensors *and* sam3.pt, while facebook/sam3.1 ships only
            # sam3.1_multiplex.pt -- excluding *.pt left SAM 3.1 with configs
            # but no weights at all.
            p = snapshot_download(repo_id=repo_id, local_dir=str(dest_dir),
                                  allow_patterns=patterns or
                                  ["*.json", "*.safetensors",
                                   "*.bin", "*.pt", "*.pth",
                                   "*.txt", "*.model",
                                   "*.yaml", "*.py"])
    except Exception as e:                       # hub raises many types
        raise DownloadError(
            f"HuggingFace 다운로드 실패: {repo_id}\n{e}\n"
            "비공개/게이트 저장소라면 `huggingface-cli login` 후 다시 시도하세요.") from e
    return Path(p)


def install(key: str, weights_dir: Path | str = "weights",
            progress: ProgressFn = None, cancel=None,
            force: bool = False) -> Path:
    """Download one catalog entry into weights/<family>/.

    `force` re-runs the fetch even when the entry already counts as installed.
    That is what completes a *partial* HF snapshot: `is_installed` only asks
    whether some weight file is present, so a repo that ships two of them
    (facebook/sam3 has model.safetensors and sam3.pt) would otherwise stop at
    the first one. Files already on disk are skipped by the hub, so this is
    cheap to re-run.
    """
    weights_dir = Path(weights_dir)
    catalog.apply_overrides(weights_dir)
    spec = CATALOG.get(key)
    if spec is None:
        raise DownloadError(f"카탈로그에 없는 모델: {key}")

    dest = spec.local_path(weights_dir)
    if spec.is_installed(weights_dir) and not force:
        return dest
    if spec.url:
        return download_url(spec.url, dest, progress=progress, cancel=cancel)
    if spec.repo_id:
        return download_hf(spec.repo_id, dest, spec.hf_file, progress=progress,
                           patterns=spec.hf_patterns or None)
    raise DownloadError(f"{key} 에 다운로드 경로가 없습니다.")


def uninstall(key: str, weights_dir: Path | str = "weights") -> bool:
    spec = CATALOG.get(key)
    if not spec:
        return False
    p = spec.local_path(Path(weights_dir))
    if p.is_dir():
        shutil.rmtree(p, ignore_errors=True)
        return True
    if p.exists():
        p.unlink()
        return True
    return False


def missing_packages(spec: ModelSpec) -> List[str]:
    """Which pip packages this checkpoint still needs."""
    import importlib.util
    mods = {"segment-anything": "segment_anything", "sam2": "sam2",
            "ultralytics": "ultralytics", "huggingface_hub": "huggingface_hub"}
    out = []
    for req in spec.requires:
        base = req.split(">=")[0].split("==")[0].strip()
        mod = mods.get(base, base.replace("-", "_"))
        if importlib.util.find_spec(mod) is None:
            out.append(req)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli(argv: List[str]) -> int:
    import argparse
    for stream in (sys.stdout, sys.stderr):      # Korean output on cp949 consoles
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass
    ap = argparse.ArgumentParser(description="SAM 체크포인트 다운로드 도구")
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--probe", action="store_true", help="하드웨어 정보 출력")
    ap.add_argument("--list", action="store_true", help="카탈로그 출력")
    ap.add_argument("--recommend", action="store_true", help="이 PC 추천 모델")
    ap.add_argument("--install", metavar="KEY", help="모델 다운로드")
    ap.add_argument("--install-recommended", action="store_true")
    ap.add_argument("--uninstall", metavar="KEY")
    a = ap.parse_args(argv)

    wd = Path(a.weights)
    catalog.apply_overrides(wd)
    hw = probe_hardware(wd)

    if a.probe or not any([a.list, a.recommend, a.install,
                           a.install_recommended, a.uninstall]):
        print("[하드웨어]", hw.summary())
        print("  torch:", hw.torch_version or "미설치", "| device:", hw.device)

    if a.list:
        print(f"\n{'KEY':26} {'계열':7} {'크기':>8}  {'최소VRAM':>8}  설치")
        for fam in catalog.FAMILY_ORDER:
            for m in catalog.by_family(fam):
                mark = "O" if m.is_installed(wd) else "-"
                print(f"{m.key:26} {m.family:7} {m.size_mb:6}MB  "
                      f"{m.min_vram_gb:6.1f}GB  {mark}   {m.display}")

    if a.recommend or a.install_recommended:
        recs = recommend(hw)
        print("\n[이 PC 추천]")
        for i, r in enumerate(recs, 1):
            print(f"  {i}. {r.spec.key} ({r.precision}) - {r.reason}")
        if a.install_recommended and recs:
            a.install = recs[0].spec.key

    if a.uninstall:
        print("삭제:", "완료" if uninstall(a.uninstall, wd) else "설치되어 있지 않음")

    if a.install:
        spec = CATALOG.get(a.install)
        if not spec:
            print("알 수 없는 모델:", a.install)
            return 2
        miss = missing_packages(spec)
        if miss:
            print("먼저 설치가 필요합니다:  uv pip install " + " ".join(miss))
        last = [0.0]

        def prog(done, total, speed):
            now = time.time()
            if now - last[0] < 0.3 and done < total:
                return
            last[0] = now
            pct = (done / total * 100) if total else 0
            sys.stdout.write(f"\r  {spec.key}: {pct:5.1f}%  "
                             f"{_human(done)}/{_human(total)}  {_human(speed)}/s   ")
            sys.stdout.flush()

        print(f"\n다운로드 -> {spec.local_path(wd)}")
        try:
            p = install(a.install, wd, progress=prog)
        except DownloadError as e:
            print("\n오류:", e)
            return 1
        print(f"\n완료: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv[1:]))
