import time
from collections import Counter
from pathlib import Path

import numpy as np
import ultralytics
from ultralytics import YOLO


def _version_tuple(text):
    parts = []
    for chunk in str(text).split(".")[:3]:
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


# ultralytics 8.4.80 부터 half=True / int8=True 가 quantize=16 / quantize=8 로 통합됐다.
# 구버전 ultralytics 가 깔린 JetPack 환경도 있어서 버전을 보고 인자를 맞춘다.
USES_QUANTIZE = _version_tuple(getattr(ultralytics, "__version__", "0")) >= (8, 4, 80)


def precision_kwargs(quantize, for_export=False):
    """quantize(16/8/32/None)를 설치된 ultralytics 버전에 맞는 인자로 변환한다."""
    if quantize in (None, 32, "32", "fp32"):
        return {}

    value = int(quantize)
    if USES_QUANTIZE:
        return {"quantize": value}

    # 구버전: half / int8 불리언으로 되돌린다.
    if value == 16:
        return {"half": True}
    if value == 8:
        return {"int8": True} if for_export else {}
    return {}


IS_JETSON = Path("/etc/nv_tegra_release").exists()

# Jetson GPU 아키텍처. 휠이 이 중 아무것도 포함하지 않으면 Orin/Xavier/Nano에서 못 돈다.
JETSON_ARCHS = ("sm_87", "sm_72", "sm_62", "sm_53")


def _tegra_cuda_version():
    """이 장비에 실제로 설치된 CUDA 툴킷 버전 (예: '12.6')."""
    for path in sorted(Path("/usr/local").glob("cuda-*"), reverse=True):
        suffix = path.name[len("cuda-"):]
        if suffix and suffix[0].isdigit() and "." in suffix:
            return suffix
    return None


def diagnose_no_cuda():
    """CUDA를 못 쓰는 이유를 짚어 해결 방법까지 돌려준다."""
    try:
        import torch
    except ImportError:
        return ["PyTorch가 설치되어 있지 않습니다."]

    lines = [f"설치된 PyTorch : {torch.__version__} (빌드 CUDA {torch.version.cuda})"]
    if not IS_JETSON:
        lines.append("GPU 드라이버와 CUDA 설치 상태를 확인하세요.")
        return lines

    driver = _tegra_cuda_version()
    if driver:
        lines.append(f"장비 CUDA 툴킷 : {driver}")

    archs = list(getattr(torch.cuda, "get_arch_list", list)())
    if archs and not any(a in JETSON_ARCHS for a in archs):
        lines.append(f"이 휠은 Jetson GPU({JETSON_ARCHS[0]})용으로 빌드되지 않았습니다.")
        lines.append(f"  휠이 지원하는 아키텍처: {', '.join(archs)}")

    lines.append("JetPack 전용 PyTorch 휠을 설치해야 합니다. JetPack 6 / CUDA 12.6 기준:")
    lines.append("  pip3 install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \\")
    lines.append("      torch==2.8.0 torchvision==0.23.0")
    return lines


def _warn_cpu_fallback(reason):
    """CPU 폴백을 조용히 넘기지 않는다.

    Jetson에 맞지 않는 PyTorch 휠이 깔리면 torch.cuda.is_available()이 그냥
    False가 되어, 원인은 안 보이고 FPS만 10배 느려진다. 눈에 띄게 알려서
    모델이나 카메라를 엉뚱하게 의심하지 않도록 한다.
    """
    bar = "=" * 70
    print(f"\n{bar}")
    print(f">> 경고: {reason} CPU로 실행합니다 — 실시간 추론에는 너무 느립니다.")
    for line in diagnose_no_cuda():
        print(f"   {line}")
    print(f"{bar}\n")


def resolve_device(requested="auto"):
    """사용 가능한 장치를 확인해 실제로 쓸 device 문자열을 돌려준다."""
    try:
        import torch
    except ImportError:
        _warn_cpu_fallback("PyTorch를 불러오지 못했습니다.")
        return "cpu"

    requested = "auto" if requested is None else str(requested).lower()
    if requested == "cpu":
        return "cpu"

    if not torch.cuda.is_available():
        _warn_cpu_fallback("CUDA를 쓸 수 없습니다."
                           if requested == "auto"
                           else f"'{requested}' 장치를 쓸 수 없습니다.")
        return "cpu"

    if requested == "auto":
        return "cuda:0"
    # "0" 처럼 번호만 준 경우 torch가 이해하는 형태로 맞춘다.
    return f"cuda:{requested}" if requested.isdigit() else requested


def discover_models(model_dir, prefer_engine=True):
    """model_dir 안의 가중치를 찾는다.

    같은 이름의 .engine(TensorRT)과 .pt가 함께 있으면 훨씬 빠른 .engine을 쓴다.
    prefer_engine=False 면 .pt 를 고른다. FP16 엔진과 원본 가중치의 검출
    결과를 눈으로 비교할 때 쓴다.
    """
    model_dir = Path(model_dir)
    engines = {p.stem: p for p in sorted(model_dir.glob("*.engine"))}
    weights = {p.stem: p for p in sorted(model_dir.glob("*.pt"))}
    if not engines and not weights:
        raise FileNotFoundError(f"No model files (.pt/.engine) were found in {model_dir}")

    models = []
    for stem in sorted(set(engines) | set(weights)):
        if prefer_engine:
            path = engines.get(stem, weights.get(stem))
        else:
            path = weights.get(stem, engines.get(stem))
        models.append({
            "name": stem,
            "path": str(path),
            "engine": str(path).endswith(".engine"),
        })
    return models


class Detector:
    """YOLO 모델 로딩 / 전환 / 추론을 담당한다."""

    def __init__(self, model_list, index=0, device="auto", imgsz=640, quantize=None):
        if not model_list:
            raise ValueError("model_list가 비어 있습니다.")

        self.model_list = model_list
        self.index = index % len(model_list)
        self.device = resolve_device(device)
        self.imgsz = imgsz
        # FP16은 GPU에서만 의미가 있다.
        if quantize == 16 and self.device == "cpu":
            print(">> CPU에서는 FP16이 오히려 느려 FP32로 실행합니다.")
            quantize = None
        self.quantize = quantize
        self.precision = precision_kwargs(quantize)
        self.model = None
        self.load(self.index)

    @property
    def precision_label(self):
        return {16: "FP16", 8: "INT8"}.get(self.quantize, "FP32")

    @property
    def name(self):
        return self.model_list[self.index]["name"]

    @property
    def task(self):
        task = getattr(self.model, "task", "detect")
        return f"{task}/TRT" if self.is_engine else task

    @property
    def is_engine(self):
        return bool(self.model_list[self.index].get("engine"))

    def load(self, index):
        self.index = index % len(self.model_list)
        info = self.model_list[self.index]
        print(f"\n>> Loading {info['name']} ... (device={self.device})")
        start = time.time()
        self.model = YOLO(info["path"])
        # TensorRT 엔진은 이미 특정 GPU에 묶여 빌드되어 있어 .to() 를 지원하지 않는다.
        # 매번 긴 예외 메시지를 찍는 대신 아예 건너뛴다 (device 는 predict 로 넘긴다).
        if self.device != "cpu" and not self.is_engine:
            try:
                self.model.to(self.device)
            except Exception as exc:
                print(f">> {self.device} 로 올리지 못했습니다 ({exc}).")
        self._warmup()
        print(f">> Loaded in {time.time() - start:.2f}s (task={self.task})")
        return self.model

    def next_model(self):
        return self.load(self.index + 1)

    def prev_model(self):
        return self.load(self.index - 1)

    def _warmup(self):
        """첫 프레임에서 지연이 튀지 않도록 더미 입력으로 한 번 돌린다."""
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        try:
            self.model.predict(dummy, imgsz=self.imgsz, device=self.device,
                               verbose=False, **self.precision)
        except Exception as exc:  # 워밍업 실패가 실행을 막을 이유는 없다.
            print(f">> warmup skipped: {exc}")

    def predict(self, frame, conf=0.5, iou=0.45, max_det=300):
        results = self.model.predict(
            frame,
            conf=conf,
            iou=iou,
            imgsz=self.imgsz,
            device=self.device,
            max_det=max_det,
            verbose=False,
            **self.precision,
        )
        return results[0] if results else None

    def summarize(self, result):
        """클래스별 검출 개수를 {이름: 개수} 로 돌려준다."""
        boxes = getattr(result, "boxes", None) if result is not None else None
        if boxes is None or len(boxes) == 0:
            return {}

        names = result.names
        cls_ids = boxes.cls.int().tolist()
        if isinstance(names, (list, tuple)):
            counts = Counter(names[c] if c < len(names) else str(c) for c in cls_ids)
        else:
            counts = Counter(names.get(c, str(c)) for c in cls_ids)
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
