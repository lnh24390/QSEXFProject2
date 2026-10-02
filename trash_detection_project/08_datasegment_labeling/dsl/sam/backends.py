"""Concrete SAM backends: SAM 1, SAM 2/2.1, Ultralytics-packaged, SAM 3/3.1.

Every heavy import happens inside `load()` so the app starts with none of these
libraries installed (DEVELOPMENT.md section 2).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from .base import EMPTY, SamBackend, SamBackendError, SamResult
from .catalog import CATALOG, ModelSpec


def _need(pkg: str, pip: str):
    return SamBackendError(
        f"'{pkg}' 패키지가 필요합니다.\n\n    uv pip install {pip}\n\n"
        "설치 후 모델을 다시 불러오세요.")


# ---------------------------------------------------------------------------
# SAM 1  (facebookresearch/segment-anything)
# ---------------------------------------------------------------------------
class Sam1Backend(SamBackend):
    supports_auto = True
    _ARCH = {"sam1_vit_b": "vit_b", "sam1_vit_l": "vit_l", "sam1_vit_h": "vit_h"}

    def load(self) -> None:
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as e:
            raise _need("segment_anything",
                        "git+https://github.com/facebookresearch/segment-anything.git") from e
        if not self.checkpoint.exists():
            raise SamBackendError(f"체크포인트가 없습니다: {self.checkpoint}\n"
                                  "모델 관리자에서 먼저 다운로드하세요.")
        arch = self._ARCH.get(self.spec.key, "vit_b")
        model = sam_model_registry[arch](checkpoint=str(self.checkpoint))
        model = self.apply_precision(model)   # cast before it reaches the GPU
        model.to(self.device)
        model.eval()
        self.model = model
        self.predictor = SamPredictor(model)

    def _encode(self, rgb: np.ndarray) -> None:
        with self.autocast():
            self.predictor.set_image(rgb)

    def auto_masks(self, rgb, max_masks: int = 200):
        try:
            from segment_anything import SamAutomaticMaskGenerator
        except ImportError as e:
            raise _need("segment_anything",
                        "git+https://github.com/facebookresearch/segment-anything.git") from e
        if self.model is None:
            raise SamBackendError("모델이 로드되지 않았습니다.")
        gen = SamAutomaticMaskGenerator(self.model, points_per_side=24,
                                        min_mask_region_area=64)
        with self.autocast():
            found = gen.generate(rgb)
        found.sort(key=lambda d: -float(d.get("area", 0)))
        return [d["segmentation"] for d in found[:max_masks]]

    def predict(self, points=(), box=None, mask_input=None,
                multimask: bool = True) -> SamResult:
        if self.predictor is None or self.predictor.features is None:
            return EMPTY
        coords, labels = self._split_points(points)
        box_arr = np.asarray(box, dtype=np.float32)[None] if box is not None else None
        with self.autocast():
            masks, scores, _logits = self.predictor.predict(
                point_coords=coords, point_labels=labels, box=box_arr,
                mask_input=mask_input, multimask_output=multimask)
        return self._sorted(masks, scores)


# ---------------------------------------------------------------------------
# SAM 2 / 2.1  (facebookresearch/sam2)
# ---------------------------------------------------------------------------
class Sam2Backend(SamBackend):
    supports_auto = True

    def load(self) -> None:
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as e:
            raise _need("sam2", "git+https://github.com/facebookresearch/sam2.git") from e
        if not self.checkpoint.exists():
            raise SamBackendError(f"체크포인트가 없습니다: {self.checkpoint}\n"
                                  "모델 관리자에서 먼저 다운로드하세요.")
        cfg = self.spec.config or "configs/sam2.1/sam2.1_hiera_s.yaml"
        try:
            model = build_sam2(cfg, str(self.checkpoint), device=self.device)
        except Exception as e:                    # hydra raises its own types
            raise SamBackendError(
                f"SAM 2 모델 생성 실패 (config: {cfg})\n{e}\n"
                "sam2 버전과 체크포인트 조합이 맞는지 확인하세요.") from e
        model = self.apply_precision(model)
        model.to(self.device)
        model.eval()
        self.model = model
        self.predictor = SAM2ImagePredictor(model)

    def _encode(self, rgb: np.ndarray) -> None:
        with self.autocast():
            self.predictor.set_image(rgb)

    def auto_masks(self, rgb, max_masks: int = 200):
        try:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        except ImportError as e:
            raise _need("sam2",
                        "git+https://github.com/facebookresearch/sam2.git") from e
        if self.model is None:
            raise SamBackendError("모델이 로드되지 않았습니다.")
        gen = SAM2AutomaticMaskGenerator(self.model, points_per_side=24,
                                         min_mask_region_area=64)
        with self.autocast():
            found = gen.generate(rgb)
        found.sort(key=lambda d: -float(d.get("area", 0)))
        return [d["segmentation"] for d in found[:max_masks]]

    def predict(self, points=(), box=None, mask_input=None,
                multimask: bool = True) -> SamResult:
        if self.predictor is None:
            return EMPTY
        coords, labels = self._split_points(points)
        box_arr = np.asarray(box, dtype=np.float32)[None] if box is not None else None
        with self.autocast():
            masks, scores, _logits = self.predictor.predict(
                point_coords=coords, point_labels=labels, box=box_arr,
                mask_input=mask_input, multimask_output=multimask)
        return self._sorted(masks, scores)


# ---------------------------------------------------------------------------
# Ultralytics-packaged SAM (MobileSAM, sam_b, sam2.1_b/t ...)
# ---------------------------------------------------------------------------
class UltralyticsSamBackend(SamBackend):
    """No persistent embedding API, so the RGB frame is kept and re-fed."""

    supports_auto = True

    def load(self) -> None:
        try:
            from ultralytics import SAM
        except ImportError as e:
            raise _need("ultralytics", "ultralytics") from e
        if not self.checkpoint.exists():
            raise SamBackendError(f"체크포인트가 없습니다: {self.checkpoint}")
        self.model = SAM(str(self.checkpoint))
        try:
            self.model.to(self.device)
        except (AttributeError, RuntimeError):
            pass
        # Ultralytics owns the forward pass, so fp16 is requested per call
        # rather than by casting the module ourselves. int8 has no equivalent
        # hook, so it is applied to the wrapped nn.Module directly.
        #
        # 8.4 replaced `half=True` with the unified `quantize=16` and dropped
        # `half` from the default config. Probe the config instead of pinning
        # a version: the old kwarg warns on new builds, the new one is
        # unknown to old ones, and both spellings must keep working.
        self._precision_kw: dict = {}
        if self.precision == "fp16" and self.device.startswith("cuda"):
            try:
                from ultralytics.cfg import DEFAULT_CFG_DICT
                new_api = "quantize" in DEFAULT_CFG_DICT
            except (ImportError, AttributeError):
                new_api = False
            self._precision_kw = {"quantize": 16} if new_api else {"half": True}
        if self.precision == "int8" and self.device == "cpu":
            inner = getattr(self.model, "model", None)
            if inner is not None:
                self.model.model = self.quantize_cpu(inner)
        self._rgb: Optional[np.ndarray] = None

    def _encode(self, rgb: np.ndarray) -> None:
        self._rgb = rgb                            # BGR conversion done at call

    def reset_image(self) -> None:
        self._rgb = None
        self._image_key = None
        self.empty_cache()

    def auto_masks(self, rgb, max_masks: int = 200):
        if self.model is None:
            raise SamBackendError("모델이 로드되지 않았습니다.")
        import cv2
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        # No points and no boxes is ultralytics' "everything" mode.
        res = self.model(bgr, verbose=False, device=self.device,
                         **self._precision_kw)
        if not res or res[0].masks is None:
            return []
        arr = res[0].masks.data.cpu().numpy() > 0
        order = np.argsort([-m.sum() for m in arr])
        return [arr[i] for i in order[:max_masks]]

    def predict(self, points=(), box=None, mask_input=None,
                multimask: bool = True) -> SamResult:
        if self.model is None or self._rgb is None:
            return EMPTY
        import cv2
        bgr = cv2.cvtColor(self._rgb, cv2.COLOR_RGB2BGR)
        kw = {}
        coords, labels = self._split_points(points)
        if coords is not None:
            kw["points"] = coords.tolist()
            kw["labels"] = labels.tolist()
        if box is not None:
            kw["bboxes"] = [list(map(float, box))]
        if not kw:
            return EMPTY
        try:
            res = self.model(bgr, verbose=False, device=self.device,
                             **self._precision_kw, **kw)
        except Exception as e:
            raise SamBackendError(f"Ultralytics SAM 추론 실패: {e}") from e
        if not res or res[0].masks is None:
            return EMPTY
        masks = res[0].masks.data.cpu().numpy()
        conf = getattr(res[0].boxes, "conf", None)
        scores = conf.cpu().numpy() if conf is not None else np.ones(len(masks), np.float32)
        return self._sorted(masks, scores)


# ---------------------------------------------------------------------------
# SAM 3 / 3.1  (HuggingFace transformers; adds text/concept prompting)
# ---------------------------------------------------------------------------
class Sam3Backend(SamBackend):
    supports_text = True
    # Text and boxes go through the detector (`Sam3Processor` + the top-level
    # model). Points go through the *tracker* submodule instead: the detector
    # processor has no `input_points`, but `Sam3TrackerVideoModel` does, and
    # `Sam3VideoModel` bundles one that shares the detector's vision encoder
    # via `get_vision_features_for_tracker` -- whose own comment says the
    # features are cached "to avoid running it again on every SAM click".
    supports_points = True

    def load(self) -> None:
        try:
            import torch
            import transformers
        except ImportError as e:
            raise _need("transformers", "transformers huggingface_hub") from e

        src = str(self.checkpoint) if self.checkpoint.exists() else (self.spec.repo_id
                                                                    or self.spec.key)
        self._torch = torch
        model = processor = None
        errors = []
        # transformers renamed these classes between previews; try in order.
        # Sam3VideoModel first: it is what the published configs declare, and
        # it is the only one that carries the tracker needed for clicking. The
        # detector-only Sam3Model is a fallback for text/box-only builds.
        for proc_name, model_name in (("Sam3Processor", "Sam3VideoModel"),
                                      ("Sam3Processor", "Sam3Model"),
                                      ("Sam3Processor", "Sam3ForConditionalGeneration"),
                                      ("AutoProcessor", "AutoModel")):
            try:
                Proc = getattr(transformers, proc_name)
                Model = getattr(transformers, model_name)
            except AttributeError as e:
                errors.append(f"{proc_name}/{model_name}: {e}")
                continue
            try:
                processor = Proc.from_pretrained(src, trust_remote_code=True)
                dtype = (torch.float16 if self.precision == "fp16"
                         else torch.bfloat16 if self.precision in ("bf16", "bfloat16")
                         else torch.float32)
                model = Model.from_pretrained(src, dtype=dtype,
                                              trust_remote_code=True)
                break
            except Exception as e:                 # hub/network/class mismatch
                errors.append(f"{proc_name}/{model_name}: {e}")
                processor = model = None

        if model is None:
            # A repo that ships only the original Meta checkpoint (a bare
            # state_dict such as sam3.1_multiplex.pt, keys prefixed
            # `detector.` / `tracker.`) has no transformers-format weights to
            # load. Say that plainly instead of dumping class-name errors.
            local = Path(src)
            if local.is_dir() and not any(
                    (local / n).exists() for n in ("model.safetensors",
                                                   "model.safetensors.index.json",
                                                   "pytorch_model.bin")):
                raise SamBackendError(
                    f"{self.spec.display} 은 아직 transformers 형식이 아닙니다.\n"
                    f"경로: {src}\n\n"
                    "이 저장소에는 Meta 원본 체크포인트(.pt)만 있고 "
                    "transformers 가 읽는 model.safetensors 가 없습니다.\n"
                    "변환본이 공개되기 전까지는 SAM 3 를 사용하세요.")
            raise SamBackendError(
                "SAM 3 모델을 불러오지 못했습니다.\n"
                f"경로/저장소: {src}\n\n"
                "확인할 것:\n"
                "  1) uv pip install -U transformers huggingface_hub\n"
                "  2) 모델 관리자에서 SAM 3 다운로드가 끝났는지\n"
                "  3) 저장소 ID가 바뀌었다면 weights/catalog_overrides.json 에서 수정\n\n"
                "시도 기록:\n  " + "\n  ".join(errors[-3:]))

        model.to(self.device)
        model.eval()
        self.model = model
        self.processor = processor
        # Sam3VideoModel wraps the detector; text/box prompting talks to that
        # inner model, since the wrapper's forward is session-based.
        self._detector = getattr(model, "detector_model", model)
        self._rgb: Optional[np.ndarray] = None
        self._src = src
        self._tracker_proc = None      # built lazily on the first click
        self._session = None

    # ---- point prompts (tracker submodule) --------------------------------
    def _tracker(self):
        """(processor, tracker_model) or None when this build has no tracker."""
        tracker = getattr(self.model, "tracker_model", None)
        if tracker is None:
            return None
        if self._tracker_proc is None:
            try:
                from transformers.models.sam3_tracker_video \
                    .processing_sam3_tracker_video import Sam3TrackerVideoProcessor
                self._tracker_proc = Sam3TrackerVideoProcessor.from_pretrained(
                    self._src)
            except Exception as e:                 # class renamed / not shipped
                raise SamBackendError(
                    f"SAM 3 트래커 프로세서를 만들지 못했습니다: {e}") from e
        return self._tracker_proc, tracker

    def _ensure_session(self):
        """One session (and one vision-feature pass) per image, reused per click."""
        if self._session is not None:
            return self._session
        proc, _tracker = self._tracker()
        torch = self._torch
        dtype = (torch.float16 if self.precision == "fp16"
                 else torch.bfloat16 if self.precision in ("bf16", "bfloat16")
                 else torch.float32)
        sess = proc.init_video_session(video=[self._pil()],
                                       inference_device=self.device, dtype=dtype)
        with torch.inference_mode():
            pixel_values = sess.get_frame(0).unsqueeze(0)
            vision_embeds = self.model.detector_model.get_vision_features(
                pixel_values=pixel_values)
            feats, pos = self.model.get_vision_features_for_tracker(
                vision_embeds=vision_embeds)
            sess.cache.cache_vision_features(
                0, {"vision_feats": feats, "vision_pos_embeds": pos})
        self._session = sess
        return sess

    def _predict_points(self, coords, labels) -> SamResult:
        proc, tracker = self._tracker()
        torch = self._torch
        sess = self._ensure_session()
        h, w = self._rgb.shape[:2]
        try:
            with torch.inference_mode():
                proc.add_inputs_to_inference_session(
                    inference_session=sess, frame_idx=0, obj_ids=1,
                    input_points=[[coords.tolist()]],
                    input_labels=[[labels.tolist()]],
                    original_size=(h, w))
                out = tracker(inference_session=sess, frame_idx=0)
                low_res = out.pred_masks
                full = proc.post_process_masks([low_res],
                                               original_sizes=[(h, w)],
                                               binarize=False)[0]
                masks = (full > 0).squeeze(1).float().cpu().numpy()
        except Exception as e:
            raise SamBackendError(f"SAM 3 클릭 추론 실패: {e}") from e
        masks = masks.reshape(-1, *masks.shape[-2:])
        scores = np.ones((len(masks),), dtype=np.float32)
        if getattr(out, "object_score_logits", None) is not None:
            s = out.object_score_logits.detach().float().cpu().numpy().reshape(-1)
            if len(s) == len(masks):
                scores = 1.0 / (1.0 + np.exp(-s))       # logit -> 0..1
        return self._sorted(masks, scores)

    def _encode(self, rgb: np.ndarray) -> None:
        self._rgb = rgb
        self._session = None           # features belong to the previous image

    def reset_image(self) -> None:
        self._rgb = None
        self._session = None
        self._image_key = None
        self.empty_cache()

    def _pil(self):
        from PIL import Image
        return Image.fromarray(self._rgb)

    def _run(self, **proc_kw) -> SamResult:
        torch = self._torch
        try:
            inputs = self.processor(images=self._pil(), return_tensors="pt",
                                    **proc_kw).to(self.device)
            with torch.inference_mode(), self.autocast():
                outputs = self._detector(**inputs)
            post = getattr(self.processor, "post_process_instance_segmentation", None) \
                or getattr(self.processor, "post_process_masks", None)
            if post is None:
                raise SamBackendError("processor 에 후처리 함수가 없습니다.")
            h, w = self._rgb.shape[:2]
            try:
                results = post(outputs, target_sizes=[(h, w)])
            except TypeError:
                results = post(outputs, original_sizes=[(h, w)])
        except SamBackendError:
            raise
        except Exception as e:
            raise SamBackendError(f"SAM 3 추론 실패: {e}") from e

        r = results[0] if isinstance(results, (list, tuple)) else results
        masks = r["masks"] if isinstance(r, dict) and "masks" in r else r
        scores = r.get("scores") if isinstance(r, dict) else None
        # `.float()` before numpy: bf16 tensors have no numpy dtype, so a bf16
        # run would otherwise die with "unsupported ScalarType BFloat16".
        masks = (masks.detach().float().cpu().numpy() if hasattr(masks, "detach")
                 else np.asarray(masks))
        masks = masks.reshape(-1, *masks.shape[-2:])
        if scores is not None and hasattr(scores, "detach"):
            scores = scores.detach().float().cpu().numpy().reshape(-1)
        else:
            scores = np.ones((len(masks),), dtype=np.float32)
        return self._sorted(masks, scores)

    def predict(self, points=(), box=None, mask_input=None,
                multimask: bool = True) -> SamResult:
        if self.model is None or self._rgb is None:
            return EMPTY
        coords, labels = self._split_points(points)
        if coords is not None:
            # Clicks are the tracker's job; the detector processor has no
            # input_points. Box-only prompts still go to the detector below.
            if self._tracker() is None:
                raise SamBackendError(
                    f"{self.spec.display} 빌드에 트래커가 없어 점 프롬프트를 "
                    "쓸 수 없습니다. 박스/텍스트 프롬프트를 사용하세요.")
            return self._predict_points(coords, labels)
        if box is not None:
            return self._run(input_boxes=[[list(map(float, box))]])
        return EMPTY

    def predict_text(self, text: str) -> SamResult:
        """SAM 3's headline feature: 'red car' -> every matching instance."""
        if self.model is None or self._rgb is None:
            return EMPTY
        res = self._run(text=[text])
        res.labels = [text] * len(res)
        return res


# ---------------------------------------------------------------------------
# ONNX Runtime (onnx-community exports of SAM 2 / 2.1)
# ---------------------------------------------------------------------------
class OnnxSamBackend(SamBackend):
    """Two-session SAM: a vision encoder run once per image, then a light
    prompt-encoder/mask-decoder run per click.

    The precision is chosen by *file* here (the exports ship fp16/int8/q4
    variants) rather than by casting, because ONNX Runtime owns the graph.
    torch is never imported.
    """

    supports_text = False
    #: precision -> filename suffix in the export
    _SUFFIX = {"fp32": "", "fp16": "_fp16", "int8": "_int8", "q4": "_q4"}

    def _session(self, ort, stem: str):
        suffix = self._SUFFIX.get(self.precision, "")
        onnx_dir = self.checkpoint / "onnx"
        path = onnx_dir / f"{stem}{suffix}.onnx"
        if not path.exists() and suffix:            # variant not exported
            path = onnx_dir / f"{stem}.onnx"
        if not path.exists():
            raise SamBackendError(f"ONNX 파일이 없습니다: {path}")
        providers = ["CPUExecutionProvider"]
        if self.device.startswith("cuda"):
            avail = ort.get_available_providers()
            if "CUDAExecutionProvider" in avail:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ort.InferenceSession(str(path), providers=providers)

    def load(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise _need("onnxruntime", "onnxruntime") from e
        if not self.checkpoint.is_dir():
            raise SamBackendError(
                f"ONNX 모델 폴더가 없습니다: {self.checkpoint}\n"
                "모델 관리자에서 먼저 다운로드하세요.")
        self._ort = ort
        self.encoder = self._session(ort, "vision_encoder")
        self.decoder = self._session(ort, "prompt_encoder_mask_decoder")
        self.model = self.encoder            # for the generic status text
        self._size = self._input_size()
        self._embeds: Optional[dict] = None
        self._rgb: Optional[np.ndarray] = None

    def _input_size(self) -> int:
        import json
        cfg = self.checkpoint / "preprocessor_config.json"
        if cfg.exists():
            try:
                size = json.loads(cfg.read_text(encoding="utf-8")).get("size", {})
                return int(size.get("height", 1024))
            except (ValueError, OSError, TypeError):
                pass
        shape = self.encoder.get_inputs()[0].shape
        return int(shape[2]) if isinstance(shape[2], int) else 1024

    _MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    _STD = np.array([0.229, 0.224, 0.225], np.float32)

    def _encode(self, rgb: np.ndarray) -> None:
        import cv2
        self._rgb = rgb
        n = self._size
        img = cv2.resize(rgb, (n, n), interpolation=cv2.INTER_LINEAR)
        x = (img.astype(np.float32) / 255.0 - self._MEAN) / self._STD
        x = x.transpose(2, 0, 1)[None]
        names = [o.name for o in self.encoder.get_outputs()]
        outs = self.encoder.run(None, {"pixel_values": x})
        self._embeds = dict(zip(names, outs))

    def reset_image(self) -> None:
        self._rgb = None
        self._embeds = None
        self._image_key = None

    def predict(self, points=(), box=None, mask_input=None,
                multimask: bool = True) -> SamResult:
        if self._embeds is None or self._rgb is None:
            return EMPTY
        h, w = self._rgb.shape[:2]
        sx, sy = self._size / w, self._size / h      # original px -> model px
        coords, labels = self._split_points(points)

        feed = {k: v for k, v in self._embeds.items()
                if k in {i.name for i in self.decoder.get_inputs()}}
        if coords is not None:
            pts = coords.astype(np.float32) * np.array([sx, sy], np.float32)
            feed["input_points"] = pts[None, None]
            feed["input_labels"] = labels.astype(np.int64)[None, None]
        else:
            feed["input_points"] = np.zeros((1, 1, 0, 2), np.float32)
            feed["input_labels"] = np.zeros((1, 1, 0), np.int64)
        if box is not None:
            b = np.asarray(box, np.float32) * np.array([sx, sy, sx, sy], np.float32)
            feed["input_boxes"] = b.reshape(1, 1, 4)
        else:
            feed["input_boxes"] = np.zeros((1, 0, 4), np.float32)
        if coords is None and box is None:
            return EMPTY

        try:
            iou, pred, _obj = self.decoder.run(
                ["iou_scores", "pred_masks", "object_score_logits"], feed)
        except Exception as e:                       # ORT raises its own types
            raise SamBackendError(f"ONNX SAM 추론 실패: {e}") from e

        import cv2
        low = np.asarray(pred, np.float32).reshape(-1, *pred.shape[-2:])
        scores = np.asarray(iou, np.float32).reshape(-1)
        masks = np.stack([
            cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR) > 0
            for m in low]).astype(np.float32)
        if len(scores) != len(masks):
            scores = np.ones((len(masks),), np.float32)
        return self._sorted(masks, scores)


BACKEND_CLASSES = {
    "sam1": Sam1Backend,
    "sam2": Sam2Backend,
    "ultralytics": UltralyticsSamBackend,
    "sam3": Sam3Backend,
    "onnx": OnnxSamBackend,
}


def create_backend(key: str, weights_dir: Path, device: str = "cuda",
                   precision: str = "fp16") -> SamBackend:
    """Build (but do not load) the backend for a catalog key."""
    spec: Optional[ModelSpec] = CATALOG.get(key)
    if spec is None:
        raise SamBackendError(f"카탈로그에 없는 모델: {key}")
    cls = BACKEND_CLASSES.get(spec.backend)
    if cls is None:
        raise SamBackendError(f"지원하지 않는 백엔드: {spec.backend}")
    if precision not in spec.quant_modes:
        precision = spec.quant_modes[0]
    return cls(spec, spec.local_path(Path(weights_dir)), device=device,
               precision=precision)
