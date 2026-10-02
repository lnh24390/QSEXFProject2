"""YOLOv8/9/11/12/26 segmentation and RT-DETR detection via Ultralytics."""
from __future__ import annotations

import json
from pathlib import Path

from .jobs import TrainJob
from .runner import METRIC_PREFIX


#: Where ultralytics checkpoints live. A bare asset name such as
#: "yolo11s-seg.pt" makes ultralytics download into whatever directory the
#: process happens to sit in -- which is how .pt files end up littering the
#: project root. Handing it an explicit path keeps them all in one place,
#: beside the executable in a frozen build (see settings.resolve_under_app).
def _pretrained_dir() -> Path:
    from ..core.settings import resolve_under_app
    return resolve_under_app("weights") / "yolo" / "pretrained"


def _emit(d: dict) -> None:
    print(METRIC_PREFIX + json.dumps(d, ensure_ascii=False), flush=True)


def _resolve_weights(name: str) -> str:
    """Turn a bare checkpoint name into a path under weights/yolo/pretrained/.

    Anything that already carries a directory (a resume checkpoint, a
    user-picked file) is passed through untouched. HuggingFace-style ids are
    not used by this trainer, so a plain name is always an ultralytics asset.
    """
    if not name or Path(name).is_absolute() or "/" in name or "\\" in name:
        return name
    d = _pretrained_dir()
    d.mkdir(parents=True, exist_ok=True)
    return str(d / name)


def train(job: TrainJob) -> dict:
    try:
        from ultralytics import RTDETR, YOLO
    except ImportError as e:
        raise SystemExit(
            "ultralytics 가 설치되어 있지 않습니다.\n    uv pip install ultralytics") from e

    data_yaml = Path(job.dataset_dir) / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"data.yaml 이 없습니다: {data_yaml}\n"
                         "먼저 [내보내기] 에서 YOLO 포맷으로 내보내세요.")

    weights = job.resume or _resolve_weights(job.variant)
    Cls = RTDETR if job.model_key == "rtdetr" else YOLO
    try:
        model = Cls(weights)
    except Exception as e:
        raise SystemExit(
            f"모델 '{weights}' 를 불러오지 못했습니다.\n{e}\n\n"
            "설치된 ultralytics 버전이 이 모델을 모를 수 있습니다:\n"
            "    uv pip install -U ultralytics") from e

    h = job.hyper
    epochs = int(h.get("epochs", 100))

    def on_epoch_end(trainer):
        metrics = {k.split("/")[-1].replace("(B)", "_box").replace("(M)", "_mask"): float(v)
                   for k, v in (trainer.metrics or {}).items()
                   if isinstance(v, (int, float))}
        # `loss_items` is a tensor on some ultralytics versions and a dict on
        # others; never let a shape surprise here kill the training run.
        loss = {}
        try:
            items = getattr(trainer, "loss_items", None)
            if isinstance(items, dict):
                loss = {str(k): float(v) for k, v in items.items()}
            elif items is not None:
                values = (items.tolist() if hasattr(items, "tolist")
                          else list(items))
                if not isinstance(values, list):
                    values = [values]
                names = list(getattr(trainer, "loss_names", [])) or [
                    f"loss{i}" for i in range(len(values))]
                loss = {str(n): float(v) for n, v in zip(names, values)}
        except Exception as e:                      # metrics must never fail training
            print(f"[DSL] 손실 파싱 건너뜀: {type(e).__name__}: {e}", flush=True)
        try:
            lr = float(next(iter(trainer.lr.values()), 0.0)) if getattr(
                trainer, "lr", None) else 0.0
        except Exception:
            lr = 0.0
        _emit({"epoch": int(trainer.epoch) + 1, "epochs": epochs,
               "metrics": metrics, "loss": loss, "lr": lr})

    model.add_callback("on_fit_epoch_end", on_epoch_end)

    kwargs = dict(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=int(h.get("imgsz", 640)),
        batch=int(h.get("batch", 8)),
        device=job.device,
        # ultralytics joins project/name itself, so hand it the parent of
        # run_dir rather than rebuilding the layout here.
        project=str(job.run_dir.parent.resolve()),
        name=job.run_name,
        exist_ok=True,
        workers=int(h.get("workers", 4)),
        patience=int(h.get("patience", 30)),
        optimizer=h.get("optimizer", "auto"),
        lr0=float(h.get("lr0", 0.01)),
        amp=bool(h.get("amp", True)),
        cache=h.get("cache", False),
        seed=int(h.get("seed", 0)),
        resume=bool(job.resume),
        plots=True,
    )
    for opt in ("cos_lr", "close_mosaic", "warmup_epochs", "weight_decay",
                "momentum", "dropout", "freeze", "single_cls", "rect"):
        if opt in h:
            kwargs[opt] = h[opt]

    print(f"[DSL] ultralytics 학습 시작: {weights} -> {job.run_dir}", flush=True)
    results = model.train(**kwargs)

    save_dir = Path(getattr(results, "save_dir", job.run_dir))
    best = save_dir / "weights" / "best.pt"
    summary = {"save_dir": str(save_dir), "best": str(best) if best.exists() else "",
               "model": weights}
    try:
        summary["metrics"] = {k: float(v) for k, v in
                              (getattr(results, "results_dict", {}) or {}).items()
                              if isinstance(v, (int, float))}
    except (TypeError, ValueError):
        pass
    return summary
