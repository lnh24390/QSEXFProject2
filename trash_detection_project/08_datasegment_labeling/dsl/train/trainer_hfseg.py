"""SegFormer / ConvNeXt-UPerNet semantic segmentation (HuggingFace)."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import List

import numpy as np

from .jobs import TrainJob
from .runner import METRIC_PREFIX


def _emit(d: dict) -> None:
    print(METRIC_PREFIX + json.dumps(d, ensure_ascii=False), flush=True)


class MaskSegDataset:
    """images/<split>/x.jpg  +  masks/<split>/x.png (index map, 0 = background)."""

    def __init__(self, root: Path, split: str, imgsz: int = 512, train: bool = True):
        self.root = Path(root)
        self.split = split
        self.imgsz = int(imgsz)
        self.train = train
        img_dir = self.root / "images" / split
        self.items: List[Path] = sorted(p for p in img_dir.glob("*")
                                        if p.is_file())
        if not self.items:
            raise SystemExit(f"이미지가 없습니다: {img_dir}")
        self.mask_dir = self.root / "masks" / split

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        import cv2
        import torch
        p = self.items[i]
        bgr = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"이미지를 읽지 못했습니다: {p}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp = self.mask_dir / (p.stem + ".png")
        mask = cv2.imdecode(np.fromfile(str(mp), np.uint8), cv2.IMREAD_UNCHANGED) \
            if mp.exists() else np.zeros(rgb.shape[:2], np.uint8)
        if mask.ndim == 3:
            mask = mask[..., 0]

        s = self.imgsz
        rgb = cv2.resize(rgb, (s, s), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (s, s), interpolation=cv2.INTER_NEAREST)

        if self.train:
            if np.random.rand() < 0.5:
                rgb, mask = rgb[:, ::-1].copy(), mask[:, ::-1].copy()
            if np.random.rand() < 0.3:              # brightness jitter
                f = float(np.random.uniform(0.8, 1.2))
                rgb = np.clip(rgb.astype(np.float32) * f, 0, 255).astype(np.uint8)

        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std = np.array([0.229, 0.224, 0.225], np.float32)
        x = (rgb.astype(np.float32) / 255.0 - mean) / std
        return (torch.from_numpy(x.transpose(2, 0, 1)),
                torch.from_numpy(mask.astype(np.int64)))


def _miou(conf: np.ndarray) -> float:
    inter = np.diag(conf).astype(np.float64)
    union = conf.sum(0) + conf.sum(1) - inter
    valid = union > 0
    return float((inter[valid] / union[valid]).mean()) if valid.any() else 0.0


def train(job: TrainJob) -> dict:
    try:
        import torch
        from transformers import (AutoConfig, SegformerForSemanticSegmentation,
                                  UperNetForSemanticSegmentation)
    except ImportError as e:
        raise SystemExit("transformers 가 필요합니다.  uv pip install transformers") from e

    h = job.hyper
    root = Path(job.dataset_dir)
    meta_file = root / "dataset.json"
    if meta_file.exists():
        names = json.loads(meta_file.read_text(encoding="utf-8"))["classes"]
    else:
        names = ["background"] + list(job.classes)
    n_classes = len(names)

    imgsz = int(h.get("imgsz", 512))
    train_ds = MaskSegDataset(root, "train", imgsz, True)
    try:
        val_ds = MaskSegDataset(root, "val", imgsz, False)
    except SystemExit:
        val_ds = None
    print(f"[DSL] {job.model_key}: train {len(train_ds)}장, "
          f"val {len(val_ds) if val_ds else 0}장, 클래스 {n_classes}개", flush=True)

    id2label = {i: n for i, n in enumerate(names)}
    label2id = {n: i for i, n in enumerate(names)}
    Model = (UperNetForSemanticSegmentation if job.model_key == "convnext-upernet"
             else SegformerForSemanticSegmentation)
    try:
        model = Model.from_pretrained(
            job.variant, num_labels=n_classes, id2label=id2label,
            label2id=label2id, ignore_mismatched_sizes=True)
    except Exception as e:
        raise SystemExit(f"사전학습 모델을 불러오지 못했습니다: {job.variant}\n{e}\n"
                         "인터넷 연결 또는 모델 ID를 확인하세요.") from e

    device = torch.device("cuda" if (job.device not in ("cpu", "")
                                     and torch.cuda.is_available()) else "cpu")
    model.to(device)

    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=int(h.get("batch", 4)), shuffle=True,
        num_workers=int(h.get("workers", 2)), pin_memory=(device.type == "cuda"))
    val_loader = (torch.utils.data.DataLoader(val_ds, batch_size=1, num_workers=0)
                  if val_ds is not None and len(val_ds) else None)

    epochs = int(h.get("epochs", 50))
    opt = torch.optim.AdamW(model.parameters(), lr=float(h.get("lr0", 6e-5)),
                            weight_decay=float(h.get("weight_decay", 0.01)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    use_amp = bool(h.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_dir = job.run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    # -inf, not -1.0: without a validation split the score is `-train_loss`,
    # so any run whose loss stays above 1.0 would never write `best/` while
    # the returned summary still pointed at it.
    best_miou = float("-inf")
    history = []

    for ep in range(epochs):
        model.train()
        total = 0.0
        t0 = time.time()
        for it, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(pixel_values=x, labels=y)
                loss = out.loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += float(loss.detach())
            if it % 20 == 0:
                print(f"  epoch {ep + 1}/{epochs}  iter {it}/{len(loader)}  "
                      f"loss {float(loss):.4f}", flush=True)
        sched.step()
        train_loss = total / max(1, len(loader))

        miou = float("nan")
        if val_loader is not None:
            model.eval()
            conf = np.zeros((n_classes, n_classes), np.int64)
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        logits = model(pixel_values=x).logits
                    logits = torch.nn.functional.interpolate(
                        logits.float(), size=y.shape[-2:], mode="bilinear",
                        align_corners=False)
                    pred = logits.argmax(1).cpu().numpy().reshape(-1)
                    gt = y.numpy().reshape(-1)
                    k = (gt >= 0) & (gt < n_classes)
                    conf += np.bincount(
                        n_classes * gt[k].astype(int) + pred[k],
                        minlength=n_classes ** 2).reshape(n_classes, n_classes)
            miou = _miou(conf)

        rec = {"epoch": ep + 1, "epochs": epochs,
               "loss": {"train": round(train_loss, 4)},
               "metrics": {} if miou != miou else {"mIoU": round(miou, 4)},
               "lr": sched.get_last_lr()[0], "time": round(time.time() - t0, 1)}
        history.append(rec)
        _emit(rec)

        model.save_pretrained(out_dir / "last")
        score = miou if miou == miou else -train_loss
        if score > best_miou:
            best_miou = score
            model.save_pretrained(out_dir / "best")
            (out_dir / "best" / "classes.json").write_text(
                json.dumps(names, ensure_ascii=False), encoding="utf-8")

    (out_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"save_dir": str(out_dir), "best": str(out_dir / "best"),
            "best_score": round(best_miou, 4)}
