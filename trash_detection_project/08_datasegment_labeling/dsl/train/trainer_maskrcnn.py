"""Mask R-CNN fine-tuning on the exported COCO dataset (torchvision)."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from .jobs import TrainJob
from .runner import METRIC_PREFIX


def _emit(d: dict) -> None:
    print(METRIC_PREFIX + json.dumps(d, ensure_ascii=False), flush=True)


def seg_to_mask(seg, height: int, width: int) -> np.ndarray:
    """Rasterise any COCO `segmentation` into a uint8 {0,1} mask.

    Handles the three shapes COCO allows: a list of polygon rings (union), an
    uncompressed RLE (`counts` as a list of ints, what our exporter writes for
    shapes with holes) and a compressed RLE (`counts` as a string).
    """
    import cv2

    m = np.zeros((height, width), np.uint8)
    if isinstance(seg, dict):
        counts = seg.get("counts")
        h, w = seg.get("size", [height, width])
        if isinstance(counts, (str, bytes)):        # compressed RLE
            try:
                from pycocotools import mask as _M
            except ImportError as e:
                raise RuntimeError(
                    "압축 RLE 세그먼트를 읽으려면 pycocotools 가 필요합니다.\n"
                    "    uv pip install pycocotools") from e
            c = counts.encode() if isinstance(counts, str) else counts
            return _M.decode({"counts": c, "size": [h, w]}).astype(np.uint8)
        if not counts:
            return m
        flat = np.zeros(int(h) * int(w), np.uint8)
        pos, val = 0, 0
        for run in counts:
            run = int(run)
            if val:
                flat[pos:pos + run] = 1
            pos += run
            val ^= 1
        return flat.reshape((int(h), int(w)), order="F")

    for ring in seg or []:
        if len(ring) >= 6:
            pts = np.array(ring, np.float32).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(m, [pts], 1)
    return m


class CocoInstanceDataset:
    """Minimal COCO reader -- avoids a pycocotools dependency for training."""

    def __init__(self, root: Path, split: str, imgsz: int = 800, train: bool = True):
        import torch
        self.torch = torch
        self.root = Path(root)
        self.split = split
        self.imgsz = imgsz
        self.train = train
        ann_file = self.root / "annotations" / f"instances_{split}.json"
        if not ann_file.exists():
            raise SystemExit(f"COCO 주석 파일이 없습니다: {ann_file}")
        doc = json.loads(ann_file.read_text(encoding="utf-8"))
        self.images = {im["id"]: im for im in doc["images"]}
        self.cats = sorted({c["id"] for c in doc["categories"]})
        self.cat_to_idx = {c: i + 1 for i, c in enumerate(self.cats)}   # 0 = bg
        self.names = [c["name"] for c in sorted(doc["categories"],
                                                key=lambda c: c["id"])]
        self.by_image: Dict[int, List[dict]] = {}
        for a in doc["annotations"]:
            self.by_image.setdefault(a["image_id"], []).append(a)
        self.ids = [i for i in self.images if self.by_image.get(i)]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i):
        import cv2
        import torch
        img_id = self.ids[i]
        info = self.images[img_id]
        path = self.root / "images" / self.split / info["file_name"]
        buf = np.fromfile(str(path), dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"이미지를 읽지 못했습니다: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]

        scale = min(self.imgsz / max(H, W), 1.0)
        if scale < 1.0:
            rgb = cv2.resize(rgb, (int(W * scale), int(H * scale)),
                             interpolation=cv2.INTER_AREA)
        nh, nw = rgb.shape[:2]

        masks, boxes, labels = [], [], []
        for a in self.by_image[img_id]:
            m = seg_to_mask(a.get("segmentation"), H, W)
            if scale < 1.0:
                m = cv2.resize(m, (nw, nh), interpolation=cv2.INTER_NEAREST)
            ys, xs = np.where(m > 0)
            if xs.size == 0:
                continue
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            if x2 <= x1 or y2 <= y1:
                continue
            masks.append(m)
            boxes.append([float(x1), float(y1), float(x2), float(y2)])
            labels.append(self.cat_to_idx[a["category_id"]])

        if self.train and np.random.rand() < 0.5:          # horizontal flip
            rgb = rgb[:, ::-1].copy()
            masks = [m[:, ::-1].copy() for m in masks]
            boxes = [[nw - b[2], b[1], nw - b[0], b[3]] for b in boxes]

        img = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
        # torchvision accepts an empty target (verified: it returns real
        # losses), so an image whose annotations all degenerate is trained on
        # as a negative sample instead of being given a fabricated instance.
        if masks:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)
            masks_t = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            masks_t = torch.zeros((0, nh, nw), dtype=torch.uint8)
        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "masks": masks_t,
            "image_id": torch.tensor([img_id]),
        }
        return img, target


def _collate(batch):
    return tuple(zip(*batch))


def train(job: TrainJob) -> dict:
    try:
        import torch
        import torchvision
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    except ImportError as e:
        raise SystemExit("torchvision 이 필요합니다.  uv pip install torchvision") from e

    h = job.hyper
    root = Path(job.dataset_dir)
    imgsz = int(h.get("imgsz", 800))
    train_ds = CocoInstanceDataset(root, "train", imgsz, train=True)
    try:
        val_ds = CocoInstanceDataset(root, "val", imgsz, train=False)
    except SystemExit:
        val_ds = None

    n_classes = len(train_ds.cats) + 1
    print(f"[DSL] Mask R-CNN: train {len(train_ds)}장, "
          f"val {len(val_ds) if val_ds else 0}장, 클래스 {n_classes - 1}개", flush=True)

    builder = getattr(torchvision.models.detection, job.variant,
                      torchvision.models.detection.maskrcnn_resnet50_fpn_v2)
    model = builder(weights="DEFAULT",
                    trainable_backbone_layers=int(h.get("trainable_layers", 3)))
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, n_classes)
    in_m = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_m, 256, n_classes)

    device = torch.device("cuda" if (job.device not in ("cpu", "")
                                     and torch.cuda.is_available()) else "cpu")
    model.to(device)

    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=int(h.get("batch", 2)), shuffle=True,
        num_workers=int(h.get("workers", 2)), collate_fn=_collate,
        pin_memory=(device.type == "cuda"), persistent_workers=False)
    val_loader = None
    if val_ds is not None and len(val_ds):
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=0,
            collate_fn=_collate)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=float(h.get("lr0", 0.005)),
                          momentum=float(h.get("momentum", 0.9)),
                          weight_decay=float(h.get("weight_decay", 0.0005)))
    epochs = int(h.get("epochs", 30))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    use_amp = bool(h.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_dir = job.run_dir
    (out_dir / "weights").mkdir(parents=True, exist_ok=True)
    best_loss = math.inf
    history = []

    for ep in range(epochs):
        model.train()
        total = 0.0
        t0 = time.time()
        for it, (imgs, targets) in enumerate(loader):
            imgs = [i.to(device, non_blocking=True) for i in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.amp.autocast("cuda", enabled=use_amp):
                losses = model(imgs, targets)
                loss = sum(losses.values())
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

        val_loss = float("nan")
        if val_loader is not None:
            model.train()                     # torchvision returns losses in train mode
            # ...but train mode also keeps BatchNorm updating its running stats,
            # and maskrcnn_*_v2 uses live BatchNorm2d (not FrozenBatchNorm2d),
            # so validating would silently drift the statistics every epoch.
            for m in model.modules():
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    m.eval()
            with torch.no_grad():
                vt = 0.0
                for imgs, targets in val_loader:
                    imgs = [i.to(device) for i in imgs]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        vt += float(sum(model(imgs, targets).values()))
                val_loss = vt / max(1, len(val_loader))

        rec = {"epoch": ep + 1, "epochs": epochs,
               "loss": {"train": round(train_loss, 4),
                        "val": None if val_loss != val_loss else round(val_loss, 4)},
               "lr": sched.get_last_lr()[0],
               "time": round(time.time() - t0, 1)}
        history.append(rec)
        _emit(rec)

        torch.save({"model": model.state_dict(), "epoch": ep + 1,
                    "classes": train_ds.names},
                   out_dir / "weights" / "last.pt")
        score = val_loss if val_loss == val_loss else train_loss
        if score < best_loss:
            best_loss = score
            torch.save({"model": model.state_dict(), "epoch": ep + 1,
                        "classes": train_ds.names},
                       out_dir / "weights" / "best.pt")

    (out_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"save_dir": str(out_dir),
            "best": str(out_dir / "weights" / "best.pt"),
            "best_loss": round(best_loss, 4)}
