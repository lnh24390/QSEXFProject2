"""Project = image folder + class list + per-image label JSON files.

Nothing here ever loads every label into memory at once: annotations are read
on demand, and dataset statistics come from an incrementally maintained index
(labels/_index.json) instead of a full rescan.
"""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .annotation import DEFAULT_PALETTE, ClassDef, ImageAnnotation

log = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PROJECT_FILE = "project.json"
LABEL_DIR = "labels"
INDEX_FILE = "_index.json"


@dataclass
class Project:
    root: Path
    name: str = "untitled"
    image_dir: str = "images"           # relative to root, or absolute
    classes: List[ClassDef] = field(default_factory=list)
    images: List[str] = field(default_factory=list)   # relative to image root
    meta: Dict = field(default_factory=dict)
    # rel_path -> {"n": shape count, "cls": {class_id: count}, "v": verified}
    index: Dict[str, dict] = field(default_factory=dict)

    # ---- paths ------------------------------------------------------------
    @property
    def project_file(self) -> Path:
        return self.root / PROJECT_FILE

    @property
    def label_dir(self) -> Path:
        return self.root / LABEL_DIR

    @property
    def index_file(self) -> Path:
        return self.label_dir / INDEX_FILE

    @property
    def image_root(self) -> Path:
        p = Path(self.image_dir)
        return p if p.is_absolute() else (self.root / p)

    def abs_image(self, rel: str) -> Path:
        return self.image_root / rel

    def label_path(self, rel: str) -> Path:
        """Mirror the image sub-folder structure inside labels/.

        The image extension is kept, so `a/b.jpg` and `a/b.png` get their own
        label files instead of both claiming `a/b.json`.
        """
        return self.label_dir / (rel + ".json")

    def legacy_label_path(self, rel: str) -> Path:
        """Pre-0.2 layout, which dropped the image extension."""
        return (self.label_dir / rel).with_suffix(".json")

    # ---- classes ----------------------------------------------------------
    def next_class_id(self) -> int:
        return max((c.id for c in self.classes), default=-1) + 1

    def add_class(self, name: str, color: Optional[str] = None) -> ClassDef:
        cid = self.next_class_id()
        color = color or DEFAULT_PALETTE[cid % len(DEFAULT_PALETTE)]
        c = ClassDef(id=cid, name=name, color=color)
        self.classes.append(c)
        return c

    def class_by_id(self, cid: int) -> Optional[ClassDef]:
        for c in self.classes:
            if c.id == cid:
                return c
        return None

    def class_by_name(self, name: str) -> Optional[ClassDef]:
        for c in self.classes:
            if c.name == name:
                return c
        return None

    def remove_class(self, cid: int) -> int:
        """Delete a class and every shape referencing it. Returns shapes removed."""
        self.classes = [c for c in self.classes if c.id != cid]
        removed = 0
        for rel in list(self.index.keys()):
            if cid not in {int(k) for k in self.index[rel].get("cls", {})}:
                continue
            ann = self.load_annotation(rel)
            keep = [s for s in ann.shapes if s.class_id != cid]
            if len(keep) == len(ann.shapes):
                continue
            removed += len(ann.shapes) - len(keep)
            ann.shapes = keep
            self.save_annotation(ann)
        self.save_index()          # the index just changed on disk-backed data
        return removed

    def contiguous_index(self) -> Dict[int, int]:
        """class_id -> 0..N-1 in list order (what YOLO/COCO exporters need)."""
        return {c.id: i for i, c in enumerate(self.classes)}

    # ---- images -----------------------------------------------------------
    def scan_images(self, recursive: bool = True) -> List[str]:
        """Collect relative paths only. No pixel data is touched here."""
        root = self.image_root
        if not root.exists():
            self.images = []
            return self.images
        it = root.rglob("*") if recursive else root.glob("*")
        self.images = sorted(
            p.relative_to(root).as_posix()
            for p in it
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        return self.images

    def import_images(self, paths, copy: bool = True) -> List[str]:
        """Bring external files into the project image folder."""
        dst_root = self.image_root
        dst_root.mkdir(parents=True, exist_ok=True)
        added = []
        for p in paths:
            src = Path(p)
            if not src.is_file() or src.suffix.lower() not in IMAGE_EXTS:
                continue
            dst = dst_root / src.name
            n = 1
            while dst.exists() and dst.resolve() != src.resolve():
                dst = dst_root / (src.stem + "_" + str(n) + src.suffix)
                n += 1
            if dst.resolve() != src.resolve():
                if copy:
                    shutil.copy2(src, dst)
                else:
                    shutil.move(str(src), dst)
            rel = dst.relative_to(dst_root).as_posix()
            if rel not in self.images:
                self.images.append(rel)
            added.append(rel)
        self.images.sort()
        return added

    # ---- annotations ------------------------------------------------------
    def load_annotation(self, rel: str) -> ImageAnnotation:
        for lp in (self.label_path(rel), self.legacy_label_path(rel)):
            if not lp.exists():
                continue
            try:
                ann = ImageAnnotation.load(lp)
                ann.image_path = rel
                return ann
            except (json.JSONDecodeError, OSError, KeyError, ValueError) as e:
                log.warning("라벨 파일을 읽지 못했습니다: %s (%s)", lp, e)
        return ImageAnnotation(image_path=rel)

    def save_annotation(self, ann: ImageAnnotation) -> Path:
        """Write one label file and update this image's index entry."""
        rel = ann.image_path
        lp = self.label_path(rel)
        legacy = self.legacy_label_path(rel)
        if ann.shapes or ann.verified:
            ann.save(lp)
            if legacy != lp and legacy.exists():
                legacy.unlink()          # migrated to the extension-safe name
            per: Dict[str, int] = {}
            for s in ann.shapes:
                per[str(s.class_id)] = per.get(str(s.class_id), 0) + 1
            self.index[rel] = {"n": len(ann.shapes), "cls": per,
                               "v": bool(ann.verified)}
        else:
            for p in (lp, legacy):
                if p.exists():
                    p.unlink()
            self.index.pop(rel, None)
        return lp

    def is_labeled(self, rel: str) -> bool:
        entry = self.index.get(rel)
        return bool(entry and entry.get("n", 0) > 0)

    def is_verified(self, rel: str) -> bool:
        entry = self.index.get(rel)
        return bool(entry and entry.get("v"))

    def shape_count(self, rel: str) -> int:
        entry = self.index.get(rel)
        return int(entry.get("n", 0)) if entry else 0

    def labeled_images(self) -> List[str]:
        return [r for r in self.images if self.is_labeled(r)]

    def unlabeled_images(self) -> List[str]:
        return [r for r in self.images if not self.is_labeled(r)]

    def next_unlabeled(self, after: str) -> Optional[str]:
        try:
            start = self.images.index(after) + 1
        except ValueError:
            start = 0
        for rel in self.images[start:]:
            if not self.is_labeled(rel):
                return rel
        for rel in self.images[:start]:      # wrap around
            if not self.is_labeled(rel):
                return rel
        return None

    # ---- index ------------------------------------------------------------
    def rebuild_index(self, progress=None) -> Dict[str, dict]:
        """Full rescan of labels/. Only one JSON is held in memory at a time."""
        idx: Dict[str, dict] = {}
        total = len(self.images)
        for i, rel in enumerate(self.images):
            lp = self.label_path(rel)
            if not lp.exists():
                lp = self.legacy_label_path(rel)
            if lp.exists():
                try:
                    d = json.loads(lp.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError) as e:
                    log.warning("인덱스 재구축 중 건너뜀: %s (%s)", lp, e)
                    continue
                per: Dict[str, int] = {}
                for s in d.get("shapes", []):
                    k = str(s.get("class_id", 0))
                    per[k] = per.get(k, 0) + 1
                idx[rel] = {"n": len(d.get("shapes", [])), "cls": per,
                            "v": bool(d.get("verified", False))}
            if progress and (i % 200 == 0 or i == total - 1):
                progress(i + 1, total)
        self.index = idx
        return idx

    def save_index(self) -> None:
        self.label_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.index_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.index, ensure_ascii=False),
                       encoding="utf-8")
        tmp.replace(self.index_file)

    def load_index(self) -> None:
        if self.index_file.exists():
            try:
                self.index = json.loads(self.index_file.read_text(encoding="utf-8"))
                return
            except (json.JSONDecodeError, OSError):
                pass
        self.rebuild_index()

    def stats(self) -> Dict:
        """Aggregate from the in-memory index -- no disk scan."""
        per_class: Dict[int, int] = {c.id: 0 for c in self.classes}
        shapes = 0
        labeled = 0
        verified = 0
        for _rel, e in self.index.items():
            if e.get("n", 0) > 0:
                labeled += 1
            if e.get("v"):
                verified += 1
            shapes += int(e.get("n", 0))
            for k, v in e.get("cls", {}).items():
                per_class[int(k)] = per_class.get(int(k), 0) + int(v)
        return {"images": len(self.images), "labeled": labeled,
                "verified": verified, "shapes": shapes, "per_class": per_class}

    # ---- persistence ------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "version": 1,
            "name": self.name,
            "image_dir": self.image_dir,
            "classes": [c.to_dict() for c in self.classes],
            "images": self.images,
            "meta": self.meta,
        }

    def save(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        self.label_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.project_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(self.project_file)
        self.save_index()
        return self.project_file

    @staticmethod
    def load(path: Path) -> "Project":
        path = Path(path)
        if path.is_dir():
            path = path / PROJECT_FILE
        d = json.loads(path.read_text(encoding="utf-8"))
        proj = Project(
            root=path.parent,
            name=d.get("name", path.parent.name),
            image_dir=d.get("image_dir", "images"),
            classes=[ClassDef.from_dict(c) for c in d.get("classes", [])],
            images=list(d.get("images", [])),
            meta=d.get("meta", {}),
        )
        proj.load_index()
        return proj

    @staticmethod
    def create(root: Path, name: str = "", image_dir: str = "images",
               classes: Optional[List[str]] = None) -> "Project":
        root = Path(root)
        proj = Project(root=root, name=name or root.name, image_dir=image_dir)
        for cn in (classes or []):
            proj.add_class(cn)
        (root / LABEL_DIR).mkdir(parents=True, exist_ok=True)
        proj.image_root.mkdir(parents=True, exist_ok=True)
        proj.scan_images()
        proj.rebuild_index()
        proj.save()
        return proj
