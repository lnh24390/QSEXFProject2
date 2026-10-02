"""Annotation data model.

Everything on disk is polygon-based (portable, human-readable JSON).
Binary masks are only an in-memory intermediate produced by SAM.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

Point = Tuple[float, float]

# Distinct, colour-blind-friendly-ish default palette for new classes.
DEFAULT_PALETTE: List[str] = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9a6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#ffe119",
]


@dataclass
class ClassDef:
    """A label class. `id` is the stable index used by every exporter."""
    id: int
    name: str
    color: str = "#e6194b"

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "ClassDef":
        return ClassDef(id=int(d["id"]), name=str(d["name"]),
                        color=str(d.get("color", "#e6194b")))


@dataclass
class Shape:
    """One instance: an outer polygon plus optional holes."""
    class_id: int
    points: List[Point] = field(default_factory=list)
    holes: List[List[Point]] = field(default_factory=list)
    shape_type: str = "polygon"          # polygon | rectangle
    score: float = 1.0                   # SAM confidence, 1.0 for manual
    source: str = "manual"               # manual | sam1 | sam2 | sam3 | sam3.1
    group_id: Optional[int] = None       # link parts of one object
    meta: dict = field(default_factory=dict)  # e.g. {"src_bbox": [...]}
    uid: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    # ---- geometry helpers -------------------------------------------------
    def scaled(self, sx: float, sy: float) -> "Shape":
        """A copy with every coordinate multiplied.

        Used when an export resizes the images: the label has to move
        with the pixels or it describes the original resolution.
        Returns a copy so the project's own annotation is untouched.
        """
        return replace(
            self,
            points=[(x * sx, y * sy) for x, y in self.points],
            holes=[[(x * sx, y * sy) for x, y in h] for h in self.holes],
            meta=dict(self.meta or {}),
        )

    def bbox(self) -> Tuple[float, float, float, float]:
        """(x1, y1, x2, y2)"""
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        if not xs:
            return (0.0, 0.0, 0.0, 0.0)
        return (min(xs), min(ys), max(xs), max(ys))

    def area(self) -> float:
        """Shoelace area of the outer ring minus holes."""
        def ring_area(pts: List[Point]) -> float:
            n = len(pts)
            if n < 3:
                return 0.0
            s = 0.0
            for i in range(n):
                x1, y1 = pts[i]
                x2, y2 = pts[(i + 1) % n]
                s += x1 * y2 - x2 * y1
            return abs(s) * 0.5
        return ring_area(self.points) - sum(ring_area(h) for h in self.holes)

    def is_valid(self) -> bool:
        return len(self.points) >= 3

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "points": [[round(float(x), 2), round(float(y), 2)] for x, y in self.points],
            "holes": [[[round(float(x), 2), round(float(y), 2)] for x, y in h] for h in self.holes],
            "shape_type": self.shape_type,
            "score": round(float(self.score), 4),
            "source": self.source,
            "group_id": self.group_id,
            "meta": self.meta,
            "uid": self.uid,
        }

    @staticmethod
    def from_dict(d: dict) -> "Shape":
        return Shape(
            class_id=int(d.get("class_id", 0)),
            points=[(float(p[0]), float(p[1])) for p in d.get("points", [])],
            holes=[[(float(p[0]), float(p[1])) for p in h] for h in d.get("holes", [])],
            shape_type=d.get("shape_type", "polygon"),
            score=float(d.get("score", 1.0)),
            source=d.get("source", "manual"),
            group_id=d.get("group_id"),
            meta=dict(d.get("meta") or {}),
            uid=d.get("uid") or uuid.uuid4().hex[:12],
        )


@dataclass
class ImageAnnotation:
    """All shapes for a single image."""
    image_path: str                      # relative to project root
    width: int = 0
    height: int = 0
    shapes: List[Shape] = field(default_factory=list)
    verified: bool = False               # human-reviewed flag

    def scaled(self, sx: float, sy: float) -> "ImageAnnotation":
        """A copy at a different resolution, shapes and size together."""
        return ImageAnnotation(
            image_path=self.image_path,
            width=max(1, int(round(self.width * sx))) if self.width else 0,
            height=max(1, int(round(self.height * sy))) if self.height else 0,
            shapes=[sh.scaled(sx, sy) for sh in self.shapes],
            verified=self.verified,
        )

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height,
            "verified": self.verified,
            "shapes": [s.to_dict() for s in self.shapes],
        }

    @staticmethod
    def from_dict(d: dict) -> "ImageAnnotation":
        return ImageAnnotation(
            image_path=d.get("image_path", ""),
            width=int(d.get("width", 0)),
            height=int(d.get("height", 0)),
            verified=bool(d.get("verified", False)),
            shapes=[Shape.from_dict(s) for s in d.get("shapes", [])],
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1),
                       encoding="utf-8")
        tmp.replace(path)

    @staticmethod
    def load(path: Path) -> "ImageAnnotation":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return ImageAnnotation.from_dict(data)

    def classes_used(self) -> set:
        return {s.class_id for s in self.shapes}
