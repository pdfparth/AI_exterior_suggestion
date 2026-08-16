"""Region data model.

A Region is the unit of everything downstream: material assignment, area
estimation, and costing all key off this structure. It is deliberately plain
JSON-serialisable geometry rather than a baked pixel mask, so that a user
correction (5.2 "adjust or correct areas if necessary") is a data edit and does
not require re-running inference.

Walls are stored GROSS - the full wall footprint including the openings cut into
it. The deduction of windows and doors happens later, explicitly, so that the
report can show it as an auditable line item rather than folding it invisibly
into one number.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np


@dataclass
class Region:
    id: str
    label: str
    box: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    polygon: list[list[float]] = field(default_factory=list)
    px_area: float = 0.0
    source: str = "dino"  # dino | sam | heuristic
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["box"] = [round(v, 2) for v in self.box]
        d["polygon"] = [[round(x, 2), round(y, 2)] for x, y in self.polygon]
        d["px_area"] = round(self.px_area, 2)
        d["confidence"] = round(self.confidence, 4)
        return d

    @property
    def box_area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def polygon_area(points: list[list[float]]) -> float:
    """Shoelace formula. Returns 0 for degenerate polygons."""
    if len(points) < 3:
        return 0.0
    pts = np.asarray(points, dtype=float)
    x, y = pts[:, 0], pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2.0)


def box_containment(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Intersection over the *smaller* box's area.

    IoU alone misses nested duplicates: a detector often emits both a tight and
    a loose box for the same wall, which can score well under 0.6 IoU while the
    smaller box sits entirely inside the larger. Containment catches that.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    smaller = min(area_a, area_b)
    return inter / smaller if smaller > 0 else 0.0


def box_iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def save_regions(path: str, regions: list[Region], meta: dict[str, Any]) -> None:
    payload = {"meta": meta, "regions": [r.to_dict() for r in regions]}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_regions(path: str) -> tuple[list[Region], dict[str, Any]]:
    with open(path) as f:
        payload = json.load(f)
    regions = [
        Region(
            id=r["id"],
            label=r["label"],
            box=tuple(r["box"]),
            confidence=r["confidence"],
            polygon=r.get("polygon", []),
            px_area=r.get("px_area", 0.0),
            source=r.get("source", "dino"),
            notes=r.get("notes", []),
        )
        for r in payload["regions"]
    ]
    return regions, payload.get("meta", {})
