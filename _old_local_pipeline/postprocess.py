"""Clean raw detections into a usable region set.

Raw detector output is not directly usable: coordinates land out of bounds,
the same window gets detected three times, and `parapet` is not reliably
reachable by text prompt at all. This module handles those, and flags the
sanity checks that tell you the segmentation went wrong before the numbers
propagate into a cost estimate.
"""

from __future__ import annotations

import numpy as np

from config import MERGE_CONTAINMENT_THRESHOLD, MERGE_IOU_THRESHOLD, MIN_AREA_FRACTION
from regions import Region, box_containment, box_iou, polygon_area


def clamp(regions: list[Region], w: int, h: int) -> list[Region]:
    """Clip every coordinate into the image. Models do emit out-of-bounds boxes."""
    for r in regions:
        x1, y1, x2, y2 = r.box
        x1, x2 = sorted((max(0.0, min(x1, w)), max(0.0, min(x2, w))))
        y1, y2 = sorted((max(0.0, min(y1, h)), max(0.0, min(y2, h))))
        r.box = (x1, y1, x2, y2)
        if r.polygon:
            r.polygon = [
                [max(0.0, min(x, w)), max(0.0, min(y, h))] for x, y in r.polygon
            ]
    return regions


def drop_degenerate(regions: list[Region], w: int, h: int) -> list[Region]:
    """Discard slivers and noise-sized detections."""
    floor = MIN_AREA_FRACTION * w * h
    kept = []
    for r in regions:
        area = r.px_area if r.px_area > 0 else r.box_area
        if area >= floor and r.box_area > 0:
            kept.append(r)
    return kept


def _source_rank(r: Region) -> int:
    """Prefer pixel-accurate geometry when two sources describe one object.

    OneFormer+SAM beats OneFormer beats a raw detector box beats a heuristic.
    This ordering decides which region survives a merge, so it directly controls
    the area that reaches the cost engine.
    """
    if "sam" in r.source:
        return 3
    if r.source == "oneformer":
        return 2
    if r.source == "dino":
        return 1
    return 0


def merge_duplicates(regions: list[Region]) -> list[Region]:
    """Collapse boxes that describe the same object.

    Two overlap tests, because they catch different failures. IoU catches
    near-identical boxes ('window' and 'window frame' firing on one window).
    Containment catches nested boxes - a tight wall box inside a loose one
    scores only ~0.55 IoU but is plainly the same wall. Without the second test
    the duplicate survives and inflates area, which propagates straight into the
    cost estimate.

    Regions are considered in source-quality order rather than confidence order,
    so a precise OneFormer mask is kept over a coarse detector box covering the
    same object - the detector's higher score says nothing about its geometry.
    """
    out: list[Region] = []
    for r in sorted(regions, key=lambda x: (_source_rank(x), x.confidence), reverse=True):
        dup = next(
            (
                k
                for k in out
                if k.label == r.label
                and (
                    box_iou(k.box, r.box) >= MERGE_IOU_THRESHOLD
                    or box_containment(k.box, r.box) >= MERGE_CONTAINMENT_THRESHOLD
                )
            ),
            None,
        )
        if dup is None:
            out.append(r)
        else:
            dup.notes.append(f"merged={r.id}({r.source})")
    return out


def infer_parapet(regions: list[Region], w: int, h: int) -> list[Region]:
    """Recover the parapet geometrically.

    GroundingDINO does not reliably hit 'parapet' - the term is construction
    jargon and poorly represented in its training captions. The parapet is
    however geometrically predictable: a thin horizontal band spanning the
    building's width, sitting above the topmost opening. We synthesise it from
    the wall extent when the detector found no explicit roof-edge region.
    """
    if any(r.label in ("parapet", "roof_edge") for r in regions):
        return regions

    walls = [r for r in regions if r.label == "wall"]
    if not walls:
        return regions

    # Anchor to the largest wall only. Using the union of every wall surface
    # spans disjoint parts of the facade and produces a band across the sky.
    main = max(walls, key=lambda r: r.px_area)
    x1, top, x2 = main.box[0], main.box[1], main.box[2]

    wall_height = main.box[3] - top
    openings = [r for r in regions if r.label in ("window", "door", "balcony")]
    # Only openings that actually sit within this wall's span - an opening on a
    # different part of the facade says nothing about where this parapet ends.
    above = [
        r for r in openings if r.box[0] < x2 and r.box[2] > x1 and r.box[1] >= top
    ]
    if above:
        highest = min(r.box[1] for r in above)
        band = max(0.0, highest - top)
    else:
        band = wall_height * 0.12

    # A parapet is a thin band, not a storey. If the gap above the topmost
    # opening is a large fraction of the wall, that gap is wall - the detector
    # simply found no opening up there - and calling it parapet would hand the
    # cost engine a fabricated several-hundred-square-foot surface.
    if wall_height > 0 and band > 0.25 * wall_height:
        band = wall_height * 0.12

    # Too thin to be a real parapet band; likely a tight wall crop.
    if band < 0.02 * h or (x2 - x1) < 0.2 * w:
        return regions

    regions.append(
        Region(
            id="p0",
            label="parapet",
            box=(x1, top, x2, top + band),
            confidence=0.35,
            px_area=(x2 - x1) * band,
            source="heuristic",
            notes=["inferred geometrically; not detected by prompt"],
        )
    )
    return regions


def compute_areas(regions: list[Region]) -> list[Region]:
    for r in regions:
        if r.polygon:
            r.px_area = polygon_area(r.polygon)
        if r.px_area <= 0:
            r.px_area = r.box_area
    return regions


def renumber(regions: list[Region]) -> list[Region]:
    for i, r in enumerate(regions):
        r.id = f"r{i}"
    return regions


def sanity_checks(regions: list[Region], w: int, h: int) -> list[str]:
    """Cheap correctness signals. These are warnings for the report, not errors.

    Segmentation failures are quiet - you get plausible-looking JSON that
    describes the wrong building. These checks surface the common ones.
    """
    warnings: list[str] = []
    img_area = float(w * h)
    by_label: dict[str, float] = {}
    for r in regions:
        by_label[r.label] = by_label.get(r.label, 0.0) + r.px_area

    total = sum(by_label.values())
    coverage = total / img_area if img_area else 0.0
    if coverage < 0.25:
        warnings.append(
            f"low coverage ({coverage:.0%}) - building may be small in frame "
            "or detection largely failed"
        )
    if coverage > 1.6:
        warnings.append(
            f"coverage {coverage:.0%} implies heavy overlap - areas may be double-counted"
        )

    wall = by_label.get("wall", 0.0)
    if wall == 0:
        warnings.append("no wall detected - area estimation cannot proceed")

    openings = sum(by_label.get(k, 0.0) for k in ("window", "door", "balcony"))
    if wall > 0 and openings > 0.40 * wall:
        warnings.append(
            f"openings are {openings / wall:.0%} of wall area - unusually high, "
            "check for duplicate window detections"
        )

    if not any(r.label == "door" for r in regions):
        warnings.append(
            "no door detected - scale reference for area estimation must fall back "
            "to window or user input"
        )

    return warnings


def run(regions: list[Region], w: int, h: int) -> tuple[list[Region], list[str]]:
    regions = clamp(regions, w, h)
    regions = compute_areas(regions)
    regions = drop_degenerate(regions, w, h)
    regions = merge_duplicates(regions)
    regions = infer_parapet(regions, w, h)
    regions = compute_areas(regions)
    regions = sorted(regions, key=lambda r: r.px_area, reverse=True)
    regions = renumber(regions)
    return regions, sanity_checks(regions, w, h)
