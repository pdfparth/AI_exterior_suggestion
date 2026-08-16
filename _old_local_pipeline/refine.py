"""MobileSAM box-prompted mask refinement.

SAM is used in box-prompt mode rather than automatic-mask-generation mode. That
matters: the automatic generator returns 30-80 unlabeled masks per facade and
happily splits one wall along shadow lines, which then needs a fragile merge
pass. Prompting it with the detector's boxes gives exactly one mask per detected
object, already labeled, and sidesteps over-segmentation entirely.

If SAM is unavailable or fails on a box, the region keeps its rectangular box
geometry. Degrading to a box costs accuracy but never breaks the pipeline.
"""

from __future__ import annotations

import cv2
import numpy as np

from config import SAM_MODEL
from regions import Region, polygon_area

_sam = None
_sam_name: str | None = None


def _load(model_name: str | None = None):
    """Load (and cache) a SAM checkpoint. Switching model reloads."""
    global _sam, _sam_name
    name = model_name or SAM_MODEL
    if _sam is None or _sam_name != name:
        from ultralytics import SAM

        print(f"[refine] loading {name} (downloads on first use)")
        _sam = SAM(name)
        _sam_name = name
    return _sam


def _mask_to_polygon(mask: np.ndarray, epsilon_frac: float = 0.004) -> list[list[float]]:
    """Largest external contour of a binary mask, simplified.

    Simplification keeps regions.json small and hand-editable - a wall traced at
    every pixel is thousands of points and unusable as correctable geometry.
    """
    m = (mask.astype(np.uint8) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    c = max(contours, key=cv2.contourArea)
    eps = epsilon_frac * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, eps, True)
    return [[float(p[0][0]), float(p[0][1])] for p in approx]


# Labels whose boundaries SAM reliably improves. Walls are excluded: they are
# large, irregular, and frequently occluded by vegetation, and SAM prompted with
# a wall-sized box tends to snap to the whole building silhouette - which is
# worse than the OneFormer mask it would replace.
REFINABLE = {"window", "door", "gate", "garage", "balcony", "pillar"}


def refine(
    image_bgr: np.ndarray,
    regions: list[Region],
    model_name: str | None = None,
    only_labels: set[str] | None = None,
) -> list[Region]:
    """Replace region geometry with SAM-derived polygons where it helps.

    Regions already carrying a semantic mask are only refined when SAM is
    likely to do better (crisp, well-bounded objects). Everything else keeps the
    geometry it arrived with.
    """
    if not regions:
        return regions

    targets = only_labels if only_labels is not None else REFINABLE
    idxs = [i for i, r in enumerate(regions) if r.label in targets]
    if not idxs:
        return regions

    try:
        sam = _load(model_name)
    except Exception as e:  # noqa: BLE001 - degrade, never crash the pipeline
        print(f"[refine] SAM unavailable ({e}); keeping existing geometry")
        for r in regions:
            r.notes.append("sam_unavailable")
        return regions

    boxes = [list(regions[i].box) for i in idxs]
    try:
        results = sam(image_bgr, bboxes=boxes, verbose=False)
    except Exception as e:  # noqa: BLE001
        print(f"[refine] SAM inference failed ({e}); keeping existing geometry")
        for i in idxs:
            regions[i].notes.append("sam_failed")
        return regions

    masks = _extract_masks(results)
    if len(masks) != len(idxs):
        print(f"[refine] mask count {len(masks)} != prompted boxes {len(idxs)}")

    for slot, i in enumerate(idxs):
        r = regions[i]
        if slot >= len(masks) or masks[slot] is None:
            r.notes.append("sam_no_mask")
            continue
        poly = _mask_to_polygon(masks[slot])
        area = polygon_area(poly)
        # A mask that collapses or balloons relative to its prompt box is a SAM
        # failure, not a better boundary. Keep what we had in that case.
        if area <= 0 or not (0.15 * r.box_area <= area <= 2.5 * r.box_area):
            r.notes.append("sam_rejected")
            continue
        r.polygon = poly
        r.px_area = area
        r.source = f"{r.source}+sam"

    return regions


def _extract_masks(results) -> list[np.ndarray | None]:
    """Ultralytics returns a Results list; masks live on .masks.data as tensors."""
    out: list[np.ndarray | None] = []
    for res in results:
        if getattr(res, "masks", None) is None or res.masks.data is None:
            continue
        for m in res.masks.data:
            out.append(m.cpu().numpy())
    return out
