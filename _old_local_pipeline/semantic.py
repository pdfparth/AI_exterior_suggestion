"""OneFormer ADE20K semantic segmentation -> facade regions.

This is the component that actually solves "detect all walls". Walls are not
objects: an open-vocabulary detector prompted with "wall" returns either a box
around the entire frame or nothing at all, because there is no consistent
wall-shaped thing to bound. Per-pixel semantic labelling has no such problem,
and it covers the whole image rather than only what a detector happened to fire
on.

Output is connected components, not raw class masks: two separate windows must
be two Regions so openings can be deducted individually and materials assigned
per surface.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import torch

from ade20k import resolve
from config import MIN_COMPONENT_PX, ONEFORMER_MODEL
from regions import Region

_model = None
_processor = None
_maps: tuple[dict[int, str], set[int]] | None = None


def _load():
    global _model, _processor, _maps
    if _model is None:
        from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor

        print(f"[semantic] loading {ONEFORMER_MODEL} (first run downloads ~1.5GB)")
        _processor = OneFormerProcessor.from_pretrained(ONEFORMER_MODEL)
        _model = OneFormerForUniversalSegmentation.from_pretrained(
            ONEFORMER_MODEL
        ).eval()
        torch.set_num_threads(os.cpu_count() or 4)

        id2label = _model.config.id2label
        _maps = resolve(id2label)
        mapped, background = _maps
        print(
            f"[semantic] resolved {len(mapped)} facade classes, "
            f"{len(background)} background classes from {len(id2label)} ADE20K labels"
        )
    return _model, _processor, _maps


def segment(image_bgr: np.ndarray) -> tuple[list[Region], np.ndarray]:
    """Semantic segmentation -> Regions.

    Only classes that ADE20K genuinely resolves on a facade are emitted as
    Regions - in practice `stairs`, and occasionally `door`. Walls are NOT
    emitted here: see `building_envelope` for why, and for what replaces them.

    Returns (regions, semantic_map) where semantic_map is the raw per-pixel
    class-id array, kept so callers can inspect coverage without re-running.
    """
    model, processor, (class_map, background) = _load()
    from PIL import Image

    pil = Image.fromarray(image_bgr[:, :, ::-1])
    inputs = processor(images=pil, task_inputs=["semantic"], return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    sem = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[pil.size[::-1]]
    )[0]
    sem = sem.cpu().numpy().astype(np.int32)

    regions: list[Region] = []
    for class_id in np.unique(sem):
        cid = int(class_id)
        if cid in background or cid not in class_map:
            continue
        label = class_map[cid]
        # `wall` here means the whole building mass, which is the envelope, not
        # a paintable surface. It is derived in wall_regions() after openings
        # are known, so emitting it now would just create a duplicate that
        # covers every window.
        if label == "wall":
            continue
        mask = (sem == cid).astype(np.uint8)
        regions.extend(_components(mask, label, cid))

    return regions, sem


def _components(mask: np.ndarray, label: str, class_id: int) -> list[Region]:
    """Split a class mask into connected components.

    A single 'window' class mask covers every window in the image. Costing needs
    them separated - both to deduct openings individually and to let the user
    assign a material to one surface without affecting its neighbours.
    """
    # Close small gaps first so a mullion or shadow does not split one window
    # into three fragments.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

    out: list[Region] = []
    for i in range(1, n):  # 0 is background
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < MIN_COMPONENT_PX:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])

        poly = _polygon(labels == i)
        out.append(
            Region(
                id=f"s{class_id}_{i}",
                label=label,
                box=(float(x), float(y), float(x + w), float(y + h)),
                # Semantic segmentation gives no per-pixel score. 0.8 reflects
                # that these masks are reliable but unranked, and keeps them
                # above heuristic regions in the merge ordering.
                confidence=0.80,
                polygon=poly,
                px_area=float(area),
                source="oneformer",
                notes=[f"ade20k_class={class_id}"],
            )
        )
    return out


def _polygon(component: np.ndarray, epsilon_frac: float = 0.003) -> list[list[float]]:
    m = component.astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    c = max(contours, key=cv2.contourArea)
    eps = epsilon_frac * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, eps, True)
    return [[float(p[0][0]), float(p[0][1])] for p in approx]


def building_envelope(sem: np.ndarray) -> np.ndarray:
    """Binary mask of the building: everything that is not background.

    This is what OneFormer is genuinely good at on facade photos. Measured on
    three residential images, it collapses the facade into a single
    `building`/`house` class and predicts window/door/column essentially never -
    ADE20K trains those on street scenes where a building is a background mass,
    not a subject filling the frame. So it is used for what it does reliably:
    separating building from sky, vegetation, road and grass.

    Components come from the detector instead, and `wall` is derived as
    envelope minus openings. That ordering also fixes vegetation bleeding into
    the wall mask, since plants are already classified as background here.
    """
    if _maps is None:
        return np.ones_like(sem, dtype=np.uint8)
    _, background = _maps
    mask = (~np.isin(sem, list(background))).astype(np.uint8)

    # Close gaps where a tree splits the facade, then keep only the dominant
    # blob so a distant neighbouring house does not join the envelope.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8)


def wall_regions(
    envelope: np.ndarray, components: list[Region], min_px: int | None = None
) -> list[Region]:
    """Paintable wall = building envelope minus every detected component.

    This is the inversion that makes the pipeline work. Rather than asking a
    model to find "wall" - which it cannot do reliably, because a wall is
    whatever is left once you remove the things that are not wall - the wall is
    computed by subtraction. Anything inside the building silhouette that no
    detector claimed is, by definition, exposed surface.

    Two consequences worth noting. Vegetation cannot bleed in, because plants
    are background and therefore outside the envelope. And the openings are
    removed geometrically, so the area visualised is the area priced.
    """
    floor = MIN_COMPONENT_PX if min_px is None else min_px
    mask = envelope.copy().astype(np.uint8)

    h, w = mask.shape[:2]
    for c in components:
        sub = np.zeros((h, w), dtype=np.uint8)
        if len(c.polygon) >= 3:
            cv2.fillPoly(sub, [np.asarray(c.polygon, dtype=np.int32)], 1)
        else:
            x1, y1, x2, y2 = (int(v) for v in c.box)
            cv2.rectangle(sub, (x1, y1), (x2, y2), 1, -1)
        mask[sub > 0] = 0

    # Detector boxes are slightly loose; erode-then-dilate removes the resulting
    # hairline slivers of "wall" that survive between adjacent components.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    out: list[Region] = []
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < floor:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        cw = int(stats[i, cv2.CC_STAT_WIDTH])
        ch = int(stats[i, cv2.CC_STAT_HEIGHT])
        out.append(
            Region(
                id=f"w{i}",
                label="wall",
                box=(float(x), float(y), float(x + cw), float(y + ch)),
                confidence=0.75,
                polygon=_polygon(labels == i),
                px_area=float(area),
                source="derived",
                notes=["envelope minus detected components"],
            )
        )
    return out


def coverage(sem: np.ndarray) -> float:
    """Fraction of the frame that is building rather than sky/vegetation/road.

    A low value means the house is small or heavily occluded in the photo, which
    is the single best early signal that downstream area estimates will be poor.
    """
    if _maps is None or sem.size == 0:
        return 0.0
    _, background = _maps
    bg = int(np.isin(sem, list(background)).sum())
    return (sem.size - bg) / sem.size
