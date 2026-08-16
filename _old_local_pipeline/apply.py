"""Apply a material to a masked region, preserving the building's own lighting.

The requirement in 5.4 is that the output "preserve the original building
structure". A naive fill satisfies that geometrically but looks pasted on,
because it discards the shading that tells the eye where the surface is. So the
material supplies chroma while the original photograph supplies luminance: every
shadow, every gradient across a wall, every darkening under an eave survives.

That also makes the result honest. Nothing outside the mask can change, so the
redesigned image describes the same house the areas were measured from.
"""

from __future__ import annotations

import cv2
import numpy as np

from materials import Material, swatch
from regions import Region


def region_mask(shape: tuple[int, int], r: Region, feather: int = 3) -> np.ndarray:
    """Float mask in [0,1] for one region, softened at the edges."""
    h, w = shape[:2]
    m = np.zeros((h, w), dtype=np.uint8)
    if len(r.polygon) >= 3:
        cv2.fillPoly(m, [np.asarray(r.polygon, dtype=np.int32)], 255)
    else:
        x1, y1, x2, y2 = (int(v) for v in r.box)
        cv2.rectangle(m, (x1, y1), (x2, y2), 255, -1)
    if feather > 0:
        k = feather * 2 + 1
        m = cv2.GaussianBlur(m, (k, k), 0)
    return m.astype(np.float32) / 255.0


def subtract(mask: np.ndarray, others: list[Region], shape) -> np.ndarray:
    """Cut openings out of a surface mask.

    A wall is the surface minus its windows and doors. Without this the paint
    covers the glass, and - more importantly for the estimate - the area being
    visualised stops matching the area being priced.
    """
    out = mask.copy()
    for o in others:
        out *= 1.0 - region_mask(shape, o, feather=2)
    return np.clip(out, 0.0, 1.0)


def _tiled(texture: np.ndarray, h: int, w: int, scale: float) -> np.ndarray:
    """Repeat a swatch to cover h x w at a given on-image scale."""
    th = max(8, int(texture.shape[0] * scale))
    tw = max(8, int(texture.shape[1] * scale))
    tile = cv2.resize(texture, (tw, th), interpolation=cv2.INTER_LINEAR)
    reps_y = int(np.ceil(h / th))
    reps_x = int(np.ceil(w / tw))
    return np.tile(tile, (reps_y, reps_x, 1))[:h, :w]


def _luminance_transfer(
    original: np.ndarray, material: np.ndarray, strength: float
) -> np.ndarray:
    """Recolour to the material while keeping the original's shading.

    Works in LAB: take a/b (colour) from the material, and L (lightness) from a
    blend of the two. The blend re-centres the original's luminance on the
    material's mean so a dark wall repainted ivory actually reads as ivory,
    rather than as ivory-coloured shadow.
    """
    orig_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float32)
    mat_lab = cv2.cvtColor(material, cv2.COLOR_BGR2LAB).astype(np.float32)

    o_l = orig_lab[:, :, 0]
    m_l = mat_lab[:, :, 0]

    o_mean, o_std = float(o_l.mean()), float(o_l.std() + 1e-6)
    m_mean, m_std = float(m_l.mean()), float(m_l.std() + 1e-6)

    # Original shading, renormalised onto the material's tonal range.
    shaped = (o_l - o_mean) / o_std * max(m_std, 6.0) + m_mean
    out_l = shaped * strength + m_l * (1.0 - strength)

    out = mat_lab.copy()
    out[:, :, 0] = np.clip(out_l, 0, 255)
    return cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_LAB2BGR)


def apply_material(
    image: np.ndarray,
    mask: np.ndarray,
    m: Material,
    texture_scale: float = 1.0,
    shading_strength: float = 0.75,
    seed: int = 0,
) -> np.ndarray:
    """Composite one material onto `image` wherever `mask` is non-zero."""
    h, w = image.shape[:2]
    tex = swatch(m, 256, seed=seed)

    # Scale the swatch relative to the image so a 96px stone course looks like
    # stone at any input resolution.
    scale = texture_scale * (w / 1024.0) * (m.tile_size_px / 64.0)
    tiled = _tiled(tex, h, w, max(0.15, scale))

    recolored = _luminance_transfer(image, tiled, shading_strength)

    if m.gloss > 0:
        recolored = _add_gloss(recolored, m.gloss)

    m3 = mask[:, :, None]
    return (image.astype(np.float32) * (1 - m3) + recolored.astype(np.float32) * m3).astype(
        np.uint8
    )


def _add_gloss(img: np.ndarray, gloss: float) -> np.ndarray:
    """Broad vertical sheen for polished/reflective finishes."""
    h, w = img.shape[:2]
    ramp = np.linspace(1.0 + 0.18 * gloss, 1.0 - 0.10 * gloss, h, dtype=np.float32)
    out = img.astype(np.float32) * ramp[:, None, None]
    return np.clip(out, 0, 255).astype(np.uint8)


OPENING_LABELS = {"window", "door", "gate", "garage"}


def render_design(
    image: np.ndarray,
    regions: list[Region],
    selections: dict[str, str],
    catalog_get,
    shading_strength: float = 0.75,
) -> tuple[np.ndarray, list[str]]:
    """Apply a {region_id: material_id} mapping to the whole facade.

    Large surfaces are painted first so that smaller details composited later
    are not overwritten by the wall behind them.
    """
    out = image.copy()
    notes: list[str] = []
    shape = image.shape[:2]

    by_id = {r.id: r for r in regions}
    openings = [r for r in regions if r.label in OPENING_LABELS]

    ordered = sorted(
        (r for r in regions if r.id in selections),
        key=lambda r: r.px_area,
        reverse=True,
    )

    for r in ordered:
        try:
            m = catalog_get(selections[r.id])
        except KeyError as e:
            notes.append(str(e))
            continue

        if r.label not in m.applicable_to:
            notes.append(
                f"{r.id}: '{m.id}' is not listed for {r.label}; applied anyway"
            )

        mask = region_mask(shape, r)
        # Only cut openings out of surfaces that contain them.
        if r.label not in OPENING_LABELS:
            inside = [
                o for o in openings if o.id != r.id and _overlaps(o, r)
            ]
            if inside:
                mask = subtract(mask, inside, shape)

        out = apply_material(out, mask, m, shading_strength=shading_strength)

    unknown = set(selections) - set(by_id)
    if unknown:
        notes.append(f"selections reference unknown regions: {sorted(unknown)}")

    return out, notes


def _overlaps(inner: Region, outer: Region) -> bool:
    ax1, ay1, ax2, ay2 = inner.box
    bx1, by1, bx2, by2 = outer.box
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inner_area = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
    return (ix * iy) / inner_area > 0.35


def side_by_side(before: np.ndarray, after: np.ndarray, gap: int = 12) -> np.ndarray:
    """Original and redesign in one frame - 5.4 requires the comparison."""
    h = max(before.shape[0], after.shape[0])
    def fit(img):
        if img.shape[0] == h:
            return img
        s = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * s), h))

    b, a = fit(before), fit(after)
    strip = np.full((h, gap, 3), 255, dtype=np.uint8)
    out = np.hstack([b, strip, a])

    for img, x, text in ((b, 0, "ORIGINAL"), (a, b.shape[1] + gap, "REDESIGNED")):
        cv2.rectangle(out, (x + 10, 10), (x + 160, 42), (255, 255, 255), -1)
        cv2.putText(out, text, (x + 18, 33), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (30, 30, 30), 2, cv2.LINE_AA)
    return out
