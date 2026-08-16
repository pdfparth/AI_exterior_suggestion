"""Local material compositing - the fallback when Gemini cannot render. PDF 5.4.

Why this exists: image generation is the most quota-limited call in the system.
On a free-tier key the text calls (quality gate, survey) succeed comfortably
while image generation 429s, which would leave the demo dead at exactly the
step that matters most. This module renders the same design locally, instantly,
and for free.

The technique is a luminance transfer in LAB space rather than a colour fill.
A flat fill satisfies "apply the material" geometrically but looks pasted on,
because it discards the shading that tells the eye where the surface is. Here
the material supplies chroma (a/b) while the photograph supplies lightness (L),
so every shadow, every gradient across a wall and every darkening under an eave
survives the repaint.

Two properties this has that the generative path does not:

  Nothing outside the mask can change. The redesign is provably the same house
  the areas were measured from - the generative model can silently move a
  window, this cannot.

  It is deterministic. The same selections produce the same image every time.

What it gives up is photorealism: this reads as a well-executed recolour, not
as a photograph of a finished renovation. That is the honest trade, and it is
why Gemini is still tried first.
"""

from __future__ import annotations

import cv2
import numpy as np

from .materials import Material
from .schemas import Analysis, Region

# Openings are cut out of the surfaces behind them, so paint does not cover
# the glass. Mirrors OPENING_LABELS in estimate.py - the area visualised must
# be the area priced.
OPENING_LABELS = {"window", "door", "garage"}


# --- procedural swatches ----------------------------------------------------
# Generated rather than shipped as image assets so the prototype stays
# self-contained. Swapping in real product photographs later means returning
# them from `swatch()` instead.


def swatch(m: Material, size: int = 256, seed: int = 0) -> np.ndarray:
    """Seamless-ish BGR texture tile for a material."""
    rng = np.random.default_rng(seed + abs(hash(m.id)) % 10_000)
    base = np.full((size, size, 3), m.base_color, dtype=np.float32)

    if m.texture == "sand":
        # GaussianBlur collapses a trailing singleton axis, so blur in 2D and
        # restore it before broadcasting across channels.
        grain = rng.normal(0, 9, (size, size)).astype(np.float32)
        base += cv2.GaussianBlur(grain, (0, 0), 0.7)[:, :, None]

    elif m.texture == "stone":
        base = _stone(base, m, rng, size)

    elif m.texture == "tile":
        base = _tile(base, m, rng, size)

    elif m.texture == "wood":
        base = _wood(base, m, rng, size)

    elif m.texture == "metal":
        grad = np.linspace(-12, 12, size, dtype=np.float32)[None, :, None]
        base += grad + rng.normal(0, 2.5, (size, size, 1)).astype(np.float32)

    elif m.texture == "glass":
        grad = np.linspace(18, -18, size, dtype=np.float32)[:, None, None]
        base += grad + rng.normal(0, 1.5, (size, size, 1)).astype(np.float32)

    else:  # flat
        base += rng.normal(0, 2.0, (size, size, 1)).astype(np.float32)

    return np.clip(base, 0, 255).astype(np.uint8)


def _stone(base: np.ndarray, m: Material, rng, size: int) -> np.ndarray:
    """Irregular courses with mortar joints and per-block tonal variation."""
    course = max(12, m.tile_size_px // 2)
    y, row = 0, 0
    while y < size:
        h = course + int(rng.integers(-4, 5))
        # Stagger alternate courses so joints do not line up vertically.
        x = -int(rng.integers(0, m.tile_size_px)) if row % 2 else 0
        while x < size:
            w = m.tile_size_px + int(rng.integers(-18, 19))
            tone = rng.normal(0, 11)
            cv2.rectangle(
                base, (x, y), (x + w - 3, y + h - 3),
                tuple(float(c + tone) for c in m.base_color), -1,
            )
            x += w
        y += h
        row += 1
    base = cv2.GaussianBlur(base, (0, 0), 0.6)
    return base + rng.normal(0, 5, (size, size, 1)).astype(np.float32)


def _tile(base: np.ndarray, m: Material, rng, size: int) -> np.ndarray:
    """Regular grid with thin joints and speckle."""
    t = m.tile_size_px
    joint = tuple(float(c * 0.72) for c in m.base_color)
    for y in range(0, size, t):
        for x in range(0, size, t):
            tone = rng.normal(0, 6)
            cv2.rectangle(
                base, (x + 1, y + 1), (x + t - 2, y + t - 2),
                tuple(float(c + tone) for c in m.base_color), -1,
            )
    for y in range(0, size, t):
        cv2.line(base, (0, y), (size, y), joint, 2)
    for x in range(0, size, t):
        cv2.line(base, (x, 0), (x, size), joint, 2)
    return base + rng.normal(0, 4, (size, size, 1)).astype(np.float32)


def _wood(base: np.ndarray, m: Material, rng, size: int) -> np.ndarray:
    """Vertical planks with grain."""
    w = m.tile_size_px
    for x in range(0, size, w):
        tone = rng.normal(0, 9)
        cv2.rectangle(
            base, (x + 1, 0), (x + w - 2, size),
            tuple(float(c + tone) for c in m.base_color), -1,
        )
        for _ in range(6):  # grain lines
            gx = x + int(rng.integers(2, max(3, w - 2)))
            cv2.line(base, (gx, 0), (gx, size),
                     tuple(float(c * 0.9) for c in m.base_color), 1)
    base = cv2.GaussianBlur(base, (3, 1), 0)
    return base + rng.normal(0, 3, (size, size, 1)).astype(np.float32)


# --- compositing ------------------------------------------------------------


def region_mask(shape: tuple[int, int], r: Region, feather: int = 3) -> np.ndarray:
    """Float mask in [0,1] for one region, softened at the edges.

    Regions carry normalised polygons, so they are scaled to pixels here. The
    feather hides the polygon's straight-line approximation of a curved or
    slightly-off boundary; a hard edge makes model imprecision obvious.
    """
    h, w = shape[:2]
    m = np.zeros((h, w), dtype=np.uint8)
    if len(r.polygon) >= 3:
        pts = np.asarray([[x * w, y * h] for x, y in r.polygon], dtype=np.int32)
        cv2.fillPoly(m, [pts], 255)
    if feather > 0:
        k = feather * 2 + 1
        m = cv2.GaussianBlur(m, (k, k), 0)
    return m.astype(np.float32) / 255.0


def subtract(mask: np.ndarray, others: list[Region], shape) -> np.ndarray:
    """Cut openings out of a surface mask."""
    out = mask.copy()
    for o in others:
        out *= 1.0 - region_mask(shape, o, feather=2)
    return np.clip(out, 0.0, 1.0)


def _tiled(texture: np.ndarray, h: int, w: int, scale: float) -> np.ndarray:
    """Repeat a swatch to cover h x w at a given on-image scale."""
    th = max(8, int(texture.shape[0] * scale))
    tw = max(8, int(texture.shape[1] * scale))
    tile = cv2.resize(texture, (tw, th), interpolation=cv2.INTER_LINEAR)
    return np.tile(
        tile, (int(np.ceil(h / th)), int(np.ceil(w / tw)), 1)
    )[:h, :w]


def _luminance_transfer(
    original: np.ndarray, material: np.ndarray, strength: float
) -> np.ndarray:
    """Recolour to the material while keeping the original's shading.

    Works in LAB: take a/b (colour) from the material, and L (lightness) from a
    blend of the two. The blend re-centres the original's luminance on the
    material's mean, so a dark wall repainted ivory actually reads as ivory
    rather than as ivory-coloured shadow.
    """
    orig_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float32)
    mat_lab = cv2.cvtColor(material, cv2.COLOR_BGR2LAB).astype(np.float32)

    o_l, m_l = orig_lab[:, :, 0], mat_lab[:, :, 0]
    o_mean, o_std = float(o_l.mean()), float(o_l.std() + 1e-6)
    m_mean, m_std = float(m_l.mean()), float(m_l.std() + 1e-6)

    # Original shading, renormalised onto the material's tonal range.
    shaped = (o_l - o_mean) / o_std * max(m_std, 6.0) + m_mean
    out = mat_lab.copy()
    out[:, :, 0] = np.clip(shaped * strength + m_l * (1.0 - strength), 0, 255)
    return cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_LAB2BGR)


def _add_gloss(img: np.ndarray, gloss: float) -> np.ndarray:
    """Broad vertical sheen for polished/reflective finishes."""
    h = img.shape[0]
    ramp = np.linspace(1.0 + 0.18 * gloss, 1.0 - 0.10 * gloss, h, dtype=np.float32)
    return np.clip(img.astype(np.float32) * ramp[:, None, None], 0, 255).astype(np.uint8)


def apply_material(
    image: np.ndarray,
    mask: np.ndarray,
    m: Material,
    shading_strength: float = 0.75,
    seed: int = 0,
) -> np.ndarray:
    """Composite one material onto `image` wherever `mask` is non-zero."""
    h, w = image.shape[:2]
    tex = swatch(m, 256, seed=seed)

    # Scale the swatch relative to the image so a 96px stone course reads as
    # stone at any input resolution.
    scale = (w / 1024.0) * (m.tile_size_px / 64.0)
    tiled = _tiled(tex, h, w, max(0.15, scale))

    recolored = _luminance_transfer(image, tiled, shading_strength)
    if m.gloss > 0:
        recolored = _add_gloss(recolored, m.gloss)

    m3 = mask[:, :, None]
    return (
        image.astype(np.float32) * (1 - m3) + recolored.astype(np.float32) * m3
    ).astype(np.uint8)


def _overlaps(inner: Region, outer: Region) -> bool:
    ax1, ay1, ax2, ay2 = inner.box
    bx1, by1, bx2, by2 = outer.box
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inner_area = max(1e-9, (ax2 - ax1) * (ay2 - ay1))
    return (ix * iy) / inner_area > 0.35


def _poly_area(poly: list[list[float]]) -> float:
    n = len(poly)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def render(
    image_bytes: bytes,
    analysis: Analysis,
    selections: dict[str, str],
    catalog_get,
    shading_strength: float = 0.75,
) -> tuple[bytes, str]:
    """Render the design locally. Returns (png_bytes, mime).

    Large surfaces are composited first so smaller details layered afterwards
    are not overwritten by the wall behind them.
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("could not decode the source image for local rendering")

    shape = image.shape[:2]
    by_id = {r.id: r for r in analysis.regions}
    openings = [r for r in analysis.regions if r.label in OPENING_LABELS]

    ordered = sorted(
        (by_id[rid] for rid in selections if rid in by_id),
        key=lambda r: _poly_area(r.polygon),
        reverse=True,
    )

    out = image.copy()
    for r in ordered:
        try:
            m = catalog_get(selections[r.id])
        except KeyError:
            continue

        mask = region_mask(shape, r)
        # Only cut openings out of surfaces that contain them.
        if r.label not in OPENING_LABELS:
            inside = [o for o in openings if o.id != r.id and _overlaps(o, r)]
            if inside:
                mask = subtract(mask, inside, shape)

        out = apply_material(out, mask, m, shading_strength=shading_strength)

    ok, buf = cv2.imencode(".png", out)
    if not ok:
        raise RuntimeError("failed to encode the rendered image")
    return buf.tobytes(), "image/png"
