"""Material catalog.

One definition per material, carrying both what it looks like (for Stage 2
visualisation) and what it costs (for Stage 4 estimation). Keeping them in one
record is deliberate: the picture and the price then describe the same choice,
and there is no way for a material to be rendered that has no rate, or priced
with a finish the user never saw.

Rates are indicative Indian residential figures in INR and are meant to be
edited - 5.7 requires the user be able to modify rates and recalculate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Material:
    id: str
    name: str
    category: str
    applicable_to: list[str]

    # --- appearance -------------------------------------------------------
    base_color: tuple[int, int, int]  # BGR
    texture: str = "flat"  # flat | sand | stone | tile | wood | metal | glass
    tile_size_px: int = 64  # repeat size at 1024px reference width
    gloss: float = 0.0  # 0 matte .. 1 reflective
    prompt: str = ""  # diffusion prompt fragment

    # --- estimation -------------------------------------------------------
    unit: str = "sqft"
    coverage_per_unit: float = 1.0  # units of area covered by one purchase unit
    wastage_pct: float = 5.0
    material_rate: float = 0.0  # INR per purchase unit
    labor_rate: float = 0.0  # INR per sqft (or per rft for linear items)
    linear: bool = False  # priced by running feet, not area
    coats: int = 1

    notes: list[str] = field(default_factory=list)


CATALOG: dict[str, Material] = {
    m.id: m
    for m in [
        Material(
            id="paint_ivory",
            name="Exterior Emulsion - Ivory",
            category="paint",
            applicable_to=["wall", "parapet", "balcony"],
            base_color=(196, 214, 226),
            texture="flat",
            prompt="smooth matte ivory exterior emulsion paint",
            unit="litre",
            coverage_per_unit=110.0,
            wastage_pct=5.0,
            material_rate=340.0,
            labor_rate=18.0,
            coats=2,
        ),
        Material(
            id="paint_slate",
            name="Exterior Emulsion - Slate Grey",
            category="paint",
            applicable_to=["wall", "parapet", "balcony"],
            base_color=(92, 84, 78),
            texture="flat",
            prompt="smooth matte slate grey exterior emulsion paint",
            unit="litre",
            coverage_per_unit=110.0,
            wastage_pct=5.0,
            material_rate=360.0,
            labor_rate=18.0,
            coats=2,
        ),
        Material(
            id="texture_sand",
            name="Textured Finish - Sand",
            category="texture",
            applicable_to=["wall", "parapet"],
            base_color=(150, 172, 190),
            texture="sand",
            prompt="fine sand-textured exterior wall finish, subtle grain",
            unit="sqft",
            coverage_per_unit=1.0,
            wastage_pct=8.0,
            material_rate=48.0,
            labor_rate=32.0,
        ),
        Material(
            id="stone_slate",
            name="Natural Stone Cladding - Slate",
            category="cladding",
            applicable_to=["wall", "pillar", "parapet"],
            base_color=(78, 82, 88),
            texture="stone",
            tile_size_px=96,
            prompt="natural split-face slate stone cladding, irregular courses",
            unit="sqft",
            coverage_per_unit=1.0,
            wastage_pct=12.0,
            material_rate=145.0,
            labor_rate=85.0,
        ),
        Material(
            id="stone_sandstone",
            name="Sandstone Cladding",
            category="cladding",
            applicable_to=["wall", "pillar", "parapet"],
            base_color=(120, 158, 190),
            texture="stone",
            tile_size_px=96,
            prompt="warm beige sandstone cladding with visible courses",
            unit="sqft",
            coverage_per_unit=1.0,
            wastage_pct=12.0,
            material_rate=130.0,
            labor_rate=85.0,
        ),
        Material(
            id="tile_granite",
            name="Granite Tile",
            category="tile",
            applicable_to=["wall", "stairs", "pillar"],
            base_color=(70, 70, 74),
            texture="tile",
            tile_size_px=80,
            gloss=0.35,
            prompt="polished dark granite tile cladding with fine joints",
            unit="piece",
            # A 2x2 ft tile covers 4 sqft.
            coverage_per_unit=4.0,
            wastage_pct=10.0,
            material_rate=420.0,
            labor_rate=70.0,
        ),
        Material(
            id="wood_panel",
            name="WPC Wood-Finish Panel",
            category="panel",
            applicable_to=["wall", "balcony", "parapet", "gate", "garage", "door"],
            base_color=(58, 96, 140),
            texture="wood",
            tile_size_px=48,
            prompt="vertical wood-finish composite cladding panels",
            unit="sqft",
            coverage_per_unit=1.0,
            wastage_pct=10.0,
            material_rate=165.0,
            labor_rate=60.0,
        ),
        Material(
            id="rail_glass",
            name="Toughened Glass Railing",
            category="railing",
            applicable_to=["railing", "balcony"],
            base_color=(190, 180, 165),
            texture="glass",
            gloss=0.8,
            prompt="frameless toughened glass railing with steel spigots",
            unit="rft",
            coverage_per_unit=1.0,
            wastage_pct=5.0,
            material_rate=850.0,
            labor_rate=180.0,
            linear=True,
        ),
        Material(
            id="rail_ms",
            name="MS Powder-Coated Railing",
            category="railing",
            applicable_to=["railing", "balcony", "stairs", "gate"],
            base_color=(60, 60, 62),
            texture="metal",
            gloss=0.25,
            prompt="black powder-coated mild steel railing, slim vertical members",
            unit="rft",
            coverage_per_unit=1.0,
            wastage_pct=5.0,
            material_rate=420.0,
            labor_rate=120.0,
            linear=True,
        ),
    ]
}


def get(material_id: str) -> Material:
    if material_id not in CATALOG:
        raise KeyError(
            f"unknown material '{material_id}'. available: {', '.join(sorted(CATALOG))}"
        )
    return CATALOG[material_id]


def for_label(label: str) -> list[Material]:
    """Materials that may be applied to a given building component."""
    return [m for m in CATALOG.values() if label in m.applicable_to]


# --- procedural swatches ----------------------------------------------------
# Generated rather than shipped as image assets so the prototype stays
# self-contained. Swapping in real product photographs later is a matter of
# returning them from this function instead.


def swatch(m: Material, size: int = 256, seed: int = 0) -> np.ndarray:
    """Seamless-ish BGR texture tile for a material."""
    rng = np.random.default_rng(seed + abs(hash(m.id)) % 10_000)
    base = np.full((size, size, 3), m.base_color, dtype=np.float32)

    if m.texture == "sand":
        # GaussianBlur collapses a trailing singleton axis, so blur in 2D and
        # restore it before broadcasting across channels.
        grain = rng.normal(0, 9, (size, size)).astype(np.float32)
        grain = cv2.GaussianBlur(grain, (0, 0), 0.7)[:, :, None]
        base += grain

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
    y = 0
    row = 0
    while y < size:
        h = course + int(rng.integers(-4, 5))
        x = 0
        # Stagger alternate courses so joints do not line up vertically.
        offset = int(rng.integers(0, m.tile_size_px)) if row % 2 else 0
        x -= offset
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
    base += rng.normal(0, 5, (size, size, 1)).astype(np.float32)
    return base


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
    base += rng.normal(0, 4, (size, size, 1)).astype(np.float32)
    return base


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
    base += rng.normal(0, 3, (size, size, 1)).astype(np.float32)
    return base
