"""Material catalog. PDF 5.3.

One record per material carrying both how it looks (the prompt fragment that
drives Gemini's image generation) and what it costs (the rates the estimator
uses). Keeping them together is deliberate: the picture and the price then
describe the same choice. There is no way to render a finish that has no rate,
or to price a finish the user never saw.

Rates are indicative Indian residential figures in INR, current-ish and
approximate. They are meant to be edited - 5.7 requires the user be able to
change rates and recalculate - so treat them as defaults, not truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Material:
    id: str
    name: str
    category: str
    applicable_to: list[str]

    # Drives both the swatch shown in the UI and the redesign prompt.
    swatch_css: str  # CSS background for the catalog chip
    prompt: str  # fed to the image model, must describe a real finish

    # --- local compositing fallback ---------------------------------------
    # Used when Gemini image generation is unavailable (quota, outage). The
    # same material then has two renderers: a generative one and a procedural
    # one, and both are driven from this single record so they cannot describe
    # different finishes.
    base_color: tuple[int, int, int] = (180, 180, 180)  # BGR, for OpenCV
    texture: str = "flat"  # flat | sand | stone | tile | wood | metal | glass
    tile_size_px: int = 64  # feature size at 1024px reference width
    gloss: float = 0.0  # 0 matte .. 1 reflective

    unit: str = "sqft"
    coverage_per_unit: float = 1.0  # area (or length) one purchase unit covers
    wastage_pct: float = 5.0
    material_rate: float = 0.0  # INR per purchase unit
    labour_rate: float = 0.0  # INR per sqft (or per rft when linear)
    linear: bool = False  # priced by running feet
    coats: int = 1

    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "applicable_to": self.applicable_to,
            "swatch_css": self.swatch_css,
            "unit": self.unit,
            "material_rate": self.material_rate,
            "labour_rate": self.labour_rate,
            "wastage_pct": self.wastage_pct,
            "coverage_per_unit": self.coverage_per_unit,
            "coats": self.coats,
            "linear": self.linear,
        }


CATALOG: dict[str, Material] = {
    m.id: m
    for m in [
        # --- paint --------------------------------------------------------
        Material(
            id="paint_ivory",
            name="Exterior Emulsion — Ivory",
            category="paint",
            applicable_to=["wall", "parapet", "balcony", "pillar", "roof_edge"],
            swatch_css="#e2d6c4",
            prompt="smooth matte ivory cream exterior emulsion paint",
            base_color=(196, 214, 226),
            texture="flat",
            unit="litre",
            coverage_per_unit=110.0,
            wastage_pct=5.0,
            material_rate=340.0,
            labour_rate=18.0,
            coats=2,
        ),
        Material(
            id="paint_slate",
            name="Exterior Emulsion — Slate Grey",
            category="paint",
            applicable_to=["wall", "parapet", "balcony", "pillar", "roof_edge"],
            swatch_css="#4e545c",
            prompt="smooth matte slate grey exterior emulsion paint",
            base_color=(92, 84, 78),
            texture="flat",
            unit="litre",
            coverage_per_unit=110.0,
            wastage_pct=5.0,
            material_rate=360.0,
            labour_rate=18.0,
            coats=2,
        ),
        Material(
            id="paint_terracotta",
            name="Exterior Emulsion — Terracotta",
            category="paint",
            applicable_to=["wall", "parapet", "balcony", "pillar"],
            swatch_css="#b05c3c",
            prompt="warm terracotta earth-red exterior emulsion paint",
            base_color=(60, 92, 176),
            texture="flat",
            unit="litre",
            coverage_per_unit=110.0,
            wastage_pct=5.0,
            material_rate=355.0,
            labour_rate=18.0,
            coats=2,
        ),
        # --- texture ------------------------------------------------------
        Material(
            id="texture_sand",
            name="Textured Finish — Sand",
            category="texture",
            applicable_to=["wall", "parapet"],
            swatch_css="#c8b89a",
            prompt="fine sand-textured exterior wall finish with subtle grain, warm beige",
            base_color=(154, 184, 200),
            texture="sand",
            unit="sqft",
            coverage_per_unit=1.0,
            wastage_pct=8.0,
            material_rate=48.0,
            labour_rate=32.0,
        ),
        # --- cladding -----------------------------------------------------
        Material(
            id="stone_slate",
            name="Natural Stone Cladding — Slate",
            category="cladding",
            applicable_to=["wall", "pillar", "parapet", "balcony"],
            swatch_css="linear-gradient(160deg,#5a5f66,#3d4147)",
            prompt=(
                "natural split-face slate stone cladding in irregular horizontal "
                "courses, charcoal grey with visible depth and shadow between stones"
            ),
            base_color=(78, 82, 88),
            texture="stone",
            tile_size_px=96,
            unit="sqft",
            coverage_per_unit=1.0,
            wastage_pct=12.0,
            material_rate=145.0,
            labour_rate=85.0,
        ),
        Material(
            id="stone_sandstone",
            name="Sandstone Cladding",
            category="cladding",
            applicable_to=["wall", "pillar", "parapet", "balcony"],
            swatch_css="linear-gradient(160deg,#c9a875,#a8874f)",
            prompt=(
                "warm beige sandstone cladding with visible horizontal courses "
                "and a lightly riven natural face"
            ),
            base_color=(117, 168, 201),
            texture="stone",
            tile_size_px=96,
            unit="sqft",
            coverage_per_unit=1.0,
            wastage_pct=12.0,
            material_rate=130.0,
            labour_rate=85.0,
        ),
        # --- tile ---------------------------------------------------------
        Material(
            id="tile_granite",
            name="Granite Tile — Polished",
            category="tile",
            applicable_to=["wall", "stairs", "pillar", "balcony"],
            swatch_css="linear-gradient(150deg,#4a4a4e,#2e2e32)",
            prompt=(
                "polished dark granite tile cladding in a regular grid with fine "
                "joints and a soft specular sheen"
            ),
            base_color=(70, 70, 74),
            texture="tile",
            tile_size_px=80,
            gloss=0.35,
            unit="piece",
            coverage_per_unit=4.0,  # a 2x2 ft tile
            wastage_pct=10.0,
            material_rate=420.0,
            labour_rate=70.0,
        ),
        # --- panel --------------------------------------------------------
        Material(
            id="wood_panel",
            name="WPC Wood-Finish Panel",
            category="panel",
            applicable_to=["wall", "balcony", "parapet", "gate", "garage", "door"],
            swatch_css="linear-gradient(90deg,#8a5a32,#6d4526,#8a5a32)",
            prompt=(
                "vertical wood-finish composite cladding panels, warm teak tone "
                "with fine visible grain and slim shadow gaps between boards"
            ),
            base_color=(50, 90, 138),
            texture="wood",
            tile_size_px=48,
            unit="sqft",
            coverage_per_unit=1.0,
            wastage_pct=10.0,
            material_rate=165.0,
            labour_rate=60.0,
        ),
        # --- railing ------------------------------------------------------
        Material(
            id="rail_glass",
            name="Toughened Glass Railing",
            category="railing",
            applicable_to=["railing", "balcony"],
            swatch_css="linear-gradient(160deg,#cfe0e6,#9fb8c2)",
            prompt=(
                "frameless toughened glass railing with slim brushed-steel spigots "
                "and a clean top rail"
            ),
            base_color=(194, 224, 207),
            texture="glass",
            gloss=0.8,
            unit="rft",
            coverage_per_unit=1.0,
            wastage_pct=5.0,
            material_rate=850.0,
            labour_rate=180.0,
            linear=True,
        ),
        Material(
            id="rail_ms",
            name="MS Powder-Coated Railing",
            category="railing",
            applicable_to=["railing", "balcony", "stairs", "gate"],
            swatch_css="#33363a",
            prompt=(
                "black powder-coated mild steel railing with slim vertical members, "
                "minimal modern profile"
            ),
            base_color=(58, 54, 51),
            texture="metal",
            gloss=0.25,
            unit="rft",
            coverage_per_unit=1.0,
            wastage_pct=5.0,
            material_rate=420.0,
            labour_rate=120.0,
            linear=True,
        ),
    ]
}


def get(material_id: str) -> Material:
    if material_id not in CATALOG:
        raise KeyError(f"unknown material '{material_id}'")
    return CATALOG[material_id]


def for_label(label: str) -> list[Material]:
    """Materials that may be applied to a given component (5.3)."""
    return [m for m in CATALOG.values() if label in m.applicable_to]


def catalog_payload() -> dict:
    """Catalog grouped for the frontend, plus the label->options index."""
    return {
        "materials": [m.to_dict() for m in CATALOG.values()],
        "by_label": {
            label: [m.id for m in for_label(label)]
            for label in {lbl for m in CATALOG.values() for lbl in m.applicable_to}
        },
    }


# A sensible starting design, so the demo has something to show on first load
# without the user having to assign nine materials by hand.
DEFAULT_DESIGN = {
    "wall": "texture_sand",
    "parapet": "paint_ivory",
    "pillar": "stone_slate",
    "balcony": "paint_ivory",
    "railing": "rail_ms",
    "stairs": "tile_granite",
    "gate": "rail_ms",
    "garage": "wood_panel",
    "roof_edge": "paint_ivory",
}
