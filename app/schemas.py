"""Data contracts for the pipeline.

Everything that moves between stages is a pydantic model, for two reasons.
Gemini's structured-output mode takes a JSON schema directly, so these classes
double as the prompt contract - the model is constrained to return exactly this
shape rather than prose we then have to parse. And the same objects serialise
straight to the frontend, so there is one definition of a "region" in the
system instead of three that drift apart.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Canonical component vocabulary. Gemini is constrained to this set, so a
# detected component always has a costing rule and a material list - there is
# no path where the model invents a label the estimator has never heard of.
LABELS = [
    "wall",
    "window",
    "door",
    "balcony",
    "pillar",
    "parapet",
    "railing",
    "gate",
    "garage",
    "stairs",
    "roof_edge",
]

LabelType = Literal[
    "wall",
    "window",
    "door",
    "balcony",
    "pillar",
    "parapet",
    "railing",
    "gate",
    "garage",
    "stairs",
    "roof_edge",
]


# --- Stage 1: image quality gate (PDF 5.1) ----------------------------------


class QualityCheck(BaseModel):
    """Whether this photo can be estimated from at all.

    5.1 requires rejecting unusable input and telling the user why. Doing this
    as a separate cheap call means we fail fast with a useful message instead
    of returning confident nonsense from a photo of a wall corner.
    """

    usable: bool = Field(description="True if the image shows an estimable house exterior")
    is_building_exterior: bool = Field(description="True if this is an exterior of a building")
    reason: str = Field(description="One sentence explaining the verdict")
    guidance: str = Field(description="If unusable, how to retake the photo. Empty if usable.")


# --- Stage 2: structure identification (PDF 5.2) ----------------------------


class Region(BaseModel):
    """One building surface.

    Geometry is a normalised polygon in [0,1] so it survives any resize; the
    frontend scales it to whatever canvas it renders at, and the estimator
    scales it to real pixels. `box` is kept alongside for cheap overlap tests.
    """

    id: str
    label: LabelType
    polygon: list[list[float]] = Field(
        description="Normalised [x,y] vertices in 0..1, clockwise, 4-12 points"
    )
    confidence: float = Field(ge=0.0, le=1.0)
    note: str = Field(default="", description="What the model saw, one short phrase")

    @property
    def box(self) -> tuple[float, float, float, float]:
        if not self.polygon:
            return (0.0, 0.0, 0.0, 0.0)
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        return (min(xs), min(ys), max(xs), max(ys))


class ScaleEstimate(BaseModel):
    """How the model converted pixels to feet.

    This is the single most important number in the system and the one most
    likely to be wrong, so it is surfaced explicitly rather than buried. The
    user can override it, and the report prints it - an estimate whose scale
    assumption is invisible is not auditable.
    """

    reference_object: str = Field(
        description="What was used for scale, e.g. 'standard door', 'ground floor height'"
    )
    reference_real_feet: float = Field(description="Assumed real-world size of that object in feet")
    reference_px_fraction: float = Field(
        description="That object's size as a fraction of image height, 0..1"
    )
    building_width_ft: float = Field(description="Estimated real width of the facade in feet")
    building_height_ft: float = Field(description="Estimated real height of the facade in feet")
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(description="Two sentences on how this was derived")


class Analysis(BaseModel):
    """Everything Gemini returns from one look at the photo."""

    regions: list[Region]
    scale: ScaleEstimate
    storeys: int = Field(description="Number of visible floors")
    style_note: str = Field(description="One line describing the building's current appearance")
    warnings: list[str] = Field(
        default_factory=list, description="Anything that will degrade the estimate"
    )


# --- Stage 4: quantities and cost (PDF 5.5-5.7) -----------------------------


class LineItem(BaseModel):
    """One priced row of the estimate.

    Deliberately verbose: 5.7 asks for a *transparent* breakdown, so every
    intermediate the arithmetic passed through is kept rather than collapsing
    to a total. A contractor reading the report can check each step.
    """

    region_ids: list[str]
    label: str
    material_id: str
    material_name: str

    net_area_sqft: float
    gross_area_sqft: float
    deducted_sqft: float
    unit: str = Field(description="Unit the work is measured in: sqft or rft")
    material_unit_label: str = Field(
        description="Unit the material is purchased in: litre, sqft, piece, rft"
    )

    quantity: float
    quantity_with_wastage: float
    wastage_pct: float
    coverage_note: str

    material_rate: float
    labour_rate: float
    material_cost: float
    labour_cost: float
    total_cost: float


class Estimate(BaseModel):
    line_items: list[LineItem]
    material_total: float
    labour_total: float
    grand_total: float
    scale_used: ScaleEstimate
    assumptions: list[str]
    warnings: list[str]
