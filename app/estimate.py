"""Regions + materials -> areas -> quantities -> cost. PDF 5.5, 5.6, 5.7.

This module is deliberately plain arithmetic with no AI in it. Once Gemini has
supplied the geometry and the scale, everything downstream is quantity
surveying, and a homeowner arguing with a contractor needs those numbers to be
reproducible and inspectable rather than generated.

The chain, and where error enters it:

    normalised polygon                    (Gemini - boundary error, ~5-10%)
      x image aspect                      (exact)
      x facade real dimensions            (Gemini - scale error, THE big one)
      = gross square feet
      - openings inside the surface       (exact, geometric)
      = net square feet
      x wastage                           (trade convention)
      / coverage per unit                 (product spec)
      = purchase quantity
      x rates                             (user-editable)
      = cost

Scale error dominates everything else and it enters as a multiplier, so a 20%
scale mistake is a 20% cost mistake. That is why the scale is exposed in the UI
and printed in the report rather than hidden.
"""

from __future__ import annotations

from .materials import get as get_material
from .schemas import Analysis, Estimate, LineItem, Region, ScaleEstimate

# Surfaces whose area is cut out of the wall behind them. A window is not a
# paintable surface; charging for it is the single most common way a naive
# estimate comes in high.
OPENING_LABELS = {"window", "door", "garage"}

# Components priced by running feet rather than square feet. A railing quote is
# per rft of balustrade, so its area is meaningless - we need its length.
LINEAR_LABELS = {"railing"}

# Fraction of a region's bounding box that an opening must overlap before it is
# treated as belonging to that surface. Loose polygons make exact containment
# tests too brittle; this tolerates the slop without letting a window on the
# far side of the house deduct from this wall.
CONTAINMENT_TOLERANCE = 0.5


def _poly_area_norm(poly: list[list[float]]) -> float:
    """Shoelace in normalised units."""
    n = len(poly)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _overlap_fraction(inner: Region, outer: Region) -> float:
    """How much of `inner`'s box sits inside `outer`'s box.

    Box-level rather than polygon-level on purpose. Precise polygon clipping
    would be more correct, but the polygons are model output with looser
    boundaries than the clipping would imply - the extra precision would be
    spurious, and it is a dependency and a page of code for no real accuracy.
    """
    ax1, ay1, ax2, ay2 = inner.box
    bx1, by1, bx2, by2 = outer.box
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inner_area = max(1e-9, (ax2 - ax1) * (ay2 - ay1))
    return (ix * iy) / inner_area


def _perimeter_width_norm(poly: list[list[float]]) -> float:
    """Horizontal extent of a polygon, for linear items."""
    if not poly:
        return 0.0
    xs = [p[0] for p in poly]
    return max(xs) - min(xs)


class AreaCalculator:
    """Converts normalised geometry into real-world square feet.

    Holds the scale so the conversion happens in exactly one place. Every area
    in the estimate passes through `to_sqft`, which means a scale correction is
    a single-value change and cannot half-apply.
    """

    def __init__(self, scale: ScaleEstimate, image_w: int, image_h: int):
        self.scale = scale
        self.image_w = image_w
        self.image_h = image_h
        # Real-world area of the full image frame at the facade's depth.
        # Normalised polygon area is a fraction of the frame, so multiplying by
        # this converts directly - no per-axis bookkeeping needed.
        self.frame_sqft = scale.building_width_ft * scale.building_height_ft

    def to_sqft(self, poly: list[list[float]]) -> float:
        """Normalised polygon -> square feet of real facade.

        The facade is assumed to fill the frame, which is what the model was
        asked to estimate width/height for. If the house occupies only part of
        the photo, the model reports the facade's own dimensions and this stays
        consistent, because the polygon fractions shrink in proportion.
        """
        return _poly_area_norm(poly) * self.frame_sqft

    def to_rft(self, poly: list[list[float]]) -> float:
        """Normalised polygon -> running feet, for railings and similar."""
        return _perimeter_width_norm(poly) * self.scale.building_width_ft


def compute(
    analysis: Analysis,
    selections: dict[str, str],
    image_w: int,
    image_h: int,
    rate_overrides: dict[str, dict[str, float]] | None = None,
    scale_override: ScaleEstimate | None = None,
) -> Estimate:
    """Build the full costed estimate.

    `selections` maps region id -> material id. Regions with no selection are
    left out entirely rather than defaulted, so the total only ever reflects
    work the user actually chose.

    `rate_overrides` is {material_id: {"material_rate": x, "labour_rate": y}},
    satisfying 5.7's requirement that the user can change rates and recalculate.
    """
    scale = scale_override or analysis.scale
    calc = AreaCalculator(scale, image_w, image_h)
    overrides = rate_overrides or {}

    by_id = {r.id: r for r in analysis.regions}
    openings = [r for r in analysis.regions if r.label in OPENING_LABELS]

    # Group by (label, material) so the report reads like a real quotation -
    # "Exterior Emulsion on walls, 940 sqft" - rather than one row per polygon.
    groups: dict[tuple[str, str], list[Region]] = {}
    for rid, mid in selections.items():
        r = by_id.get(rid)
        if r is None:
            continue
        groups.setdefault((r.label, mid), []).append(r)

    items: list[LineItem] = []
    warnings: list[str] = list(analysis.warnings)

    for (label, material_id), regions in sorted(groups.items()):
        try:
            mat = get_material(material_id)
        except KeyError:
            warnings.append(f"unknown material '{material_id}' skipped")
            continue

        if label not in mat.applicable_to:
            warnings.append(
                f"'{mat.name}' is not normally used on {label}; costed anyway at your request"
            )

        ov = overrides.get(material_id, {})
        material_rate = float(ov.get("material_rate", mat.material_rate))
        labour_rate = float(ov.get("labour_rate", mat.labour_rate))

        is_linear = mat.linear or label in LINEAR_LABELS

        if is_linear:
            length = sum(calc.to_rft(r.polygon) for r in regions)
            gross = net = round(length, 2)
            deducted = 0.0
            unit_label = "rft"
        else:
            gross = sum(calc.to_sqft(r.polygon) for r in regions)
            # Deduct only openings that actually sit within these surfaces.
            deducted = 0.0
            for r in regions:
                if r.label in OPENING_LABELS:
                    continue
                for o in openings:
                    if o.id == r.id:
                        continue
                    if _overlap_fraction(o, r) >= CONTAINMENT_TOLERANCE:
                        deducted += calc.to_sqft(o.polygon)
            # An over-deduction means the model double-counted openings or the
            # wall polygon was too tight. Floor at 10% rather than going
            # negative, and say so.
            net = max(gross * 0.10, gross - deducted)
            if deducted > gross * 0.9:
                warnings.append(
                    f"openings covered {deducted / max(gross, 1e-6):.0%} of the {label} "
                    "area - the deduction was capped; check the detected regions"
                )
            gross, net, deducted = round(gross, 2), round(net, 2), round(deducted, 2)
            unit_label = "sqft"

        # --- quantities (5.6) ---------------------------------------------
        # Paint is the interesting case: it is bought by the litre, covers a
        # stated area per litre, and needs multiple coats. Everything else is
        # area or length divided by what one purchase unit covers.
        billable = net * mat.coats if mat.unit == "litre" else net
        with_wastage = billable * (1.0 + mat.wastage_pct / 100.0)
        quantity = with_wastage / max(mat.coverage_per_unit, 1e-9)

        # Tiles and boards are bought whole; you cannot order 43.2 tiles.
        if mat.unit in ("piece", "box"):
            quantity = float(int(quantity) + (1 if quantity % 1 else 0))

        if mat.unit == "litre":
            coverage_note = (
                f"{mat.coats} coat(s) over {net:,.0f} sqft = {billable:,.0f} sqft of "
                f"painting, at {mat.coverage_per_unit:.0f} sqft/litre"
            )
        elif mat.unit == "piece":
            coverage_note = f"one piece covers {mat.coverage_per_unit:.2f} sqft"
        else:
            coverage_note = f"priced per {mat.unit}"

        material_cost = quantity * material_rate
        labour_cost = net * labour_rate  # labour is on actual worked area
        total = material_cost + labour_cost

        items.append(
            LineItem(
                region_ids=[r.id for r in regions],
                label=label,
                material_id=mat.id,
                material_name=mat.name,
                net_area_sqft=net,
                gross_area_sqft=gross,
                deducted_sqft=deducted,
                unit=unit_label,
                material_unit_label=mat.unit,
                quantity=round(quantity, 2),
                quantity_with_wastage=round(with_wastage, 2),
                wastage_pct=mat.wastage_pct,
                coverage_note=coverage_note,
                material_rate=material_rate,
                labour_rate=labour_rate,
                material_cost=round(material_cost, 2),
                labour_cost=round(labour_cost, 2),
                total_cost=round(total, 2),
            )
        )

    items.sort(key=lambda i: i.total_cost, reverse=True)

    material_total = round(sum(i.material_cost for i in items), 2)
    labour_total = round(sum(i.labour_cost for i in items), 2)

    return Estimate(
        line_items=items,
        material_total=material_total,
        labour_total=labour_total,
        grand_total=round(material_total + labour_total, 2),
        scale_used=scale,
        assumptions=_assumptions(scale, items),
        warnings=warnings,
    )


def _assumptions(scale: ScaleEstimate, items: list[LineItem]) -> list[str]:
    """Everything the number depends on, in plain language.

    9.4 asks for documentation of how the estimation works. Putting it in the
    estimate itself, rather than only in a README, means it travels with the
    report to the contractor who will actually challenge it.
    """
    out = [
        f"Scale derived from {scale.reference_object}, assumed {scale.reference_real_feet:.2f} ft, "
        f"giving a facade of {scale.building_width_ft:.0f} x {scale.building_height_ft:.0f} ft.",
        "Areas are measured from a single photograph. Surfaces not visible in the "
        "photo (side and rear walls) are NOT included.",
        "Only the visible face of each surface is measured; returns, reveals and "
        "soffits are not counted.",
        "Wall areas have window, door and garage openings deducted geometrically.",
        "Rates are indicative Indian residential figures in INR and should be "
        "replaced with local quotations before committing to a budget.",
        "Labour is costed on net worked area; material is costed on quantity "
        "purchased, which includes wastage.",
    ]
    if any(i.unit == "rft" for i in items):
        out.append(
            "Railings are measured as horizontal running feet of the visible span."
        )
    out.append(
        "Scaffolding, surface preparation, repairs, waterproofing, statutory "
        "approvals and taxes are excluded."
    )
    return out
