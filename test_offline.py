"""Offline checks for everything that is not a Gemini call.

The estimation chain is the part of this system most worth testing: it is pure
arithmetic, it is where a silent error would produce a confident wrong number,
and it can be verified by hand. Gemini's perception cannot be unit-tested
meaningfully, so it is exercised separately with a real key.

    python test_offline.py
"""

from app import estimate as est
from app import materials, report
from app.schemas import Analysis, Region, ScaleEstimate


def make_analysis() -> Analysis:
    """A hand-built facade with numbers chosen so the maths is checkable.

    Frame is 30 ft x 20 ft = 600 sqft. The wall polygon covers 50% of the frame
    (0.1..0.9 horizontally, 0.15..0.85 vertically is 0.8 x 0.7 = 0.56), and two
    windows sit inside it.
    """
    return Analysis(
        regions=[
            Region(
                id="w1", label="wall", confidence=0.9, note="main facade",
                polygon=[[0.10, 0.15], [0.90, 0.15], [0.90, 0.85], [0.10, 0.85]],
            ),
            Region(
                id="win1", label="window", confidence=0.85, note="left window",
                polygon=[[0.20, 0.30], [0.35, 0.30], [0.35, 0.55], [0.20, 0.55]],
            ),
            Region(
                id="win2", label="window", confidence=0.85, note="right window",
                polygon=[[0.60, 0.30], [0.75, 0.30], [0.75, 0.55], [0.60, 0.55]],
            ),
            Region(
                id="d1", label="door", confidence=0.9, note="entrance",
                polygon=[[0.44, 0.50], [0.56, 0.50], [0.56, 0.85], [0.44, 0.85]],
            ),
            Region(
                id="r1", label="railing", confidence=0.7, note="balcony rail",
                polygon=[[0.15, 0.20], [0.85, 0.20], [0.85, 0.26], [0.15, 0.26]],
            ),
        ],
        scale=ScaleEstimate(
            reference_object="entrance door",
            reference_real_feet=6.75,
            reference_px_fraction=0.35,
            building_width_ft=30.0,
            building_height_ft=20.0,
            confidence=0.75,
            reasoning="Door spans 35% of image height at 6.75 ft, giving ~19 ft frame height.",
        ),
        storeys=1,
        style_note="Plain rendered single-storey house.",
        warnings=[],
    )


def check(name: str, got, want, tol=0.02):
    ok = abs(got - want) <= tol * max(abs(want), 1e-9)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: got {got:,.2f}, expected ~{want:,.2f}")
    return ok


def main() -> None:
    a = make_analysis()
    frame = a.scale.building_width_ft * a.scale.building_height_ft
    print(f"Frame: {a.scale.building_width_ft:.0f} x {a.scale.building_height_ft:.0f} "
          f"= {frame:,.0f} sqft\n")

    ok = True

    # --- area maths ------------------------------------------------------
    print("Area calculation")
    calc = est.AreaCalculator(a.scale, 1600, 1067)
    wall = next(r for r in a.regions if r.id == "w1")
    # 0.8 wide x 0.7 tall = 0.56 of frame = 336 sqft
    ok &= check("wall gross area", calc.to_sqft(wall.polygon), 0.56 * frame)

    win = next(r for r in a.regions if r.id == "win1")
    # 0.15 x 0.25 = 0.0375 of frame = 22.5 sqft
    ok &= check("window area", calc.to_sqft(win.polygon), 0.0375 * frame)

    rail = next(r for r in a.regions if r.id == "r1")
    # spans 0.15..0.85 = 0.7 of 30 ft = 21 rft
    ok &= check("railing length", calc.to_rft(rail.polygon), 0.7 * 30.0)

    # --- deduction -------------------------------------------------------
    print("\nOpening deduction")
    e = est.compute(
        a,
        {"w1": "paint_ivory", "r1": "rail_ms"},
        1600, 1067,
    )
    wall_item = next(i for i in e.line_items if i.label == "wall")
    # two windows (22.5 each) + one door (0.12 x 0.35 = 0.042 -> 25.2)
    expected_deduction = 2 * 22.5 + 25.2
    ok &= check("deducted area", wall_item.deducted_sqft, expected_deduction)
    ok &= check("net wall area", wall_item.net_area_sqft, 336.0 - expected_deduction)

    # --- paint quantity --------------------------------------------------
    print("\nPaint quantity (the multi-coat case)")
    m = materials.get("paint_ivory")
    net = wall_item.net_area_sqft
    # 2 coats, +5% wastage, 110 sqft/litre
    expected_litres = (net * 2 * 1.05) / 110.0
    ok &= check("litres of paint", wall_item.quantity, expected_litres)
    ok &= check("material cost", wall_item.material_cost, expected_litres * m.material_rate)
    ok &= check("labour cost", wall_item.labour_cost, net * m.labour_rate)

    # --- linear item -----------------------------------------------------
    print("\nRailing (priced per running foot)")
    rail_item = next(i for i in e.line_items if i.label == "railing")
    ok &= check("railing rft", rail_item.net_area_sqft, 21.0)
    assert rail_item.unit == "rft", "railing should be linear"
    print(f"  PASS  railing unit is '{rail_item.unit}', not sqft")

    # --- whole-number units ----------------------------------------------
    print("\nTile quantity rounds up to whole pieces")
    e2 = est.compute(a, {"w1": "tile_granite"}, 1600, 1067)
    tile_item = e2.line_items[0]
    is_whole = tile_item.quantity == int(tile_item.quantity)
    print(f"  {'PASS' if is_whole else 'FAIL'}  {tile_item.quantity} pieces (whole number)")
    ok &= is_whole

    # --- scale sensitivity ------------------------------------------------
    print("\nScale sensitivity (why scale is the number that matters)")
    bigger = a.scale.model_copy(update={"building_width_ft": 36.0})  # +20%
    e3 = est.compute(a, {"w1": "paint_ivory"}, 1600, 1067, scale_override=bigger)
    base = next(i for i in e.line_items if i.label == "wall").total_cost
    scaled = e3.line_items[0].total_cost
    print(f"  +20% facade width -> cost {base:,.0f} to {scaled:,.0f} "
          f"({(scaled / base - 1) * 100:+.0f}%)")
    ok &= check("cost scales linearly with scale error", scaled / base, 1.20)

    # --- rate override ----------------------------------------------------
    print("\nRate override (5.7)")
    e4 = est.compute(
        a, {"w1": "paint_ivory"}, 1600, 1067,
        rate_overrides={"paint_ivory": {"material_rate": 680.0}},
    )
    doubled = e4.line_items[0].material_cost
    orig = next(i for i in e.line_items if i.label == "wall").material_cost
    ok &= check("doubling the rate doubles material cost", doubled / orig, 2.0)

    # --- local compositor -------------------------------------------------
    # The fallback renderer must be provably safe: it may only change pixels
    # inside the selected surfaces. If it can touch sky or windows, the
    # redesign no longer describes the house the areas were measured from.
    print("\nLocal compositor (Gemini fallback)")
    ok &= _check_composite(a)

    # --- model fallback ---------------------------------------------------
    # Model ids move around between API versions and tiers. The fallback must
    # route around a missing model but must NOT burn extra calls on a quota
    # error, which would multiply the cost of an already-failing request.
    print("\nModel fallback")
    ok &= _check_fallback()

    # --- report -----------------------------------------------------------
    print("\nPDF report")
    png = _tiny_png()
    pdf = report.build(png, png, a, e)
    is_pdf = pdf[:4] == b"%PDF"
    print(f"  {'PASS' if is_pdf else 'FAIL'}  generated {len(pdf):,} bytes, "
          f"header {pdf[:4]!r}")
    ok &= is_pdf
    with open("out_test_report.pdf", "wb") as f:
        f.write(pdf)
    print("  wrote out_test_report.pdf")

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    raise SystemExit(0 if ok else 1)


def _check_composite(a: Analysis) -> bool:
    """Render locally and assert the mask is respected."""
    import cv2
    import numpy as np

    from app import composite, materials

    # Synthetic facade with a left-to-right lighting gradient, so we can check
    # the shading survives the repaint rather than being flattened.
    im = np.zeros((600, 900, 3), np.uint8)
    im[:] = (222, 200, 176)                                    # sky
    cv2.rectangle(im, (90, 130), (810, 470), (180, 198, 206), -1)  # wall
    grad = np.linspace(-45, 45, 720, dtype=np.float32)[None, :, None]
    im[130:470, 90:810] = np.clip(
        im[130:470, 90:810].astype(np.float32) + grad, 0, 255
    ).astype(np.uint8)
    cv2.rectangle(im, (180, 220), (310, 340), (148, 122, 96), -1)  # window
    src = cv2.imencode(".png", im)[1].tobytes()

    # `a`'s wall spans 0.10-0.90 x 0.15-0.85; window w1 sits inside it.
    out_bytes, mime = composite.render(src, a, {"w1": "stone_slate"}, materials.get)
    out = cv2.imdecode(np.frombuffer(out_bytes, np.uint8), cv2.IMREAD_COLOR)

    ok = True

    sky_delta = float(np.abs(out[0:80, :].astype(int) - im[0:80, :].astype(int)).mean())
    passed = sky_delta < 0.5
    print(f"  {'PASS' if passed else 'FAIL'}  sky untouched: mean delta {sky_delta:.2f}")
    ok &= passed

    # Inside the wall polygon but away from its edges, pixels must change.
    wall_delta = float(
        np.abs(out[200:260, 400:700].astype(int) - im[200:260, 400:700].astype(int)).mean()
    )
    passed = wall_delta > 5
    print(f"  {'PASS' if passed else 'FAIL'}  wall repainted: mean delta {wall_delta:.1f}")
    ok &= passed

    # The photo's lighting gradient must still be visible across the new
    # material - that is what separates this from a flat colour fill.
    row = out[300, 120:780].mean(axis=1)
    kept = float(row[-60:].mean() - row[:60].mean())
    passed = kept > 3
    print(f"  {'PASS' if passed else 'FAIL'}  original shading preserved: "
          f"gradient {kept:+.1f} across the wall")
    ok &= passed

    passed = mime == "image/png" and out_bytes[:4] == b"\x89PNG"
    print(f"  {'PASS' if passed else 'FAIL'}  returns a valid PNG ({len(out_bytes):,} bytes)")
    ok &= passed

    return ok


def _check_fallback() -> bool:
    """Exercise model resolution without touching the network."""
    from unittest.mock import MagicMock, patch

    from app import vision

    ok = True
    calls: list[str] = []

    def missing_then_ok(model, contents, config):
        calls.append(model)
        if model != "gemini-2.0-flash":
            raise Exception(f"404 models/{model} is not found for API version v1beta")
        return MagicMock(parsed="OK")

    fake = MagicMock()
    fake.models.generate_content = missing_then_ok
    vision._resolved_vision = None

    with patch.object(vision, "client", lambda: fake):
        vision.generate_with_fallback(
            ["gemini-2.5-flash", "gemini-flash-latest", "gemini-2.0-flash"],
            "x", None, remember=vision._remember_vision,
        )
    passed = vision._resolved_vision == "gemini-2.0-flash" and len(calls) == 3
    print(f"  {'PASS' if passed else 'FAIL'}  falls through missing models: {calls}")
    ok &= passed

    calls.clear()
    with patch.object(vision, "client", lambda: fake):
        vision.generate_with_fallback(
            vision._vision_models(), "x", None, remember=vision._remember_vision
        )
    passed = calls == ["gemini-2.0-flash"]
    print(f"  {'PASS' if passed else 'FAIL'}  caches resolution: {calls}")
    ok &= passed

    # A 503 is Google being busy, not a bad model. It must retry the SAME model
    # rather than burning through the fallback list.
    calls.clear()
    vision._resolved_vision = None
    busy = ("503 UNAVAILABLE. This model is currently experiencing high demand.")

    def flaky(model, contents, config):
        calls.append(model)
        if len(calls) < 3:
            raise Exception(busy)
        return MagicMock(parsed="OK")

    fake.models.generate_content = flaky
    with patch.object(vision, "client", lambda: fake), \
            patch.object(vision.time, "sleep", lambda s: None):
        vision.generate_with_fallback(
            ["gemini-2.5-flash", "gemini-2.0-flash"], "x", None,
            remember=vision._remember_vision,
        )
    passed = calls == ["gemini-2.5-flash"] * 3
    print(f"  {'PASS' if passed else 'FAIL'}  503 retries same model: {len(calls)} attempts, "
          f"{len(set(calls))} model(s)")
    ok &= passed

    calls.clear()
    vision._resolved_vision = None

    def quota(model, contents, config):
        calls.append(model)
        raise Exception("429 RESOURCE_EXHAUSTED: quota exceeded")

    fake.models.generate_content = quota
    with patch.object(vision, "client", lambda: fake):
        try:
            vision.generate_with_fallback(["a", "b", "c"], "x", None)
            passed = False
        except Exception as e:
            passed = "quota" in str(e).lower() and calls == ["a"]
    print(f"  {'PASS' if passed else 'FAIL'}  quota error does not retry: {calls}")
    ok &= passed

    vision._resolved_vision = None  # leave no state behind
    return ok


def _tiny_png() -> bytes:
    """A small valid image, so the report has something to lay out."""
    import io

    from PIL import Image

    im = Image.new("RGB", (640, 420), (170, 180, 165))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


if __name__ == "__main__":
    main()
