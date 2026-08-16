"""Gemini vision: photo -> labelled facade regions + a real-world scale.

Why one multimodal model instead of a segmentation stack:

An earlier version of this project ran OneFormer (ADE20K) for the building
envelope, GroundingDINO for components and SAM 2.1 for boundary refinement.
It worked, but it cost ~80s per image on CPU, ~2.2GB of model downloads, and
it still missed windows and doors on real photos - ADE20K trains those classes
on street scenes where a building is background mass, so a house filling the
frame gets labelled monolithically.

The deeper problem is that a segmentation stack only returns *where* things
are. It cannot tell you a door is about 7 feet tall, and without that the
pixels never become square feet. That step had to be a hand-written heuristic
sitting downstream of three models.

A multimodal model does both in one call: it identifies the components *and*
reasons about scale from the same visual evidence, because it knows what a
domestic door is. That is the actual insight here - the hard part of this
problem is not segmentation, it is monocular metric scale, and that is a
reasoning task rather than a pixel-labelling one.

The tradeoff, stated honestly: polygon boundaries are looser than SAM's, and
the model is non-deterministic. We spend effort on validating its output
(see `_sanitise`) rather than on refining edges, because a 5% boundary error
moves the estimate far less than a wrong scale assumption does.
"""

from __future__ import annotations

import json
import time
import os

from google import genai
from google.genai import types

from .schemas import LABELS, Analysis, QualityCheck

# Flash is the right tier here: this is perception plus arithmetic-free
# reasoning, not long-form generation, and the latency difference is what makes
# the demo feel interactive.
#
# Overridable by env so a newer model can be tried without touching code, and
# so the fallback list below can be skipped when you know what you want.
VISION_MODEL = os.environ.get("GEMINI_VISION_MODEL", "gemini-2.5-flash")

# Tried in order if the configured model is rejected. Model availability varies
# by API key, region and tier, and a hard-coded id that 404s would make the
# whole demo look broken when the fix is one substitution.
VISION_FALLBACKS = [
    "gemini-2.5-flash",
    "gemini-flash-latest",
    "gemini-2.0-flash",
    "gemini-2.5-pro",
]

_client: genai.Client | None = None
_resolved_vision: str | None = None


def client() -> genai.Client:
    """Lazily construct the SDK client so importing this module never fails."""
    global _client
    if _client is None:
        key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Put it in a .env file at the project "
                "root as GEMINI_API_KEY=your_key_here"
            )
        _client = genai.Client(api_key=key)
    return _client


def _part(image_bytes: bytes, mime: str) -> types.Part:
    return types.Part.from_bytes(data=image_bytes, mime_type=mime)


def _is_missing_model(e: Exception) -> bool:
    """Distinguish 'no such model' from real failures.

    Only a missing/unsupported model should trigger a fallback. A quota error
    or a safety block must surface to the user, not silently re-run against a
    different model and cost another call.
    """
    s = str(e).lower()
    return ("not found" in s or "404" in s or "not supported" in s
            or "does not exist" in s) and "model" in s


def _is_transient(e: Exception) -> bool:
    """Server-side wobble that a retry genuinely fixes.

    503 UNAVAILABLE ("this model is experiencing high demand") is common on the
    free tier and clears within seconds. Retrying is the correct response, and
    it is distinct both from a missing model (fall back) and from a quota error
    (stop immediately - retrying makes that worse).
    """
    s = str(e).lower()
    return (
        "503" in s
        or "unavailable" in s
        or "500" in s
        or "internal error" in s
        or "overloaded" in s
        or "deadline" in s
        or "timeout" in s
    )


def generate_with_fallback(
    models: list[str], contents, config, remember=None, attempts: int = 3
):
    """Call the first model that works, remembering the winner.

    Three distinct failure classes, deliberately handled differently:

      transient (503/500)  retry the same model with backoff
      missing model (404)  move to the next candidate
      anything else        raise immediately - quota errors especially, where
                           retrying burns the remaining allowance

    Resolution is cached per process, so the fallback scan happens at most once.
    """
    last: Exception | None = None

    for name in models:
        for attempt in range(attempts):
            try:
                resp = client().models.generate_content(
                    model=name, contents=contents, config=config
                )
                if remember:
                    remember(name)
                return resp
            except Exception as e:  # noqa: BLE001
                last = e

                if _is_transient(e) and attempt < attempts - 1:
                    wait = 2 ** attempt  # 1s, 2s
                    print(
                        f"[vision] {name} busy (attempt {attempt + 1}/{attempts}), "
                        f"retrying in {wait}s"
                    )
                    time.sleep(wait)
                    continue

                if _is_missing_model(e):
                    print(f"[vision] model '{name}' unavailable, trying next")
                    break  # next model

                raise  # quota, auth, safety - caller must see it

    raise RuntimeError(
        f"No usable Gemini model. Tried {', '.join(models)}. Last error: {last}"
    )


def _vision_models() -> list[str]:
    """Configured model first, then fallbacks, without duplicates."""
    if _resolved_vision:
        return [_resolved_vision]
    seen, out = set(), []
    for m in [VISION_MODEL, *VISION_FALLBACKS]:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _remember_vision(name: str) -> None:
    global _resolved_vision
    if _resolved_vision != name:
        _resolved_vision = name
        print(f"[vision] using {name}")


# --- 5.1 quality gate -------------------------------------------------------

QUALITY_PROMPT = """You are screening photographs for a house-renovation estimator.

Decide whether this photo can support an exterior renovation estimate.

Mark it UNUSABLE if any of these are true:
- it is not a building exterior (interior, person, object, screenshot, document)
- the facade is so dark, blurred or overexposed that surfaces cannot be told apart
- the building is a tiny part of the frame, or is mostly hidden behind trees, walls or vehicles
- it is an extreme close-up of one surface with no sense of the whole building

Mark it USABLE if a reasonable person could look at it and say "that is the
outside of that house" - it does NOT need to be a perfect straight-on shot.
Angled views, partial cropping at the edges, parked cars and ordinary
foreground clutter are all fine.

If unusable, `guidance` must be one practical sentence telling the homeowner
how to retake the photo. If usable, leave `guidance` empty."""


def check_quality(image_bytes: bytes, mime: str) -> QualityCheck:
    """PDF 5.1: reject unusable input and tell the user how to fix it."""
    resp = generate_with_fallback(
        _vision_models(),
        [_part(image_bytes, mime), QUALITY_PROMPT],
        types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=QualityCheck,
            temperature=0.0,  # a gate should be repeatable
        ),
        remember=_remember_vision,
    )
    return _parse(resp, QualityCheck)


# --- 5.2 + 5.5 structure identification and scale ---------------------------

ANALYSIS_PROMPT = f"""You are a quantity surveyor examining a photograph of a
house exterior before an exterior renovation. Produce a structured survey.

## Task 1 - identify surfaces

Outline every renovatable exterior surface you can see, using ONLY these labels:
{", ".join(LABELS)}

Rules that matter for costing:

- `wall` means an exposed, paintable/claddable wall plane. Emit ONE region per
  distinct wall plane. Do NOT emit one giant polygon covering the whole house,
  and do NOT cut holes for the windows - trace the wall's full outer extent and
  let the estimator deduct openings. A typical facade has 1-4 wall planes.
- `window` and `door` must each be a SEPARATE region, one per opening. These
  are deducted from wall area and are the primary scale reference, so missing
  one costs accuracy twice. Include every opening you can see, even partial ones.
- `parapet` is the low wall along the roofline. `roof_edge` is the projecting
  eave, band or overhang. They are different things - do not merge them.
- `railing` is the balustrade only (the bars/glass), NOT the balcony slab.
  `balcony` is the slab and its face.
- `pillar` covers columns and structural posts on the facade.
- Ignore anything that is not part of this building: sky, trees, plants, cars,
  road, neighbouring houses, people, compound walls that are clearly separate.

Geometry: give each region a polygon of 4-12 points, normalised to 0..1 with
[0,0] at the TOP-LEFT of the image. Trace the actual visible shape - if a wall
is a trapezoid in perspective, the polygon should be a trapezoid. Order points
clockwise. Confidence should reflect how sure you are it is that component.

## Task 2 - establish real-world scale

This is the most important part. The estimate is worthless without it.

Find a reference object whose real size you can rely on and measure it against
the image. In order of preference:
  1. an entrance door - domestic doors are ~6.75 ft tall (7 ft with frame)
  2. a standard window - typically 4-5 ft tall in Indian residential construction
  3. a storey - floor-to-floor is typically 10-11 ft
  4. a garage shutter - typically 7-8 ft tall

State which you used, its assumed real size in feet, and what fraction of the
IMAGE HEIGHT it occupies. Then derive the facade's real width and height in feet.

Sanity-check yourself before answering: a single-storey Indian house is
typically 10-14 ft tall and 20-40 ft wide. A two-storey is 20-24 ft tall. If
your numbers fall far outside that, you have misjudged the reference - redo it.

Account for perspective. If the photo is taken at an angle, the facade is
foreshortened and its true width is greater than it appears.

## Task 3 - report problems

In `warnings`, note anything that will degrade the estimate: heavy occlusion,
a strongly oblique angle, surfaces cut off by the frame, no usable scale
reference, deep shadow hiding part of the facade. Be specific. Empty list if
the photo is genuinely clean."""


def analyse(image_bytes: bytes, mime: str) -> Analysis:
    """PDF 5.2 + 5.5: components, geometry, and the pixel->feet bridge."""
    resp = generate_with_fallback(
        _vision_models(),
        [_part(image_bytes, mime), ANALYSIS_PROMPT],
        types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Analysis,
            # Low but non-zero. Fully greedy decoding made the model terser and
            # more likely to emit a single merged wall; a little slack produces
            # better-separated regions without harming repeatability much.
            temperature=0.2,
            # Perception here is genuinely hard - occlusion, perspective, scale
            # arithmetic - and thinking measurably improves region separation
            # and scale sanity. It is the main latency cost of this call.
            thinking_config=types.ThinkingConfig(thinking_budget=4096),
        ),
        remember=_remember_vision,
    )
    analysis = _parse(resp, Analysis)
    return _sanitise(analysis)


# --- validation -------------------------------------------------------------


def _sanitise(a: Analysis) -> Analysis:
    """Repair what the model gets wrong in predictable ways.

    Structured output guarantees the shape, never the sense. These are the
    failure modes actually observed in testing, each cheap to correct here and
    expensive to debug once the numbers reach a cost estimate.
    """
    seen: set[str] = set()
    clean = []

    for i, r in enumerate(a.regions):
        # Clamp strays. The model occasionally runs a polygon slightly past the
        # frame edge on surfaces that are cut off by the crop.
        r.polygon = [
            [min(1.0, max(0.0, float(x))), min(1.0, max(0.0, float(y)))]
            for x, y in r.polygon
            if len([x, y]) == 2
        ]
        if len(r.polygon) < 3:
            continue  # degenerate, cannot carry area

        # Drop specks. Below ~0.05% of frame this is noise, not a surface, and
        # it would add a line item worth a few rupees to the report.
        if _poly_area(r.polygon) < 0.0005:
            continue

        # Ids must be unique and stable - the frontend keys material selections
        # off them, and a duplicate id silently reassigns the wrong surface.
        rid = r.id.strip() or f"r{i}"
        while rid in seen:
            rid = f"{rid}_{i}"
        r.id = rid
        seen.add(rid)
        clean.append(r)

    a.regions = clean

    # Scale sanity. A wrong scale is the one error that silently multiplies
    # every number in the report, so it is bounded to physically plausible
    # residential dimensions and flagged loudly when we intervene.
    s = a.scale
    if not (6.0 <= s.building_height_ft <= 60.0):
        a.warnings.append(
            f"model estimated facade height {s.building_height_ft:.0f} ft, which is "
            "outside the plausible range for a low-rise house; clamped to 12 ft "
            "- set the height manually for a reliable estimate"
        )
        s.building_height_ft = 12.0
        s.confidence = min(s.confidence, 0.3)

    if not (8.0 <= s.building_width_ft <= 200.0):
        a.warnings.append(
            f"model estimated facade width {s.building_width_ft:.0f} ft, which is "
            "implausible; clamped to 30 ft - set the width manually"
        )
        s.building_width_ft = 30.0
        s.confidence = min(s.confidence, 0.3)

    if not any(r.label == "wall" for r in a.regions):
        a.warnings.append(
            "no wall surface was identified - there is nothing to estimate; "
            "try a photo showing more of the facade"
        )

    return a


def _poly_area(poly: list[list[float]]) -> float:
    """Shoelace, in normalised units."""
    n = len(poly)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _parse(resp, model_cls):
    """Pull a validated object out of a response.

    `resp.parsed` is the happy path. It comes back None when the model emits
    JSON that satisfies the schema loosely but not pydantic strictly, so we
    fall back to parsing the text before giving up - the alternative is losing
    a good answer to a technicality.
    """
    if getattr(resp, "parsed", None) is not None:
        return resp.parsed
    text = (getattr(resp, "text", "") or "").strip()
    if not text:
        raise RuntimeError(
            "Gemini returned an empty response. This usually means the image was "
            "blocked by a safety filter or the request exceeded a quota."
        )
    if text.startswith("```"):
        text = text.split("```")[1].removeprefix("json").strip()
    return model_cls.model_validate(json.loads(text))
