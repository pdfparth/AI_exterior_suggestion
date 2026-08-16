"""Gemini image generation: the user's actual house, refinished. PDF 5.4.

The requirement has a sharp edge to it. 5.4 asks for a redesign that
"preserves the original building structure" - so this must be an *edit* of the
photograph, not a generated house that resembles it. A text-to-image model
given a description produces a plausible bungalow that is not the customer's
bungalow, which is worse than useless for a pre-construction decision.

Two things keep it honest:

1. The original image is passed in as the primary input, with the prompt framed
   as a refinishing job on an existing photo. The model edits rather than
   invents.
2. The prompt enumerates what must not change - geometry, openings, camera,
   surroundings - because negative constraints are what actually hold structure
   in image editing. Positive instructions alone drift.

The material list is assembled from the same catalog the estimator prices from,
so the image and the invoice always describe one design.
"""

from __future__ import annotations

import os

from google.genai import types

from .materials import get as get_material
from .schemas import Analysis
from .vision import client, generate_with_fallback

# Gemini's image editing model. Given an input image it does localised
# refinishing well, which is exactly the operation 5.4 describes.
IMAGE_MODEL = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")

# Image model naming has churned more than the text models, so the fallback
# list matters more here. Tried in order until one accepts the request.
IMAGE_FALLBACKS = [
    "gemini-2.5-flash-image",
    "gemini-2.0-flash-preview-image-generation",
    "gemini-2.5-flash-image-preview",
]

_resolved_image: str | None = None


def _design_brief(analysis: Analysis, selections: dict[str, str]) -> list[str]:
    """Turn region->material choices into per-surface instructions.

    Grouped by component so the prompt reads as a specification ("all wall
    surfaces: sandstone cladding") rather than a list of polygon ids, which
    the image model has no way to interpret.
    """
    by_label: dict[str, set[str]] = {}
    regions = {r.id: r for r in analysis.regions}

    for rid, mid in selections.items():
        r = regions.get(rid)
        if r is None:
            continue
        by_label.setdefault(r.label, set()).add(mid)

    readable = {
        "wall": "the main wall surfaces",
        "parapet": "the parapet wall along the roofline",
        "balcony": "the balcony slab and its outer face",
        "pillar": "the pillars and columns",
        "railing": "the railings",
        "gate": "the gate",
        "garage": "the garage shutter",
        "stairs": "the external steps",
        "roof_edge": "the projecting roof edge and eaves band",
        "door": "the entrance door",
        "window": "the window frames",
    }

    lines = []
    for label, mids in sorted(by_label.items()):
        where = readable.get(label, f"the {label}")
        for mid in sorted(mids):
            try:
                mat = get_material(mid)
            except KeyError:
                continue
            lines.append(f"- Apply to {where}: {mat.prompt}.")
    return lines


def build_prompt(analysis: Analysis, selections: dict[str, str]) -> str:
    """Assemble the full editing instruction."""
    brief = _design_brief(analysis, selections)
    if not brief:
        brief = ["- Refresh the existing finishes without changing any colours."]

    return f"""Photorealistically refinish the exterior of the house in this
photograph. This is a renovation visualisation for the homeowner, so it must
show THEIR house with new finishes — not a different house.

MATERIAL SPECIFICATION
{chr(10).join(brief)}

MUST NOT CHANGE — these define whether the image is usable:
- The building's geometry. Every wall plane, roofline and projection stays
  exactly where it is. Do not add, remove, resize or move any storey.
- The position, size and shape of every window, door, balcony and opening.
- The camera angle, framing, focal length and perspective. This is the same
  photograph, refinished.
- The lighting. Keep the original sun direction, shadow positions and time of
  day. Existing shadows must fall across the new materials exactly as they fall
  across the old ones.
- The surroundings: sky, trees, plants, ground, road, neighbouring structures,
  parked vehicles and any people all stay as they are.

EXECUTION NOTES
- Apply the materials as a real contractor would: correct scale for the courses
  and joints, following the surface's perspective, wrapping around corners
  consistently.
- Keep it believable rather than glamorous. This should look like a good
  photograph of a finished renovation, not a CGI render or an advertisement.
- Do not add furniture, signage, watermarks, text, plants or decorative
  elements that are not in the original photograph.

Output the edited photograph only."""


def generate(
    image_bytes: bytes,
    mime: str,
    analysis: Analysis,
    selections: dict[str, str],
) -> tuple[bytes, str]:
    """Render the redesign. Returns (image_bytes, mime_type).

    Raises on failure rather than silently returning the original - a user
    comparing "before" against an identical "after" would reasonably conclude
    the renovation changes nothing.
    """
    prompt = build_prompt(analysis, selections)

    resp = generate_with_fallback(
        _image_models(),
        [types.Part.from_bytes(data=image_bytes, mime_type=mime), prompt],
        types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            # Editing should follow the brief, not improvise on it.
            temperature=0.4,
        ),
        remember=_remember_image,
    )

    for part in _parts(resp):
        inline = getattr(part, "inline_data", None)
        if inline is not None and getattr(inline, "data", None):
            return inline.data, (inline.mime_type or "image/png")

    raise RuntimeError(
        "Gemini returned no image. The request may have been blocked by a safety "
        "filter, or the model returned only text. Try a different photo."
    )


def _image_models() -> list[str]:
    if _resolved_image:
        return [_resolved_image]
    seen, out = set(), []
    for m in [IMAGE_MODEL, *IMAGE_FALLBACKS]:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _remember_image(name: str) -> None:
    global _resolved_image
    if _resolved_image != name:
        _resolved_image = name
        print(f"[redesign] using {name}")


def _parts(resp):
    """Walk candidates defensively - any of these can be absent on a block."""
    for cand in getattr(resp, "candidates", None) or []:
        content = getattr(cand, "content", None)
        for part in (getattr(content, "parts", None) or []):
            yield part
