"""Configuration for the segmentation pipeline.

Thresholds are the main tuning knob. GroundingDINO's box_threshold trades
missed regions against false positives; the right value depends on the image.
Defaults here are tuned for residential facade photos.
"""

# --- Models -----------------------------------------------------------------

GROUNDING_DINO_MODEL = "IDEA-Research/grounding-dino-base"

# Semantic segmentation over ADE20K. This is what actually finds walls: walls
# are not objects, so a detector prompted with "wall" tends to return either the
# whole frame or nothing. Per-pixel labelling handles them properly and gives
# full-frame coverage for free.
ONEFORMER_MODEL = "shi-labs/oneformer_ade20k_swin_large"

# Mask refinement. sam2.1_l is the best available boundary quality; mobile_sam
# is ~10x faster and noticeably rougher. Both are box-prompted.
SAM_MODEL = "sam2.1_l.pt"
MOBILE_SAM_MODEL = "mobile_sam.pt"

# Minimum pixels for an ADE20K connected component to become a Region. Below
# this it is segmentation speckle, not a building surface.
MIN_COMPONENT_PX = 900

# --- Detection thresholds ---------------------------------------------------

BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.25

# Regions smaller than this fraction of the image are discarded as noise.
MIN_AREA_FRACTION = 0.001

# Two boxes of the same label overlapping more than this are merged.
MERGE_IOU_THRESHOLD = 0.6

# ...or when one box sits almost entirely inside another of the same label.
# Catches nested duplicates that score below the IoU threshold.
MERGE_CONTAINMENT_THRESHOLD = 0.85

# --- Component vocabulary ---------------------------------------------------
# GroundingDINO wants lowercase noun phrases separated by periods. It is strong
# on common nouns and weak on construction jargon, so several PDF components map
# to descriptive phrases rather than their trade names. `parapet` in particular
# is not reliably detectable by text prompt and is recovered geometrically in
# postprocess.py instead.

# Every building COMPONENT comes from here.
#
# This was measured, not assumed. On three residential facade photos OneFormer
# predicted `building`/`house` for 40-60% of pixels and predicted `window` zero
# times, `column` zero times, and `door` once. ADE20K trains those classes on
# street scenes where a building is a background mass; when the house fills the
# frame the model labels it monolithically. So semantic segmentation is used for
# the building envelope only, and components are detected here instead.
PROMPT_MAP = {
    "window": ["window", "glass window"],
    "door": ["door", "front door", "wooden door"],
    "gate": ["gate", "metal gate"],
    "garage": ["garage door"],
    "balcony": ["balcony"],
    "pillar": ["pillar", "column"],
    "railing": ["railing", "handrail"],
    "roof_edge": ["roof", "eave"],
}

# Canonical labels the rest of the pipeline understands. Anything the detector
# returns that does not map into this set becomes `other` rather than inventing
# a label the cost engine has no rate for.
CANONICAL_LABELS = [
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
    "other",
]

# Reverse lookup: prompt phrase -> canonical label
PHRASE_TO_LABEL = {
    phrase: label for label, phrases in PROMPT_MAP.items() for phrase in phrases
}


def build_prompt() -> str:
    """GroundingDINO expects 'phrase one. phrase two. phrase three.'"""
    phrases = [p for phrases in PROMPT_MAP.values() for p in phrases]
    return ". ".join(phrases) + "."


# --- Overlay rendering ------------------------------------------------------
# BGR, since OpenCV. Distinct hues so labels are separable at a glance.

LABEL_COLORS = {
    "wall": (180, 190, 100),
    "window": (60, 120, 240),
    "door": (60, 200, 255),
    "balcony": (200, 120, 200),
    "pillar": (120, 220, 120),
    "parapet": (220, 180, 80),
    "railing": (100, 100, 240),
    "gate": (240, 200, 120),
    "garage": (200, 160, 240),
    "stairs": (90, 200, 200),
    "roof_edge": (140, 160, 200),
    "other": (160, 160, 160),
}
