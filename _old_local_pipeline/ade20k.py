"""ADE20K class -> facade component mapping.

OneFormer labels every pixel across 150 ADE20K classes. Only a handful matter
for a facade, and their names do not line up with construction vocabulary, so
this module is the translation layer.

The index numbers below are ADE20K's 0-based semantic ids as used by the
HuggingFace `shi-labs/oneformer_ade20k_*` checkpoints. They are resolved by name
at runtime from the model's own id2label rather than hardcoded, because the
0-based/1-based convention differs between ADE20K releases and getting it wrong
silently mislabels everything.
"""

from __future__ import annotations

# ADE20K class name (lowercased, first synonym) -> our canonical label.
# ADE20K separates 'wall' (interior/generic) from 'building' (exterior mass);
# on a facade photo both land on the same surface, so both map to wall and the
# postprocess merge collapses them.
ADE_TO_LABEL: dict[str, str] = {
    "wall": "wall",
    "building": "wall",
    "house": "wall",
    "skyscraper": "wall",
    "windowpane": "window",
    "window": "window",
    "door": "door",
    "double door": "door",
    "screen door": "door",
    "column": "pillar",
    "pillar": "pillar",
    "railing": "railing",
    "bannister": "railing",
    "fence": "railing",
    "stairs": "stairs",
    "step": "stairs",
    "stairway": "stairs",
    "roof": "roof_edge",
    "awning": "roof_edge",
    "canopy": "roof_edge",
    "balcony": "balcony",
    "gate": "gate",
}

# Classes that are definitively not part of the building. Anything here is
# dropped rather than mapped to `other`, so it never reaches the cost engine.
BACKGROUND_CLASSES = {
    "sky",
    "tree",
    "grass",
    "earth",
    "ground",
    "road",
    "sidewalk",
    "path",
    "plant",
    "flower",
    "car",
    "person",
    "water",
    "sea",
    "mountain",
    "rock",
    "sand",
    "field",
    "land",
    "hill",
    "palm",
    "bush",
    "shrub",
    "streetlight",
    "signboard",
    "pole",
    "truck",
    "van",
    "bicycle",
    "flowerpot",
    "pot",
}


def resolve(id2label: dict[int, str]) -> tuple[dict[int, str], set[int]]:
    """Build id -> canonical-label and the set of background ids.

    ADE20K names are comma-separated synonym lists ("windowpane, window").
    Each synonym is tried so a rename between checkpoint versions does not
    silently drop a class.
    """
    mapping: dict[int, str] = {}
    background: set[int] = set()

    for idx, raw in id2label.items():
        idx = int(idx)
        synonyms = [s.strip().lower() for s in str(raw).split(",")]

        if any(s in BACKGROUND_CLASSES for s in synonyms):
            background.add(idx)
            continue

        for s in synonyms:
            if s in ADE_TO_LABEL:
                mapping[idx] = ADE_TO_LABEL[s]
                break

    return mapping, background
