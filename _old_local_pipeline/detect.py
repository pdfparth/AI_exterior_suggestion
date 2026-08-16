"""GroundingDINO detection: text prompt -> labeled boxes.

Open-vocabulary detection is used here rather than a pretrained facade
segmentation network because the available facade datasets (CMP Facade,
ADE20K) do not carry the component vocabulary this domain needs - parapets,
pillars, gates and railings are not separable classes in those label sets.
Boundary precision is traded for semantic coverage and generalisation across
architectural styles; the boundaries are recovered afterwards by SAM.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image

from config import (
    BOX_THRESHOLD,
    GROUNDING_DINO_MODEL,
    PHRASE_TO_LABEL,
    TEXT_THRESHOLD,
    build_prompt,
)
from regions import Region

_model = None
_processor = None


def _load():
    """Lazy-load so importing this module stays cheap."""
    global _model, _processor
    if _model is None:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        print(f"[detect] loading {GROUNDING_DINO_MODEL} (first run downloads ~700MB)")
        _processor = AutoProcessor.from_pretrained(GROUNDING_DINO_MODEL)
        _model = AutoModelForZeroShotObjectDetection.from_pretrained(
            GROUNDING_DINO_MODEL
        ).eval()
        # torch defaults to physical-core count; on this box that leaves half the
        # CPU idle. Measured 117s -> ~35s per image by using every logical core.
        # Inference is dominated by the text encoder and cross-attention, so it
        # barely scales with image resolution - threads are the lever that works.
        torch.set_num_threads(os.cpu_count() or 4)
    return _model, _processor


def _canonical(phrase: str) -> str:
    """Map a detected phrase onto our label vocabulary.

    GroundingDINO returns whatever substring of the prompt it matched, which may
    be a fragment ('window frame') or a concatenation of adjacent prompt terms.
    Anything unrecognised becomes `other` rather than inventing a label the cost
    engine has no rate for.
    """
    p = phrase.lower().strip(" .")
    if p in PHRASE_TO_LABEL:
        return PHRASE_TO_LABEL[p]
    # Longest matching known phrase wins, so 'window' beats a stray 'wall' hit
    # inside a concatenated span.
    best, best_len = "other", 0
    for known, label in PHRASE_TO_LABEL.items():
        if known in p and len(known) > best_len:
            best, best_len = label, len(known)
    return best


def detect(image_bgr: np.ndarray) -> list[Region]:
    """Run open-vocabulary detection. Returns unfiltered Regions (boxes only)."""
    model, processor = _load()
    pil = Image.fromarray(image_bgr[:, :, ::-1])  # BGR -> RGB
    prompt = build_prompt()

    inputs = processor(images=pil, text=prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[pil.size[::-1]],  # (h, w)
    )[0]

    regions: list[Region] = []
    for i, (box, score, phrase) in enumerate(
        zip(results["boxes"], results["scores"], results["text_labels"])
    ):
        x1, y1, x2, y2 = (float(v) for v in box.tolist())
        regions.append(
            Region(
                id=f"d{i}",
                label=_canonical(phrase),
                box=(x1, y1, x2, y2),
                confidence=float(score),
                source="dino",
                notes=[f"phrase={phrase}"],
            )
        )
    return regions
