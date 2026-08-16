"""Stage 1 CLI: image -> labeled facade regions.

    python segment.py house.jpg
    python segment.py house.jpg --fast          # MobileSAM instead of SAM 2.1 L
    python segment.py house.jpg --no-dino       # semantic only
    python segment.py house.jpg --no-sam        # skip edge refinement

Three models, each doing what it is actually good at:

  OneFormer (ADE20K)  walls, windows, doors, columns, railings, roof - full
                      per-pixel coverage. Walls are not objects, so this is the
                      only component that finds them reliably.
  GroundingDINO       gates, balconies, garage doors - absent or unreliable in
                      ADE20K, reachable by text prompt.
  SAM 2.1 Large       boundary refinement on crisp objects, where a tight mask
                      measurably changes the area that reaches costing.

Writes regions.json (editable geometry) and an overlay PNG.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2

import config
import overlay
import postprocess
from regions import save_regions

MAX_EDGE = 1024


def load_image(path: str):
    if not os.path.exists(path):
        sys.exit(f"error: no such file: {path}")
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"error: could not decode as an image: {path}")
    return img


def downscale(img, max_edge: int = MAX_EDGE):
    """Cap the long edge for inference speed. Returns (image, scale_back)."""
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_edge:
        return img, 1.0
    f = max_edge / longest
    return cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA), 1 / f


def rescale_regions(regions, factor: float):
    """Map geometry from inference resolution back to original pixels."""
    if factor == 1.0:
        return regions
    for r in regions:
        r.box = tuple(v * factor for v in r.box)
        r.polygon = [[x * factor, y * factor] for x, y in r.polygon]
        r.px_area *= factor * factor
    return regions


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 1: facade segmentation")
    ap.add_argument("image")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--no-sam", action="store_true", help="skip edge refinement")
    ap.add_argument("--no-dino", action="store_true", help="skip open-vocab detection")
    ap.add_argument(
        "--fast",
        action="store_true",
        help="MobileSAM instead of SAM 2.1 Large (~10x faster, rougher edges)",
    )
    ap.add_argument("--box-threshold", type=float, default=config.BOX_THRESHOLD)
    ap.add_argument("--text-threshold", type=float, default=config.TEXT_THRESHOLD)
    ap.add_argument("--max-edge", type=int, default=MAX_EDGE)
    args = ap.parse_args()

    config.BOX_THRESHOLD = args.box_threshold
    config.TEXT_THRESHOLD = args.text_threshold
    sam_model = config.MOBILE_SAM_MODEL if args.fast else config.SAM_MODEL

    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image))[0]

    original = load_image(args.image)
    oh, ow = original.shape[:2]
    small, scale_back = downscale(original, args.max_edge)
    print(f"[main] {args.image}  {ow}x{oh} -> inference at {small.shape[1]}x{small.shape[0]}")

    regions = []
    timings = {}

    # --- 1. envelope: building vs sky/vegetation/road ------------------------
    import semantic

    t0 = time.time()
    sem_regions, sem_map = semantic.segment(small)
    envelope = semantic.building_envelope(sem_map)
    timings["semantic"] = round(time.time() - t0, 2)
    regions.extend(sem_regions)
    build_coverage = semantic.coverage(sem_map)
    print(
        f"[main] envelope: {build_coverage:.0%} of frame is building, "
        f"{len(sem_regions)} semantic region(s) in {timings['semantic']}s"
    )

    # --- 2. components: windows, doors, railings, ... ------------------------
    if not args.no_dino:
        import detect

        t0 = time.time()
        det_regions = detect.detect(small)
        timings["detect"] = round(time.time() - t0, 2)
        regions.extend(det_regions)
        print(f"[main] detector: {len(det_regions)} components in {timings['detect']}s")

    # --- 3. refine component edges ------------------------------------------
    if not args.no_sam and regions:
        import refine

        t0 = time.time()
        regions = refine.refine(small, regions, model_name=sam_model)
        timings["refine"] = round(time.time() - t0, 2)
        n = sum(1 for r in regions if "sam" in r.source)
        print(f"[main] refined {n} regions with {sam_model} in {timings['refine']}s")

    # --- 4. wall = envelope - components ------------------------------------
    # Done after refinement so the subtraction uses the tightest available
    # component geometry; a loose box here would eat wall area that is really
    # paintable surface.
    walls = semantic.wall_regions(envelope, regions)
    regions.extend(walls)
    wall_px = sum(w.px_area for w in walls)
    print(
        f"[main] derived {len(walls)} wall surface(s), "
        f"{wall_px / max(1, envelope.sum()):.0%} of envelope"
    )

    regions = rescale_regions(regions, scale_back)
    regions, warnings = postprocess.run(regions, ow, oh)
    print(f"[main] {len(regions)} regions after postprocessing")

    if build_coverage is not None and build_coverage < 0.20:
        warnings.append(
            f"building occupies only {build_coverage:.0%} of the frame - "
            "a closer or less obstructed photo would estimate more accurately"
        )

    meta = {
        "source_image": os.path.abspath(args.image),
        "image_size": {"width": ow, "height": oh},
        "models": {
            "semantic": config.ONEFORMER_MODEL,
            "detector": None if args.no_dino else config.GROUNDING_DINO_MODEL,
            "refiner": None if args.no_sam else sam_model,
        },
        "thresholds": {"box": args.box_threshold, "text": args.text_threshold},
        "building_coverage": round(build_coverage, 4) if build_coverage else None,
        "timings_sec": timings,
        "warnings": warnings,
    }

    json_path = os.path.join(args.outdir, f"{stem}_regions.json")
    png_path = os.path.join(args.outdir, f"{stem}_overlay.png")
    save_regions(json_path, regions, meta)
    overlay.save(png_path, original, regions)

    _summary(regions, ow, oh, warnings)
    print(f"\n  regions -> {json_path}\n  overlay -> {png_path}")


def _summary(regions, w: int, h: int, warnings: list[str]) -> None:
    print(f"\n{'label':<12}{'count':>6}{'px area':>14}{'% frame':>10}  source")
    print("-" * 56)
    agg: dict[str, list] = {}
    for r in regions:
        agg.setdefault(r.label, []).append(r)
    for label, rs in sorted(agg.items(), key=lambda kv: -sum(x.px_area for x in kv[1])):
        total = sum(x.px_area for x in rs)
        srcs = sorted({x.source for x in rs})
        print(
            f"{label:<12}{len(rs):>6}{total:>14,.0f}{total / (w * h):>9.1%}  "
            f"{','.join(srcs)}"
        )

    if warnings:
        print("\nwarnings:")
        for wmsg in warnings:
            print(f"  ! {wmsg}")


if __name__ == "__main__":
    main()
