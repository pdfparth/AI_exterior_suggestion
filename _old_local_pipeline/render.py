"""Stage 2 CLI: regions + material selections -> redesigned image.

    # list what can go where
    python render.py out/h1_regions.json --list

    # auto-assign a sensible design and render
    python render.py out/h1_regions.json --auto

    # explicit control
    python render.py out/h1_regions.json --set r0=stone_slate --set r3=paint_ivory

    # optional diffusion pass for realism (slow on CPU)
    python render.py out/h1_regions.json --auto --diffusion

Writes the redesigned image and a before/after comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2

import apply
import materials
from regions import Region, load_regions

# Default design used by --auto. Chosen to read clearly in a demo: a light
# textured base with a stone plinth and dark trim.
AUTO_DESIGN = {
    "wall": "texture_sand",
    "parapet": "paint_ivory",
    "pillar": "stone_slate",
    "balcony": "paint_ivory",
    "railing": "rail_ms",
    "stairs": "tile_granite",
    "gate": "wood_panel",
    "garage": "wood_panel",
}


def parse_set(pairs: list[str]) -> dict[str, str]:
    out = {}
    for p in pairs or []:
        if "=" not in p:
            sys.exit(f"error: --set expects region=material, got '{p}'")
        rid, mid = p.split("=", 1)
        out[rid.strip()] = mid.strip()
    return out


def auto_select(regions: list[Region]) -> dict[str, str]:
    sel = {}
    for r in regions:
        mid = AUTO_DESIGN.get(r.label)
        if mid and r.label in materials.get(mid).applicable_to:
            sel[r.id] = mid
    return sel


def list_options(regions: list[Region]) -> None:
    print(f"\n{'region':<7}{'label':<10}{'px area':>12}  applicable materials")
    print("-" * 78)
    for r in regions:
        opts = [m.id for m in materials.for_label(r.label)]
        print(
            f"{r.id:<7}{r.label:<10}{r.px_area:>12,.0f}  "
            f"{', '.join(opts) if opts else '(none catalogued)'}"
        )
    print(f"\n{'material':<18}{'category':<10}{'unit':<7}{'rate':>8}  applies to")
    print("-" * 78)
    for m in materials.CATALOG.values():
        print(
            f"{m.id:<18}{m.category:<10}{m.unit:<7}{m.material_rate:>8,.0f}  "
            f"{', '.join(m.applicable_to)}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 2: apply materials")
    ap.add_argument("regions_json")
    ap.add_argument("--image", help="override source image path from regions.json")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--list", action="store_true", help="show regions and materials")
    ap.add_argument("--auto", action="store_true", help="auto-assign a design")
    ap.add_argument("--set", action="append", metavar="REGION=MATERIAL")
    ap.add_argument(
        "--shading",
        type=float,
        default=0.75,
        help="0..1, how much of the original lighting to keep (default 0.75)",
    )
    ap.add_argument("--diffusion", action="store_true", help="SD1.5 refinement pass")
    ap.add_argument("--steps", type=int, default=18, help="diffusion steps")
    ap.add_argument("--strength", type=float, default=0.35, help="diffusion strength")
    args = ap.parse_args()

    regions, meta = load_regions(args.regions_json)
    if args.list:
        list_options(regions)
        return

    img_path = args.image or meta.get("source_image")
    if not img_path or not os.path.exists(img_path):
        sys.exit(f"error: source image not found ({img_path}); pass --image")
    image = cv2.imread(img_path)
    if image is None:
        sys.exit(f"error: could not decode {img_path}")

    selections = auto_select(regions) if args.auto else {}
    selections.update(parse_set(args.set))
    if not selections:
        sys.exit("error: nothing selected. use --auto, --set, or --list")

    by_id = {r.id: r for r in regions}
    print(f"[render] {len(selections)} region(s) assigned:")
    for rid, mid in selections.items():
        lbl = by_id[rid].label if rid in by_id else "?"
        name = materials.CATALOG[mid].name if mid in materials.CATALOG else "UNKNOWN"
        print(f"  {rid:<6}{lbl:<10}-> {name}")

    out, notes = apply.render_design(
        image, regions, selections, materials.get, shading_strength=args.shading
    )
    for n in notes:
        print(f"  ! {n}")

    stem = os.path.splitext(os.path.basename(args.regions_json))[0].replace(
        "_regions", ""
    )
    os.makedirs(args.outdir, exist_ok=True)

    if args.diffusion:
        import inpaint

        print(f"[render] diffusion pass ({args.steps} steps) - slow on CPU")
        out = inpaint.refine(
            image, out, regions, selections, materials.get,
            steps=args.steps, strength=args.strength,
        )

    design_path = os.path.join(args.outdir, f"{stem}_redesign.png")
    compare_path = os.path.join(args.outdir, f"{stem}_compare.png")
    sel_path = os.path.join(args.outdir, f"{stem}_selections.json")

    cv2.imwrite(design_path, out)
    cv2.imwrite(compare_path, apply.side_by_side(image, out))
    with open(sel_path, "w") as f:
        json.dump(
            {
                "source_image": img_path,
                "regions_json": os.path.abspath(args.regions_json),
                "selections": selections,
                "shading_strength": args.shading,
                "diffusion": bool(args.diffusion),
            },
            f,
            indent=2,
        )

    print(f"\n  redesign -> {design_path}\n  compare  -> {compare_path}")
    print(f"  choices  -> {sel_path}")


if __name__ == "__main__":
    main()
