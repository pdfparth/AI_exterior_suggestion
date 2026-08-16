"""Capture the full v1 UI with Gemini mocked out — no API key required.

    pip install playwright && playwright install chromium
    python demo_capture_v1.py

Writes PNGs to docs/v1-screens-raw/. The companion of demo_capture.py, which
does the same for v2.

Every Gemini call is replaced with a canned response so the flow runs offline
and deterministically. The redesign uses the real local compositor, so the
"after" image is genuine output from the app.
"""

import os
import threading
import time

import uvicorn
from unittest.mock import patch
from playwright.sync_api import sync_playwright

from app.schemas import Analysis, QualityCheck, Region, ScaleEstimate

SP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "v1-screens-raw")
PORT = 8130

# Same canned survey as the v2 capture, so the two walkthroughs are directly
# comparable — only the interface differs.
SURVEY = Analysis(
    regions=[
        Region(id="r1", label="wall", confidence=.94, note="main render, front plane",
               polygon=[[.055,.295],[.615,.243],[.618,.862],[.058,.878]]),
        Region(id="r2", label="wall", confidence=.87, note="right wing, set back",
               polygon=[[.620,.246],[.945,.318],[.945,.845],[.620,.862]]),
        Region(id="r3", label="parapet", confidence=.74, note="flat roofline band",
               polygon=[[.050,.207],[.950,.243],[.950,.318],[.050,.295]]),
        Region(id="r4", label="window", confidence=.92, note="upper left, twin sash",
               polygon=[[.128,.372],[.263,.360],[.265,.470],[.130,.482]]),
        Region(id="r5", label="window", confidence=.90, note="upper right",
               polygon=[[.404,.348],[.520,.339],[.522,.442],[.406,.451]]),
        Region(id="r6", label="window", confidence=.85, note="right wing, horizontal",
               polygon=[[.700,.398],[.878,.424],[.878,.487],[.700,.464]]),
        Region(id="r7", label="door", confidence=.95, note="timber entrance door",
               polygon=[[.296,.556],[.390,.549],[.392,.842],[.298,.850]]),
        Region(id="r8", label="pillar", confidence=.79, note="porch column, left",
               polygon=[[.243,.548],[.281,.545],[.283,.845],[.245,.849]]),
        Region(id="r9", label="railing", confidence=.71, note="first-floor balustrade",
               polygon=[[.636,.520],[.905,.556],[.905,.622],[.636,.590]]),
        Region(id="r10", label="stairs", confidence=.83, note="entrance steps",
               polygon=[[.262,.848],[.436,.842],[.452,.900],[.246,.906]]),
    ],
    scale=ScaleEstimate(
        reference_object="entrance door",
        reference_real_feet=6.75,
        reference_px_fraction=.294,
        building_width_ft=37.0,
        building_height_ft=21.0,
        confidence=.81,
        reasoning="The timber entrance door spans roughly 29% of the image height. "
                  "At a standard 6.75 ft leaf that puts the visible facade near 21 ft "
                  "tall, and the frame width scales to about 37 ft across.",
    ),
    storeys=2,
    style_note="Two-storey rendered house, flat parapet roofline with a set-back right wing.",
    warnings=[
        "The right wing is foreshortened by the camera angle; its area is likely "
        "under-measured by 10-15%.",
        "Planting along the base hides the lower edge of the front wall.",
    ],
)


def q_ok(*a, **k):
    return QualityCheck(usable=True, is_building_exterior=True,
        reason="Clear two-storey facade, evenly lit and mostly unobstructed.",
        guidance="")


def q_reject(*a, **k):
    return QualityCheck(usable=False, is_building_exterior=False,
        reason="This looks like an indoor room, not a building exterior.",
        guidance="Step outside and photograph the front of the house from across "
                 "the street, with the whole facade in frame.")


def shot(pg, name):
    pg.screenshot(path=f"{SP}/{name}.png")
    print(f"  captured {name}")


def main():
    os.makedirs(SP, exist_ok=True)
    errors = []

    with patch("app.vision.check_quality", q_ok), \
         patch("app.vision.analyse", lambda *a, **k: SURVEY), \
         patch("app.redesign.generate",
               side_effect=Exception("429 RESOURCE_EXHAUSTED quota exceeded")):

        from app.main import app
        cfg = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="error")
        srv = uvicorn.Server(cfg)
        threading.Thread(target=srv.run, daemon=True).start()
        time.sleep(3)

        with sync_playwright() as pw:
            b = pw.chromium.launch()
            pg = b.new_page(viewport={"width": 1400, "height": 1000},
                            device_scale_factor=2)
            pg.on("pageerror", lambda e: errors.append(str(e)))

            # ---- 01 landing ------------------------------------------------
            pg.goto(f"http://127.0.0.1:{PORT}/", wait_until="networkidle")
            pg.wait_for_timeout(1100)
            shot(pg, "01_landing")

            # ---- 02 upload running ----------------------------------------
            pg.click(".samples img")
            pg.wait_for_timeout(240)
            shot(pg, "02_uploading")

            # ---- 03 analysing ---------------------------------------------
            pg.wait_for_timeout(600)
            shot(pg, "03_analysing")

            # ---- 04 detected surfaces -------------------------------------
            pg.wait_for_selector("#region-list .region", timeout=25000)
            pg.wait_for_timeout(1400)
            pg.eval_on_selector("#stage-analyse", "e=>e.scrollIntoView()")
            pg.wait_for_timeout(800)
            shot(pg, "04_surfaces")

            # ---- 05 overlay close-up --------------------------------------
            box = pg.query_selector(".canvas-wrap").bounding_box()
            pg.screenshot(path=f"{SP}/05_overlay.png", clip={
                "x": box["x"]-8, "y": box["y"]-8,
                "width": box["width"]+16, "height": box["height"]+16})
            print("  captured 05_overlay")

            # ---- 06 scale + region list -----------------------------------
            sb = pg.query_selector("#scale-box").bounding_box()
            rl = pg.query_selector("#region-list").bounding_box()
            pg.screenshot(path=f"{SP}/06_scale_regions.png", clip={
                "x": sb["x"]-10, "y": sb["y"]-10,
                "width": max(sb["width"], rl["width"])+20,
                "height": (rl["y"]+rl["height"]) - sb["y"] + 20})
            print("  captured 06_scale_regions")

            # ---- 07 materials ---------------------------------------------
            pg.eval_on_selector("#stage-design", "e=>e.scrollIntoView()")
            pg.wait_for_timeout(900)
            shot(pg, "07_materials")

            # ---- 08 material chips close-up -------------------------------
            mp = pg.query_selector("#material-picker").bounding_box()
            pg.screenshot(path=f"{SP}/08_material_chips.png", clip={
                "x": mp["x"]-10, "y": mp["y"]-10,
                "width": mp["width"]+20, "height": min(mp["height"]+20, 620)})
            print("  captured 08_material_chips")

            # ---- 09 rendering ---------------------------------------------
            pg.click("#btn-render")
            pg.wait_for_timeout(420)
            shot(pg, "09_rendering")

            # ---- 10 before / after ----------------------------------------
            pg.wait_for_selector("#compare:not(.hidden)", timeout=45000)
            pg.wait_for_timeout(2000)
            pg.eval_on_selector("#stage-design", "e=>e.scrollIntoView()")
            pg.wait_for_timeout(700)
            shot(pg, "10_before_after")

            # ---- 11 comparison close-up -----------------------------------
            cg = pg.query_selector(".compare-grid").bounding_box()
            pg.screenshot(path=f"{SP}/11_compare_closeup.png", clip={
                "x": cg["x"]-8, "y": cg["y"]-8,
                "width": cg["width"]+16, "height": cg["height"]+16})
            print("  captured 11_compare_closeup")

            # ---- 12 estimate ----------------------------------------------
            pg.eval_on_selector("#stage-estimate", "e=>e.scrollIntoView()")
            pg.wait_for_timeout(1200)
            shot(pg, "12_estimate")

            # ---- 13 rate editor -------------------------------------------
            summary = pg.query_selector("details.rate-editor > summary")
            if summary:
                summary.click()
                pg.wait_for_timeout(700)
                pg.eval_on_selector("#stage-estimate", "e=>e.scrollIntoView()")
                pg.wait_for_timeout(500)
                shot(pg, "13_rate_editor")

                inp = pg.query_selector_all(".rate-input")
                # skip the two scale inputs; the material rates come after
                target = [i for i in inp if i.get_attribute("data-mat")]
                if target:
                    before = pg.inner_text("#estimate-out .headline .big")
                    target[0].fill("980")
                    target[0].dispatch_event("change")
                    pg.wait_for_timeout(1500)
                    after = pg.inner_text("#estimate-out .headline .big")
                    shot(pg, "14_rate_changed")
                    print(f"     rate edit: {before} -> {after}")

            # ---- 15 excluding a surface ------------------------------------
            pg.eval_on_selector("#stage-analyse", "e=>e.scrollIntoView()")
            pg.wait_for_timeout(600)
            cb = pg.query_selector("#region-list .region input[type=checkbox]")
            if cb:
                cb.uncheck()
                pg.wait_for_timeout(1400)
                shot(pg, "15_surface_excluded")
                cb.check()
                pg.wait_for_timeout(1200)

            # ---- 16 full page ----------------------------------------------
            pg.screenshot(path=f"{SP}/16_full_page.png", full_page=True)
            print("  captured 16_full_page")

            # ---- 17 tablet --------------------------------------------------
            tab = b.new_page(viewport={"width": 834, "height": 1100}, device_scale_factor=2)
            tab.goto(f"http://127.0.0.1:{PORT}/", wait_until="networkidle")
            tab.click(".samples img")
            tab.wait_for_selector("#region-list .region", timeout=25000)
            tab.wait_for_timeout(2000)
            tab.screenshot(path=f"{SP}/17_tablet.png")
            print("  captured 17_tablet")
            tab.close()

            # ---- 18 phone ---------------------------------------------------
            ph = b.new_page(viewport={"width": 400, "height": 900}, device_scale_factor=3)
            ph.goto(f"http://127.0.0.1:{PORT}/", wait_until="networkidle")
            ph.wait_for_timeout(800)
            ph.screenshot(path=f"{SP}/18_phone_landing.png")
            ph.click(".samples img")
            ph.wait_for_selector("#region-list .region", timeout=25000)
            ph.wait_for_timeout(2000)
            ph.screenshot(path=f"{SP}/19_phone_surfaces.png")
            print("  captured 18/19 phone")
            ph.close()

            b.close()
        srv.should_exit = True

    # ---- 20 rejection state ------------------------------------------------
    time.sleep(1)
    with patch("app.vision.check_quality", q_reject):
        import app.main
        cfg = uvicorn.Config(app.main.app, host="127.0.0.1", port=PORT + 1, log_level="error")
        srv2 = uvicorn.Server(cfg)
        threading.Thread(target=srv2.run, daemon=True).start()
        time.sleep(3)
        with sync_playwright() as pw:
            b = pw.chromium.launch()
            pg = b.new_page(viewport={"width": 1400, "height": 1000}, device_scale_factor=2)
            pg.on("pageerror", lambda e: errors.append(str(e)))
            pg.goto(f"http://127.0.0.1:{PORT + 1}/", wait_until="networkidle")
            pg.click(".samples img")
            pg.wait_for_selector(".msg.err", timeout=25000)
            pg.wait_for_timeout(800)
            shot(pg, "20_photo_rejected")
            b.close()
        srv2.should_exit = True

    print("\nJS ERRORS:", errors or "none")


if __name__ == "__main__":
    main()
