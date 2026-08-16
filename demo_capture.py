"""Capture the full v2 UI with Gemini mocked out — no API key required.

    pip install playwright && playwright install chromium
    python demo_capture.py

Writes PNGs to docs/v2-screens-raw/. Regenerates the walkthrough in
docs/v2-demo.html after you change the UI.


Every Gemini call is replaced with a realistic canned response, so the whole
flow runs offline and deterministically. The redesign uses the real local
compositor, so the "after" image is genuinely produced by the app.
"""

import threading, time, uvicorn
from unittest.mock import patch
from playwright.sync_api import sync_playwright

from app.schemas import Analysis, Region, ScaleEstimate, QualityCheck

import os
SP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "v2-screens-raw")
PORT = 8110

# A survey shaped like a real Gemini response for the h1 sample: two wall
# planes, several openings, a parapet band, railing and a pillar.
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
    import os
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
            pg = b.new_page(viewport={"width": 1560, "height": 1000},
                            device_scale_factor=2)
            pg.on("pageerror", lambda e: errors.append(str(e)))

            # ---- 01 landing -------------------------------------------
            pg.goto(f"http://127.0.0.1:{PORT}/v2", wait_until="networkidle")
            pg.wait_for_timeout(1200)
            shot(pg, "01_landing")

            # ---- 02 dropzone hover ------------------------------------
            pg.hover("#drop"); pg.wait_for_timeout(500)
            shot(pg, "02_dropzone_hover")

            # ---- 03 upload in progress --------------------------------
            pg.click(".sample")
            pg.wait_for_timeout(260)
            shot(pg, "03_uploading")

            # ---- 04 survey --------------------------------------------
            pg.wait_for_selector("#survey-body:not([hidden])", timeout=25000)
            pg.wait_for_timeout(2600)
            pg.eval_on_selector("#p-survey", "e=>e.scrollIntoView()")
            pg.wait_for_timeout(900)
            shot(pg, "04_survey")

            # ---- 05 region hover highlight ----------------------------
            pg.hover(".region:nth-child(1)")
            pg.wait_for_timeout(600)
            shot(pg, "05_region_hover")

            # ---- 06 scale panel close-up ------------------------------
            box = pg.query_selector("#scale-card").bounding_box()
            pg.screenshot(path=f"{SP}/06_scale_panel.png", clip={
                "x": box["x"]-14, "y": box["y"]-14,
                "width": box["width"]+28, "height": box["height"]+28})
            print("  captured 06_scale_panel")

            # ---- 07 materials -----------------------------------------
            pg.eval_on_selector("#p-design", "e=>e.scrollIntoView()")
            pg.wait_for_timeout(1100)
            shot(pg, "07_materials")

            # ---- 08 material card hover -------------------------------
            cards = pg.query_selector_all(".mat")
            cards[5].hover(); pg.wait_for_timeout(500)
            shot(pg, "08_material_hover")
            cards[5].click(); pg.wait_for_timeout(900)
            shot(pg, "09_material_selected")

            # ---- 10 render running ------------------------------------
            pg.eval_on_selector("#p-visual", "e=>e.scrollIntoView()")
            pg.wait_for_timeout(500)
            pg.click("#btn-render")
            pg.wait_for_timeout(420)
            shot(pg, "10_rendering")

            # ---- 11 compare, mid reveal -------------------------------
            pg.wait_for_selector("#compare-wrap:not([hidden])", timeout=45000)
            pg.wait_for_timeout(650)
            pg.eval_on_selector("#p-visual", "e=>e.scrollIntoView()")
            shot(pg, "11_compare_revealing")

            # ---- 12 compare settled -----------------------------------
            pg.wait_for_timeout(2200)
            shot(pg, "12_compare_settled")

            # ---- 13/14 dragged both ways ------------------------------
            cb = pg.query_selector("#compare").bounding_box()
            cy = cb["y"] + cb["height"]*.5
            pg.mouse.move(cb["x"]+cb["width"]*.5, cy)
            pg.mouse.down()
            pg.mouse.move(cb["x"]+cb["width"]*.18, cy, steps=20)
            pg.wait_for_timeout(400)
            shot(pg, "13_compare_mostly_after")
            pg.mouse.move(cb["x"]+cb["width"]*.84, cy, steps=24)
            pg.mouse.up()
            pg.wait_for_timeout(400)
            shot(pg, "14_compare_mostly_before")

            # ---- 15 estimate ------------------------------------------
            pg.eval_on_selector("#p-estimate", "e=>e.scrollIntoView()")
            pg.wait_for_timeout(1500)
            shot(pg, "15_estimate")

            # ---- 16 rate editor open ----------------------------------
            pg.click("details.rates > summary")
            pg.wait_for_timeout(800)
            pg.eval_on_selector("#p-estimate", "e=>e.scrollIntoView()")
            pg.wait_for_timeout(500)
            shot(pg, "16_rate_editor")

            # ---- 17 rate changed, total recalculated ------------------
            inp = pg.query_selector_all(".rate-in")[0]
            before = pg.inner_text("#hero-total")
            inp.fill("980"); inp.dispatch_event("change")
            pg.wait_for_timeout(1700)
            after = pg.inner_text("#hero-total")
            shot(pg, "17_rate_changed")
            print(f"     rate edit: {before} -> {after}")

            # ---- 18 region excluded -----------------------------------
            pg.eval_on_selector("#p-survey", "e=>e.scrollIntoView()")
            pg.wait_for_timeout(600)
            pg.query_selector_all(".region")[0].click()
            pg.wait_for_timeout(1500)
            shot(pg, "18_region_excluded")

            # ---- 19 full page -----------------------------------------
            pg.eval_on_selector(".region", "e=>e.click()")   # restore
            pg.wait_for_timeout(1200)
            pg.screenshot(path=f"{SP}/19_full_page.png", full_page=True)
            print("  captured 19_full_page")

            # ---- 20 tablet --------------------------------------------
            tab = b.new_page(viewport={"width": 834, "height": 1100}, device_scale_factor=2)
            tab.goto(f"http://127.0.0.1:{PORT}/v2", wait_until="networkidle")
            tab.click(".sample")
            tab.wait_for_selector("#survey-body:not([hidden])", timeout=25000)
            tab.wait_for_timeout(2600)
            tab.screenshot(path=f"{SP}/20_tablet.png")
            print("  captured 20_tablet")
            tab.close()

            # ---- 21 phone ---------------------------------------------
            ph = b.new_page(viewport={"width": 400, "height": 900}, device_scale_factor=3)
            ph.goto(f"http://127.0.0.1:{PORT}/v2", wait_until="networkidle")
            ph.wait_for_timeout(900)
            ph.screenshot(path=f"{SP}/21_phone_landing.png")
            ph.click(".sample")
            ph.wait_for_selector("#survey-body:not([hidden])", timeout=25000)
            ph.wait_for_timeout(2600)
            ph.screenshot(path=f"{SP}/22_phone_survey.png")
            print("  captured 21/22 phone")
            ph.close()

            b.close()
        srv.should_exit = True

    # ---- 23 rejection state (separate patch) ----------------------------
    time.sleep(1)
    with patch("app.vision.check_quality", q_reject):
        import app.main
        cfg = uvicorn.Config(app.main.app, host="127.0.0.1", port=PORT+1, log_level="error")
        srv2 = uvicorn.Server(cfg)
        threading.Thread(target=srv2.run, daemon=True).start()
        time.sleep(3)
        with sync_playwright() as pw:
            b = pw.chromium.launch()
            pg = b.new_page(viewport={"width": 1560, "height": 1000}, device_scale_factor=2)
            pg.on("pageerror", lambda e: errors.append(str(e)))
            pg.goto(f"http://127.0.0.1:{PORT+1}/v2", wait_until="networkidle")
            pg.click(".sample")
            pg.wait_for_selector(".note.err", timeout=25000)
            pg.wait_for_timeout(900)
            shot(pg, "23_photo_rejected")
            b.close()
        srv2.should_exit = True

    print("\nJS ERRORS:", errors or "none")


main()
