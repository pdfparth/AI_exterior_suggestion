"""FastAPI application. One endpoint per stage of the PDF's workflow.

The API is intentionally stage-by-stage rather than one upload-to-estimate
call. Each stage is slow (a Gemini round trip) and each has a decision point
the user is meant to control — review the detected regions before costing them,
pick materials before rendering, adjust rates and recalculate. Collapsing them
into one endpoint would remove exactly the interaction the requirements ask for.

    POST /api/projects              upload + quality gate  (5.1)
    POST /api/projects/{id}/analyse identify structure     (5.2, 5.5)
    POST /api/projects/{id}/design  apply materials        (5.3, 5.4)
    POST /api/projects/{id}/estimate quantities + cost     (5.6, 5.7)
    GET  /api/projects/{id}/report  PDF                    (5.8)
"""

from __future__ import annotations

import io
import os
import re
import traceback

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()  # read GEMINI_API_KEY from .env before anything imports the client

from . import estimate as estimate_mod  # noqa: E402
from . import composite, materials, redesign, report, store, vision  # noqa: E402
from .schemas import ScaleEstimate  # noqa: E402

app = FastAPI(title="AI Exterior Renovation Estimator", version="0.1.0")

MAX_UPLOAD_MB = 12
ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(os.path.dirname(HERE), "static")


# --- helpers ----------------------------------------------------------------


def _image_size(data: bytes) -> tuple[int, int]:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        return im.size


def _downscale(data: bytes, mime: str, max_edge: int = 1600) -> tuple[bytes, str]:
    """Cap the long edge before sending to Gemini.

    Large phone photos cost tokens and latency without improving perception at
    this task — the model is identifying wall planes, not reading fine print.
    Kept generous enough that window edges stay crisp.
    """
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        if max(im.size) <= max_edge:
            return data, mime
        im = im.convert("RGB")
        ratio = max_edge / max(im.size)
        im = im.resize((int(im.width * ratio), int(im.height * ratio)), Image.LANCZOS)
        out = io.BytesIO()
        im.save(out, format="JPEG", quality=88)
        return out.getvalue(), "image/jpeg"


def _project_or_404(pid: str):
    p = store.get(pid)
    if p is None:
        raise HTTPException(404, f"no project {pid}")
    return p


def _gemini_error(e: Exception) -> HTTPException:
    """Turn SDK failures into something a user can act on.

    The full text is always logged. A 429 in particular can mean several very
    different things - per-minute rate limit, exhausted daily free-tier quota,
    or a model not enabled for billing - and the caller can only react
    correctly if the message says which.
    """
    msg = str(e)
    print(f"\n[gemini] call failed: {msg}\n")

    if "GEMINI_API_KEY" in msg:
        return HTTPException(503, msg)

    low = msg.lower()

    if "quota" in low or "429" in low or "resource_exhausted" in low:
        # The API returns the retry delay it wants; pass it through so the user
        # knows whether this is a 30-second wait or a tomorrow problem.
        delay = re.search(r"retryDelay['\"]?:\s*['\"]?(\d+)s", msg)
        per_day = "per day" in low or "perday" in low or "daily" in low
        hint = (
            "You have used up today's free-tier quota for the image model. "
            "It resets on Google's daily schedule, or you can enable billing."
            if per_day else
            f"Rate limit hit. Wait {delay.group(1) if delay else '30-60'} seconds "
            "and try again."
        )
        return HTTPException(429, f"{hint} (Gemini said: {msg[:200]})")

    if "api key" in low or "401" in low or "permission" in low:
        return HTTPException(401, "Gemini rejected the API key. Check GEMINI_API_KEY in .env")

    # Google-side overload. Already retried with backoff inside the client, so
    # reaching here means it stayed busy - the user just needs to try again.
    if "503" in low or "unavailable" in low or "overloaded" in low or "500" in low:
        return HTTPException(
            503,
            "Gemini is temporarily overloaded (the model is in high demand). "
            "This is on Google's side, not your setup - wait a few seconds and "
            "try again.",
        )

    traceback.print_exc()
    return HTTPException(502, f"Gemini call failed: {msg[:300]}")


# --- 5.1 upload -------------------------------------------------------------


@app.post("/api/projects")
async def create_project(file: UploadFile):
    """Upload a photo, gate it for quality, return a project id."""
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty file")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"file larger than {MAX_UPLOAD_MB}MB")

    mime = file.content_type or "image/jpeg"
    if mime not in ALLOWED_MIME:
        raise HTTPException(415, f"unsupported type {mime}; use JPEG, PNG or WebP")

    try:
        w, h = _image_size(data)
    except Exception:
        raise HTTPException(400, "could not decode that file as an image")

    small, small_mime = _downscale(data, mime)

    try:
        quality = vision.check_quality(small, small_mime)
    except Exception as e:
        raise _gemini_error(e)

    if not quality.usable:
        # 5.1: reject and guide, rather than failing silently downstream.
        return {
            "accepted": False,
            "reason": quality.reason,
            "guidance": quality.guidance,
        }

    p = store.create(small, small_mime, w, h, file.filename or "upload")
    return {"accepted": True, "project_id": p.id, "reason": quality.reason,
            "width": w, "height": h}


# --- 5.2 + 5.5 analysis -----------------------------------------------------


@app.post("/api/projects/{pid}/analyse")
async def analyse(pid: str):
    """Identify surfaces and establish real-world scale."""
    p = _project_or_404(pid)
    try:
        a = vision.analyse(p.original_bytes, p.original_mime)
    except Exception as e:
        raise _gemini_error(e)

    p.analysis = a
    # Seed a starting design so the user sees a costed result immediately
    # instead of an empty form (5.3 lets them change all of it).
    p.selections = {
        r.id: materials.DEFAULT_DESIGN[r.label]
        for r in a.regions
        if r.label in materials.DEFAULT_DESIGN
    }
    return {
        "analysis": a.model_dump(),
        "selections": p.selections,
        "catalog": materials.catalog_payload(),
    }


class RegionEdit(BaseModel):
    """5.2 requires the user be able to correct detected regions."""

    id: str
    label: str | None = None
    polygon: list[list[float]] | None = None
    delete: bool = False


@app.post("/api/projects/{pid}/regions")
async def edit_regions(pid: str, edits: list[RegionEdit]):
    """Apply user corrections to the detected geometry."""
    p = _project_or_404(pid)
    if p.analysis is None:
        raise HTTPException(409, "analyse the project first")

    by_id = {r.id: r for r in p.analysis.regions}
    for e in edits:
        r = by_id.get(e.id)
        if r is None:
            continue
        if e.delete:
            p.analysis.regions = [x for x in p.analysis.regions if x.id != e.id]
            p.selections.pop(e.id, None)
            continue
        if e.label:
            r.label = e.label  # type: ignore[assignment]
            # The old material may not apply to the new label.
            if r.id in p.selections:
                if e.label not in materials.get(p.selections[r.id]).applicable_to:
                    p.selections.pop(r.id)
        if e.polygon:
            r.polygon = e.polygon
    return {"analysis": p.analysis.model_dump(), "selections": p.selections}


# --- 5.3 + 5.4 design and visualisation ------------------------------------


class DesignRequest(BaseModel):
    selections: dict[str, str]
    # "auto"   try Gemini, composite locally if it is unavailable (default)
    # "gemini" fail loudly instead of falling back
    # "local"  skip Gemini entirely - instant, free, deterministic
    engine: str = "auto"


@app.post("/api/projects/{pid}/design")
async def design(pid: str, req: DesignRequest):
    """Apply the chosen materials and render the redesigned photo."""
    p = _project_or_404(pid)
    if p.analysis is None:
        raise HTTPException(409, "analyse the project first")
    if not req.selections:
        raise HTTPException(400, "no materials selected")

    p.selections = req.selections

    # Try the generative path first - it is the one that produces a genuinely
    # photorealistic result. If it is unavailable (quota, outage, safety
    # block), fall back to local compositing rather than failing the request:
    # the demo must not die on the one call most likely to be rate-limited.
    engine, note = "gemini", ""

    if req.engine == "local":
        img, mime = composite.render(
            p.original_bytes, p.analysis, req.selections, materials.get
        )
        p.redesign_bytes, p.redesign_mime, p.redesign_engine = img, mime, "local"
        return {
            "ok": True,
            "engine": "local",
            "note": "Composited locally: materials applied to the detected "
                    "surfaces with the photo's original lighting preserved.",
            "image_url": f"/api/projects/{pid}/image/redesign",
        }

    try:
        img, mime = redesign.generate(
            p.original_bytes, p.original_mime, p.analysis, req.selections
        )
    except Exception as e:
        if req.engine == "gemini":
            raise _gemini_error(e)  # caller explicitly demanded Gemini

        detail = _gemini_error(e).detail
        print(f"[design] Gemini unavailable, compositing locally instead")
        try:
            img, mime = composite.render(
                p.original_bytes, p.analysis, req.selections, materials.get
            )
        except Exception as local_err:
            traceback.print_exc()
            raise HTTPException(
                502, f"Both renderers failed. Gemini: {detail} Local: {local_err}"
            )
        engine = "local"
        note = (
            "Gemini image generation was unavailable, so this preview was "
            "composited locally. Colours and textures are applied to the "
            "detected surfaces with the photo's original lighting preserved. "
            f"({detail})"
        )

    p.redesign_bytes = img
    p.redesign_mime = mime
    p.redesign_engine = engine
    return {
        "ok": True,
        "engine": engine,
        "note": note,
        "image_url": f"/api/projects/{pid}/image/redesign",
    }


@app.get("/api/projects/{pid}/image/{which}")
async def image(pid: str, which: str):
    p = _project_or_404(pid)
    if which == "original":
        return Response(p.original_bytes, media_type=p.original_mime)
    if which == "redesign":
        if not p.redesign_bytes:
            raise HTTPException(404, "no redesign generated yet")
        return Response(p.redesign_bytes, media_type=p.redesign_mime)
    raise HTTPException(404, "unknown image")


# --- 5.6 + 5.7 quantities and cost -----------------------------------------


class EstimateRequest(BaseModel):
    selections: dict[str, str] | None = None
    rate_overrides: dict[str, dict[str, float]] | None = None
    scale_override: dict | None = None


@app.post("/api/projects/{pid}/estimate")
async def make_estimate(pid: str, req: EstimateRequest):
    """Quantities and costs. Cheap and local — no AI call, so it recomputes
    instantly when the user edits a rate."""
    p = _project_or_404(pid)
    if p.analysis is None:
        raise HTTPException(409, "analyse the project first")

    if req.selections is not None:
        p.selections = req.selections
    if req.rate_overrides is not None:
        p.rate_overrides = req.rate_overrides
    if req.scale_override is not None:
        p.scale_override = req.scale_override

    scale = None
    if p.scale_override:
        base = p.analysis.scale.model_dump()
        base.update(p.scale_override)
        scale = ScaleEstimate.model_validate(base)

    est = estimate_mod.compute(
        p.analysis,
        p.selections,
        p.image_w,
        p.image_h,
        rate_overrides=p.rate_overrides,
        scale_override=scale,
    )
    return est.model_dump()


# --- 5.8 report -------------------------------------------------------------


@app.get("/api/projects/{pid}/report")
async def download_report(pid: str):
    p = _project_or_404(pid)
    if p.analysis is None:
        raise HTTPException(409, "analyse the project first")

    scale = None
    if p.scale_override:
        base = p.analysis.scale.model_dump()
        base.update(p.scale_override)
        scale = ScaleEstimate.model_validate(base)

    est = estimate_mod.compute(
        p.analysis, p.selections, p.image_w, p.image_h,
        rate_overrides=p.rate_overrides, scale_override=scale,
    )
    pdf = report.build(
        p.original_bytes, p.redesign_bytes, p.analysis, est,
        redesign_engine=p.redesign_engine,
    )
    return Response(
        pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="renovation_estimate_{pid}.pdf"'},
    )


# --- misc -------------------------------------------------------------------


@app.get("/api/catalog")
async def catalog():
    return materials.catalog_payload()


@app.get("/api/projects")
async def list_projects():
    return [p.summary() for p in store.all_projects()]


SAMPLES_DIR = os.path.join(os.path.dirname(HERE), "samples")
SAMPLE_EXT = (".jpg", ".jpeg", ".png", ".webp")


@app.get("/api/samples")
async def list_samples():
    """Sample photos, so the demo works without the user hunting for an image."""
    if not os.path.isdir(SAMPLES_DIR):
        return []
    return sorted(
        f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith(SAMPLE_EXT)
    )


@app.get("/api/samples/{name}")
async def get_sample(name: str):
    # Serve by basename only; never let a path fragment escape the directory.
    safe = os.path.basename(name)
    path = os.path.join(SAMPLES_DIR, safe)
    if not os.path.isfile(path) or not safe.lower().endswith(SAMPLE_EXT):
        raise HTTPException(404, "no such sample")
    return FileResponse(path)


@app.get("/api/health")
async def health():
    # Report the model actually in use once fallback has resolved one, so the
    # UI never claims a model the requests are not going to.
    return {
        "ok": True,
        "gemini_key_present": bool(os.environ.get("GEMINI_API_KEY", "").strip()),
        "vision_model": vision._resolved_vision or vision.VISION_MODEL,
        "image_model": redesign._resolved_image or redesign.IMAGE_MODEL,
        "resolved": bool(vision._resolved_vision),
    }


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
