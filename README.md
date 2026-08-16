# AI-Based Exterior House Renovation & Cost Estimation

A prototype that takes one photograph of a house, identifies its exterior
surfaces, applies chosen materials to produce a photorealistic redesign, and
returns a transparent cost estimate.

Built as a 2-day assessment prototype. The emphasis is on the AI/ML approach
and the estimation logic, not on production concerns.

---

## The core problem, and how it is solved

The requirement that actually determines whether this system works is not
segmentation. It is **monocular metric scale**: turning pixels into square feet
from a single photo with no reference measurement.

Everything else is downstream of that. You can identify surfaces perfectly and
still produce a cost estimate that is wrong by a factor of two if the scale is
wrong — because scale enters the calculation as a multiplier.

### The approach I started with, and why I moved off it

My first implementation was a conventional CV stack, kept in
`_old_local_pipeline/`:

| Model | Job |
|---|---|
| OneFormer (ADE20K) | building envelope vs sky/vegetation/road |
| GroundingDINO | components by text prompt (gate, balcony, garage) |
| SAM 2.1 Large | boundary refinement |

It worked, and the wall-by-subtraction idea in it was sound — derive wall area
as `envelope − detected components`, so openings are removed geometrically
rather than guessed at.

But measuring it on real facade photos showed three problems:

1. **~80 seconds per image on CPU**, plus ~2.2GB of model downloads.
2. **ADE20K does not fit this domain.** On three test photos it predicted
   `window` zero times and `column` zero times. Those classes are trained on
   street scenes where a building is background mass; when a house fills the
   frame, the model labels it monolithically. The run on `h1.webp` produced no
   windows and no doors at all.
3. **It cannot produce scale.** A segmentation stack tells you *where* things
   are, never how big they are. That step still had to be a hand-written
   heuristic bolted on afterwards.

Point 3 is the decisive one. A multimodal model identifies the components *and*
reasons about scale from the same visual evidence, because it knows what a
domestic door is. Scale estimation is a **reasoning** task, not a pixel-labelling
task — so it belongs to a model that reasons.

### What the system does now

```
photo ─→ Gemini quality gate ─→ Gemini survey ─→ area maths ─→ quantities ─→ cost
                (5.1)          (5.2 + 5.5)        (5.5)         (5.6)       (5.7)
                                     │
                                     └─→ Gemini image edit ─→ redesign (5.4)
                                            └─ falls back to local composite
                                               if the API is rate-limited
```

One vision call returns labelled polygons **and** the scale derivation:

```json
{
  "regions": [
    {"id": "r1", "label": "wall", "polygon": [[0.1,0.2], ...], "confidence": 0.92}
  ],
  "scale": {
    "reference_object": "entrance door",
    "reference_real_feet": 6.75,
    "building_width_ft": 34.0,
    "building_height_ft": 21.0,
    "reasoning": "The door spans ~28% of image height..."
  }
}
```

Structured output (`response_schema`) constrains the model to the exact pydantic
shape, so there is no prose parsing and no path where it invents a component
label the cost engine has no rate for.

**AI is used only where judgement is needed.** Once geometry and scale exist,
every number is plain arithmetic — inspectable, reproducible, and unit-tested.
A homeowner arguing with a contractor needs the numbers to be checkable, not
generated.

### Two renderers for the redesign

Image generation is the most quota-limited call in the system: on a free-tier
key the text calls succeed comfortably while image generation returns 429. So
there are two paths, and `/design` falls back automatically.

| | Gemini | Local composite |
|---|---|---|
| Realism | photorealistic | reads as a recolour |
| Speed | 10–30s | instant |
| Cost | quota-limited | free |
| Determinism | varies per run | identical every run |
| Structure safety | model *could* move a window | provably cannot — only masked pixels change |

The local path is a **LAB luminance transfer**, not a colour fill. The material
supplies chroma (a/b channels) while the photograph supplies lightness (L), so
every shadow and gradient survives the repaint — a flat fill looks pasted on
because it discards exactly the shading that tells the eye where a surface is.
Textures (stone courses, tile grids, wood grain) are generated procedurally, so
no image assets ship with the project.

The response says which engine ran, the UI labels the image, and the PDF caption
reads "local composite preview" when applicable — a cheaper render is never
passed off as the photorealistic one.

---

## How the estimation works

The chain, and where error enters it:

```
normalised polygon           Gemini      boundary error, ~5-10%
  × image frame              exact
  × facade real dimensions   Gemini      scale error  ← DOMINATES
  = gross square feet
  − openings inside surface  exact       geometric deduction
  = net square feet
  × wastage                  convention  5-12% by material
  ÷ coverage per unit        product spec
  = purchase quantity
  × rates                    user-editable
  = cost
```

Specific decisions worth noting:

- **Walls are traced gross.** The model is told *not* to cut holes for windows.
  Openings are deducted separately so the report can show it as an auditable
  line (`336 sqft gross − 70 sqft openings = 266 sqft net`) instead of folding
  it invisibly into one number.
- **Paint is multi-coat.** 2 coats over 266 sqft is 532 sqft of painting at
  110 sqft/litre — not 266.
- **Railings are linear.** Priced per running foot, so area is meaningless;
  the horizontal span is measured instead.
- **Tiles round up to whole pieces.** You cannot order 43.2 tiles.
- **Labour is costed on net worked area; material on quantity purchased**
  (which includes wastage). Paying labour for wastage would be wrong.
- **Scale is exposed and editable in the UI**, because it is the number most
  likely to be wrong and it multiplies everything.

### Validation of model output

Structured output guarantees shape, never sense. `vision._sanitise` repairs the
failure modes actually observed:

- polygons clamped into frame
- degenerate (<3 point) and speck (<0.05% frame) regions dropped
- duplicate region ids made unique — the frontend keys selections off them
- facade dimensions bounded to plausible residential ranges (6–60 ft tall,
  8–200 ft wide); outside that, clamped, confidence lowered, and a **loud
  warning** raised rather than silently costing a fantasy building

---

## Setup

### Prerequisites

- **Python 3.10 or newer** (developed on 3.12). Check with `python3 --version`.
- A **Gemini API key** — free from https://aistudio.google.com/apikey
- No GPU, no model downloads, no database. Install is ~350MB and takes a minute
  or two.

On a bare Debian/Ubuntu box you may also need the venv module:

```bash
sudo apt install python3-venv
```

### 1. Create a virtual environment

```bash
cd house_outer_modification
python3 -m venv venv
```

### 2. Install dependencies

```bash
./venv/bin/pip install -r requirements.txt
```

<details>
<summary>Or activate the venv first, if you prefer <code>python</code> over <code>./venv/bin/python</code></summary>

```bash
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Every `./venv/bin/…` command below then becomes plain `python` / `pip`.
</details>

### 3. Add your API key

```bash
cp .env.example .env
```

Open `.env` and paste your key:

```
GEMINI_API_KEY=AIza...your_actual_key...
```

The key is read at startup. `.env` is gitignored, so it will not be committed.

### 4. Check the install (no API key needed)

```bash
./venv/bin/python test_offline.py
```

Expect **`ALL CHECKS PASSED`** — 22 checks covering the area maths, opening
deduction, paint/tile/railing quantity rules, rate overrides, the local
compositor and the PDF. This exercises everything except the Gemini calls, so
it passes even with no key and no network.

### 5. Run the server

```bash
./venv/bin/python -m uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000**.

The header should read *"Gemini connected"*. If it shows a red key warning, the
`.env` file was not picked up — confirm it sits next to `README.md` and restart.

### 6. Walk through it

1. Click a sample photo (or drop your own house exterior).
2. Wait ~10–20s — the photo is quality-checked, then surveyed for surfaces and scale.
3. **Check the facade dimensions** in the scale panel and correct them if they
   look wrong. Everything is priced off those two numbers.
4. Pick materials per component.
5. **Generate redesign.** Leave the engine on *Auto*; select *Local only* for an
   instant, quota-free render.
6. Adjust rates if you want, then **Download PDF report**.

### Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Header shows a red key warning | `.env` missing or misnamed. It must be `.env` in the project root, and the server must be restarted after creating it. |
| `429` / "quota exceeded" on redesign | Free-tier image generation is limited. *Auto* falls back to a local render automatically; or select *Local only*, wait for the stated retry delay, or enable billing. |
| `503` / "temporarily overloaded" | Google-side spike. The client already retries with backoff — just try again, or pin a different model via `GEMINI_VISION_MODEL` in `.env`. |
| "No usable Gemini model" | Every candidate model was rejected for your key. Set `GEMINI_VISION_MODEL` / `GEMINI_IMAGE_MODEL` in `.env` to a model you have access to. |
| Photo rejected at upload | Working as intended (5.1). The message says what is wrong; use a clearer, less obstructed shot of the facade. |
| `ModuleNotFoundError: app` | Run uvicorn from the project root, not from inside `app/`. |
| Port already in use | `--port 8001`, or stop the process using 8000. |

---

## Environment variables

Only the first is required.

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | — | **Required.** From https://aistudio.google.com/apikey |
| `GEMINI_VISION_MODEL` | `gemini-2.5-flash` | Perception model. Only set this if the default is unavailable on your key. |
| `GEMINI_IMAGE_MODEL` | `gemini-2.5-flash-image` | Redesign model. Same caveat. |

Model ids drift between API versions and tiers, so the app does not trust one
hard-coded name: if the configured model is rejected it walks a fallback list,
caches whichever works, and prints it (`[vision] using …`). A 503 retries the
same model with backoff instead of falling back; a 429 stops immediately rather
than burning the remaining quota.

---

## Project layout

```
app/
  main.py       FastAPI routes, one per workflow stage
  vision.py     Gemini perception: surfaces + scale       ← core AI
  redesign.py   Gemini image editing: the redesign        ← core AI
  estimate.py   areas → quantities → cost (no AI)         ← core logic
  composite.py  local render fallback (OpenCV, no API)
  materials.py  catalog: appearance + rates in one record
  schemas.py    pydantic contracts, double as Gemini's response schema
  report.py     PDF generation
  store.py      in-memory project store
static/         single-page frontend, no build step
test_offline.py 22 checks: estimation chain, compositor, model fallback
_old_local_pipeline/  first approach (OneFormer/DINO/SAM), kept for comparison
```

### API

| Endpoint | Requirement |
|---|---|
| `POST /api/projects` | 5.1 upload + quality gate |
| `POST /api/projects/{id}/analyse` | 5.2 structure, 5.5 scale |
| `POST /api/projects/{id}/regions` | 5.2 user corrections |
| `POST /api/projects/{id}/design` | 5.3 materials, 5.4 redesign (`engine`: auto/gemini/local) |
| `POST /api/projects/{id}/estimate` | 5.6 quantities, 5.7 cost |
| `GET  /api/projects/{id}/report` | 5.8 PDF report |

Stages are separate endpoints rather than one call because each is a slow Gemini
round trip with a decision point the user is meant to control — review regions
before costing, pick materials before rendering, adjust rates and recalculate.

---

## Limitations

Stated plainly, because an estimate whose error sources are hidden is worse than
no estimate.

**Accuracy**

- **Only visible surfaces are measured.** One photo shows one or two elevations.
  Side and rear walls are not in the estimate. For a whole-house quote this is
  the single largest omission.
- **Scale is the dominant error term.** It is inferred from assumed object sizes
  (a door is ~6.75 ft). A house with unusually tall doors produces a
  proportionally wrong estimate. A 20% scale error is a 20% cost error — the
  test suite asserts exactly this relationship.
- **Perspective is not rectified.** Areas are measured in image space against a
  frontal-plane assumption. An oblique photo foreshortens the facade and
  under-measures it. A homography correction would fix this and is the first
  thing I would add.
- **Boundaries are looser than a dedicated segmentation model's.** Deliberate
  trade: boundary error moves the total far less than scale error does.
- **Non-deterministic.** Two runs on one photo give slightly different polygons
  and totals.
- **The local fallback is not photorealistic.** It recolours the detected
  surfaces convincingly but does not re-render the house, so it reads as a good
  visualisation rather than a photograph of a finished renovation.

**Scope**

- Rates are indicative Indian residential figures, not live market data.
- Excludes scaffolding, surface preparation, repairs, waterproofing, approvals
  and taxes.
- Assumes low-rise residential, as per the brief.

**Prototype-level by design**

- In-memory storage — projects do not survive a restart.
- No authentication, no rate limiting, no persistence layer.
- Region editing is supported by the API (`POST /regions`) but the frontend only
  exposes include/exclude toggles, not polygon dragging.

### What I would do next, in priority order

1. **Perspective rectification** — detect vanishing points, rectify to a frontal
   plane before measuring. Biggest accuracy win available.
2. **Multi-photo input** — two or three elevations to cover the whole building,
   which removes the largest scope limitation.
3. **Ask the user for one real measurement** — a single known door width would
   collapse the dominant error term to near zero.
4. **Mask-constrained image editing** — pass the region masks to constrain edits
   spatially, guaranteeing the priced area and the visualised area are identical.
# AI_exterior_suggestion
