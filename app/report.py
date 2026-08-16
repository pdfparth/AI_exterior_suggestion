"""Downloadable PDF report. PDF 5.8.

5.8 asks for a document "usable as a discussion document with contractors",
which sets the bar: a contractor must be able to disagree with it specifically.
That means showing the working, not just the total - the scale assumption, the
gross-minus-openings step, the wastage percentage and the rates all appear as
their own columns, so an argument can be about one number rather than about
whether the whole thing is trustworthy.
"""

from __future__ import annotations

import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .schemas import Analysis, Estimate

INK = colors.HexColor("#1a1a1a")
MUTED = colors.HexColor("#6b6b6b")
RULE = colors.HexColor("#d8d8d8")
BAND = colors.HexColor("#f2f2f0")
ACCENT = colors.HexColor("#2f5d50")


def _styles():
    ss = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "t", parent=ss["Title"], fontName="Helvetica-Bold",
            fontSize=19, leading=23, textColor=INK, alignment=TA_LEFT, spaceAfter=2,
        ),
        "sub": ParagraphStyle(
            "s", parent=ss["Normal"], fontSize=8.5, leading=12, textColor=MUTED,
        ),
        "h2": ParagraphStyle(
            "h", parent=ss["Heading2"], fontName="Helvetica-Bold",
            fontSize=11.5, leading=14, textColor=INK, spaceBefore=14, spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "b", parent=ss["Normal"], fontSize=8.8, leading=12.5, textColor=INK,
        ),
        "small": ParagraphStyle(
            "sm", parent=ss["Normal"], fontSize=7.6, leading=10.5, textColor=MUTED,
        ),
        "cell": ParagraphStyle(
            "c", parent=ss["Normal"], fontSize=7.6, leading=9.5, textColor=INK,
        ),
    }


def _money(v: float) -> str:
    """Indian digit grouping - 12,34,567 not 1,234,567."""
    s = f"{abs(v):,.0f}"
    n = f"{abs(v):.0f}"
    if len(n) > 3:
        head, tail = n[:-3], n[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join(parts) + "," + tail
    return ("-" if v < 0 else "") + s


def _fit(image_bytes: bytes, max_w: float, max_h: float) -> Image | None:
    """Scale an image into a box, preserving aspect."""
    if not image_bytes:
        return None
    try:
        from PIL import Image as PILImage

        with PILImage.open(io.BytesIO(image_bytes)) as im:
            w, h = im.size
        ratio = min(max_w / w, max_h / h)
        return Image(io.BytesIO(image_bytes), width=w * ratio, height=h * ratio)
    except Exception:
        return None


def build(
    original: bytes,
    redesigned: bytes | None,
    analysis: Analysis,
    estimate: Estimate,
    project_name: str = "Exterior Renovation Estimate",
    redesign_engine: str = "",
) -> bytes:
    """Render the full report to PDF bytes."""
    st = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=16 * mm, rightMargin=16 * mm,
        topMargin=15 * mm, bottomMargin=15 * mm,
        title=project_name, author="AI Exterior Renovation Estimator",
    )
    avail = doc.width
    story = []

    # --- header -----------------------------------------------------------
    story.append(Paragraph(project_name, st["title"]))
    story.append(
        Paragraph(
            f"Generated {datetime.now().strftime('%d %B %Y, %H:%M')} &nbsp;·&nbsp; "
            "Advisory estimate from a single photograph &nbsp;·&nbsp; not a binding quotation",
            st["sub"],
        )
    )
    story.append(Spacer(1, 5))
    story.append(_rule(avail))

    # --- headline total ---------------------------------------------------
    story.append(Spacer(1, 9))
    total_tbl = Table(
        [[
            Paragraph("<b>ESTIMATED TOTAL</b>", st["small"]),
            Paragraph(
                f"<font size=17 color='#2f5d50'><b>Rs {_money(estimate.grand_total)}</b></font>",
                st["body"],
            ),
        ],
        [
            Paragraph("Material", st["small"]),
            Paragraph(f"Rs {_money(estimate.material_total)}", st["body"]),
        ],
        [
            Paragraph("Labour", st["small"]),
            Paragraph(f"Rs {_money(estimate.labour_total)}", st["body"]),
        ]],
        colWidths=[avail * 0.25, avail * 0.75],
    )
    total_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BAND),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
    ]))
    story.append(total_tbl)

    # --- before / after ---------------------------------------------------
    story.append(Paragraph("1. Before and after", st["h2"]))
    # Leave real whitespace between the panes; without it two similar photos
    # read as one wide image and the comparison is lost.
    gutter = 7 * mm
    cell_w = (avail - gutter) / 2 - 3
    before = _fit(original, cell_w, 62 * mm)
    after = _fit(redesigned, cell_w, 62 * mm) if redesigned else None

    if after is None:
        after = Paragraph(
            "<i>No redesign was generated for this estimate.</i>", st["small"]
        )

    img_tbl = Table(
        [[before or Paragraph("—", st["small"]), "", after],
         [Paragraph("<b>ORIGINAL</b>", st["small"]), "",
          Paragraph(_redesign_caption(redesign_engine), st["small"])]],
        colWidths=[cell_w, gutter, cell_w],
    )
    img_tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, 0), "TOP"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 1), (-1, 1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        # Thin frames so a light-toned facade still reads as a bounded photo.
        ("BOX", (0, 0), (0, 0), 0.5, RULE),
        ("BOX", (2, 0), (2, 0), 0.5, RULE),
    ]))
    story.append(img_tbl)

    # --- how the areas were derived --------------------------------------
    story.append(Paragraph("2. How the area was measured", st["h2"]))
    s = estimate.scale_used
    story.append(Paragraph(
        f"Pixel measurements were converted to real dimensions using "
        f"<b>{s.reference_object}</b>, assumed to be <b>{s.reference_real_feet:.2f} ft</b>. "
        f"That gives a visible facade of approximately "
        f"<b>{s.building_width_ft:.0f} ft wide by {s.building_height_ft:.0f} ft tall</b>. "
        f"Model confidence in this scale: {s.confidence:.0%}.",
        st["body"],
    ))
    if s.reasoning:
        story.append(Spacer(1, 3))
        story.append(Paragraph(f"<i>{s.reasoning}</i>", st["small"]))

    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<b>This scale multiplies every number in this report.</b> If the facade "
        "dimensions above look wrong, correct them in the tool and regenerate — "
        "a 20% error in scale is a 20% error in the total.",
        st["small"],
    ))

    # --- detected surfaces -------------------------------------------------
    story.append(Paragraph("3. Surfaces identified", st["h2"]))
    counts: dict[str, int] = {}
    for r in analysis.regions:
        counts[r.label] = counts.get(r.label, 0) + 1
    surface_rows = [[
        Paragraph("<b>Component</b>", st["cell"]),
        Paragraph("<b>Count</b>", st["cell"]),
    ]] + [
        [Paragraph(k.replace("_", " ").title(), st["cell"]),
         Paragraph(str(v), st["cell"])]
        for k, v in sorted(counts.items())
    ]
    st_tbl = Table(surface_rows, colWidths=[avail * 0.3, avail * 0.15])
    st_tbl.setStyle(_grid())
    story.append(st_tbl)
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        f"{analysis.storeys} storey(s) visible. {analysis.style_note}", st["small"]
    ))

    # --- the estimate ------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("4. Cost breakdown", st["h2"]))

    head = ["Component", "Material", "Net qty", "Purchase qty", "Rate", "Material", "Labour", "Total"]
    rows = [[Paragraph(f"<b>{h}</b>", st["cell"]) for h in head]]

    for i in estimate.line_items:
        rows.append([
            Paragraph(i.label.replace("_", " ").title(), st["cell"]),
            Paragraph(i.material_name, st["cell"]),
            Paragraph(f"{i.net_area_sqft:,.0f} {i.unit}", st["cell"]),
            Paragraph(f"{i.quantity:,.1f} {i.material_unit_label}", st["cell"]),
            Paragraph(f"{_money(i.material_rate)}", st["cell"]),
            Paragraph(_money(i.material_cost), st["cell"]),
            Paragraph(_money(i.labour_cost), st["cell"]),
            Paragraph(f"<b>{_money(i.total_cost)}</b>", st["cell"]),
        ])

    rows.append([
        Paragraph("<b>GRAND TOTAL</b>", st["cell"]), Paragraph("", st["cell"]),
        Paragraph("", st["cell"]), Paragraph("", st["cell"]), Paragraph("", st["cell"]),
        Paragraph(f"<b>{_money(estimate.material_total)}</b>", st["cell"]),
        Paragraph(f"<b>{_money(estimate.labour_total)}</b>", st["cell"]),
        Paragraph(f"<b>{_money(estimate.grand_total)}</b>", st["cell"]),
    ])

    w = avail
    cost_tbl = Table(
        rows,
        colWidths=[w*0.11, w*0.22, w*0.12, w*0.13, w*0.09, w*0.11, w*0.11, w*0.11],
        repeatRows=1,
    )
    cost_tbl.setStyle(_grid(total_row=True))
    story.append(cost_tbl)

    # --- quantity working --------------------------------------------------
    story.append(Paragraph("5. Quantity working", st["h2"]))
    story.append(Paragraph(
        "Each line shows how the purchase quantity was reached, so it can be "
        "checked line by line.", st["small"]
    ))
    story.append(Spacer(1, 5))

    qhead = ["Component", "Gross", "Openings", "Net", "Wastage", "Basis"]
    qrows = [[Paragraph(f"<b>{h}</b>", st["cell"]) for h in qhead]]
    for i in estimate.line_items:
        qrows.append([
            Paragraph(f"{i.label.replace('_',' ').title()}", st["cell"]),
            Paragraph(f"{i.gross_area_sqft:,.0f}", st["cell"]),
            Paragraph(f"-{i.deducted_sqft:,.0f}" if i.deducted_sqft else "—", st["cell"]),
            Paragraph(f"<b>{i.net_area_sqft:,.0f} {i.unit}</b>", st["cell"]),
            Paragraph(f"+{i.wastage_pct:.0f}%", st["cell"]),
            Paragraph(i.coverage_note, st["cell"]),
        ])
    q_tbl = Table(qrows, colWidths=[w*0.13, w*0.09, w*0.10, w*0.13, w*0.09, w*0.46], repeatRows=1)
    q_tbl.setStyle(_grid())
    story.append(q_tbl)

    # --- assumptions and limits -------------------------------------------
    block = [Paragraph("6. Assumptions and exclusions", st["h2"])]
    for a in estimate.assumptions:
        block.append(Paragraph(f"• {a}", st["small"]))
        block.append(Spacer(1, 2))
    story.append(KeepTogether(block))

    if estimate.warnings:
        wblock = [Paragraph("7. Warnings", st["h2"])]
        wblock.append(Paragraph(
            "These affect how much you should trust the numbers above.", st["small"]
        ))
        wblock.append(Spacer(1, 4))
        for wmsg in estimate.warnings:
            wblock.append(Paragraph(f"! {wmsg}", st["small"]))
            wblock.append(Spacer(1, 2))
        story.append(KeepTogether(wblock))

    story.append(Spacer(1, 12))
    story.append(_rule(avail))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "This estimate was produced automatically from one photograph by an AI "
        "system. Areas are approximate and only cover surfaces visible in that "
        "photograph. It is intended as a starting point for discussion with a "
        "contractor, not as a substitute for a site measurement.",
        st["small"],
    ))

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()


def _redesign_caption(engine: str) -> str:
    """Name the renderer on the image itself.

    A locally composited preview and a generative render are different kinds of
    evidence, and a contractor reading this should be able to tell which one
    they are looking at.
    """
    if engine == "local":
        return "<b>REDESIGNED</b> — local composite preview"
    if engine == "gemini":
        return "<b>REDESIGNED</b> — AI-generated visualisation"
    return "<b>REDESIGNED</b>"


def _rule(width: float) -> Table:
    t = Table([[""]], colWidths=[width], rowHeights=[0.6])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), RULE)]))
    return t


def _grid(total_row: bool = False) -> TableStyle:
    cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), BAND),
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    if total_row:
        cmds.append(("BACKGROUND", (0, -1), (-1, -1), BAND))
    return TableStyle(cmds)


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(MUTED)
    canvas.drawString(16 * mm, 9 * mm, "AI Exterior Renovation Estimator — advisory estimate")
    canvas.drawRightString(A4[0] - 16 * mm, 9 * mm, f"Page {doc.page}")
    canvas.restoreState()
