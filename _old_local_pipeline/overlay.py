"""Render regions over the source image.

This is the primary correctness check for Stage 1. Numbers in regions.json are
hard to eyeball; a coloured overlay tells you in one glance whether the detector
put the wall where the wall is.
"""

from __future__ import annotations

import cv2
import numpy as np

from config import LABEL_COLORS
from regions import Region


def _color(label: str) -> tuple[int, int, int]:
    return LABEL_COLORS.get(label, LABEL_COLORS["other"])


def render(
    image: np.ndarray,
    regions: list[Region],
    alpha: float = 0.40,
    show_boxes: bool = True,
) -> np.ndarray:
    """Return a copy of `image` with region masks and labels drawn over it."""
    base = image.copy()
    fill = image.copy()

    # Largest first, so small regions (windows) paint on top of big ones (wall).
    ordered = sorted(regions, key=lambda r: r.px_area, reverse=True)

    for r in ordered:
        color = _color(r.label)
        if len(r.polygon) >= 3:
            pts = np.asarray(r.polygon, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(fill, [pts], color)
            cv2.polylines(base, [pts], True, color, 2, cv2.LINE_AA)
        else:
            x1, y1, x2, y2 = (int(v) for v in r.box)
            cv2.rectangle(fill, (x1, y1), (x2, y2), color, -1)

    out = cv2.addWeighted(fill, alpha, base, 1 - alpha, 0)

    for r in ordered:
        color = _color(r.label)
        x1, y1, x2, y2 = (int(v) for v in r.box)
        if show_boxes:
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        _draw_label(out, f"{r.label} {r.confidence:.2f}", x1, y1, color)

    return _draw_legend(out, ordered)


def _draw_label(img: np.ndarray, text: str, x: int, y: int, color) -> None:
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    y = max(y, th + 6)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y), color, -1)
    cv2.putText(img, text, (x + 3, y - 4), font, scale, (20, 20, 20), thick, cv2.LINE_AA)


def _draw_legend(img: np.ndarray, regions: list[Region]) -> np.ndarray:
    labels: dict[str, int] = {}
    for r in regions:
        labels[r.label] = labels.get(r.label, 0) + 1
    if not labels:
        return img

    pad, row_h, box_w = 10, 22, 180
    h = pad * 2 + row_h * len(labels)
    panel = np.full((h, box_w, 3), 245, dtype=np.uint8)

    for i, (label, count) in enumerate(sorted(labels.items())):
        y = pad + i * row_h
        cv2.rectangle(panel, (pad, y + 4), (pad + 14, y + 16), _color(label), -1)
        cv2.putText(
            panel, f"{label} x{count}", (pad + 22, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1, cv2.LINE_AA,
        )

    # Bottom-left, clipped if the image is too small to hold it.
    ih, iw = img.shape[:2]
    ph, pw = min(h, ih), min(box_w, iw)
    roi = img[ih - ph:ih, 0:pw]
    cv2.addWeighted(panel[:ph, :pw], 0.85, roi, 0.15, 0, roi)
    return img


def save(path: str, image: np.ndarray, regions: list[Region], **kw) -> None:
    cv2.imwrite(path, render(image, regions, **kw))
