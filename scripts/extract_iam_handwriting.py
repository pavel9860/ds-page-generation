"""Extracts the handwriting region from IAM Handwriting Database form scans.

Layout (fixed across the IAM form template): a printed heading, a full-width horizontal rule,
a printed prompt sentence, a second full-width rule, the handwritten transcription, a third
full-width rule, then a "Name:" field. The handwriting is exactly the band between the 2nd and
3rd horizontal rule.

Detection uses the standard morphological line-extraction approach (binarize with Otsu, then
open with a horizontal structuring element) rather than a raw per-row ink-fraction threshold:
opening with a moderate-width kernel (wide enough that no text/handwriting stroke survives it,
narrow enough to tolerate the small real gaps a scanned printed rule has) leaves only long
horizontal strokes, then a row is a ruling line if enough of that survives across it — a
sum-based test, not "one unbroken run", since even a solid-looking printed rule can have small
gaps under Otsu threshold from scan noise/antialiasing.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

LINE_KERNEL_FRACTION = 1 / 20   # horizontal opening kernel width, as a fraction of page width
LINE_ROW_SURVIVAL_FRACTION = 0.25   # a ruling-line row keeps at least this much of the page width
LINE_MERGE_GAP = 5   # px: adjacent qualifying rows within this gap are one physical rule


def find_horizontal_lines(gray: np.ndarray) -> list[int]:
    """Returns the y-coordinate (row) of each detected horizontal ruling line's center,
    top to bottom, via morphological opening with a horizontal kernel (standard line-extraction
    technique for scanned documents/tables): text and handwriting have no stroke long enough to
    survive the opening, so what's left is (close to) ruling lines only."""
    h, w = gray.shape
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel_w = max(15, round(w * LINE_KERNEL_FRACTION))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    lines_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    row_survival = lines_mask.sum(axis=1) / 255 / w
    line_rows = np.flatnonzero(row_survival >= LINE_ROW_SURVIVAL_FRACTION)
    if len(line_rows) == 0:
        return []

    groups: list[list[int]] = [[int(line_rows[0])]]
    for y in line_rows[1:]:
        y = int(y)
        if y - groups[-1][-1] <= LINE_MERGE_GAP:
            groups[-1].append(y)
        else:
            groups.append([y])
    return [round(sum(g) / len(g)) for g in groups]


def extract_handwriting_region(image_path: str) -> tuple[Image.Image, dict]:
    img = Image.open(image_path).convert("L")
    gray = np.array(img)
    lines = find_horizontal_lines(gray)

    if len(lines) < 3:
        raise ValueError(f"expected >=3 ruling lines, found {len(lines)}: {lines}")

    y0, y1 = lines[1], lines[2]
    color_img = Image.open(image_path).convert("RGB")
    crop = color_img.crop((0, y0, color_img.width, y1))
    meta = {"all_lines": lines, "crop_y0": y0, "crop_y1": y1, "crop_height": y1 - y0}
    return crop, meta


# Reference line positions for the 3 IAM-form rules (top to bottom), computed once from a
# 200-page random sample of pages where all 3 lines were cleanly detected (median1=312,
# median2=655.5, median3=2798 — see docs/manifest_notes.md). Used only to fill in a rule this
# page's own detector missed, never to override one it actually found.
REFERENCE_LINE_MEDIANS = [312, 655, 2798]


def fill_missing_lines(detected: list[int], reference_medians: list[int] = REFERENCE_LINE_MEDIANS,
                        ) -> tuple[list[int], list[bool]]:
    """Matches each detected line to its nearest reference rank (1st/2nd/3rd rule), then fills
    any unmatched rank with that rank's reference median. Returns (lines, is_detected) both
    length len(reference_medians), in rank order; is_detected[i] is False where the value was
    filled in rather than found on the page."""
    n = len(reference_medians)
    assigned: list[int | None] = [None] * n
    remaining = sorted(detected)
    # Greedy nearest-rank assignment: each detected line claims whichever unclaimed rank it is
    # closest to (works because ranks and detections are both naturally in ascending y order).
    for y in remaining:
        rank = min((i for i in range(n) if assigned[i] is None),
                   key=lambda i: abs(reference_medians[i] - y))
        assigned[rank] = y
    is_detected = [v is not None for v in assigned]
    lines = [assigned[i] if assigned[i] is not None else reference_medians[i] for i in range(n)]
    return lines, is_detected
