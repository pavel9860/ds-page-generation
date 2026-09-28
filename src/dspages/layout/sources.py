"""Manifest entries and their page rasters."""
import json

import cv2
import numpy as np

from .deskew import estimate_deskew_angle, needs_deskew, rotate90, rotate_full_res

RASTER_ZOOM = 2.0


def load_manifest(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def raster_page(entry):
    """uint8 gray page of a raster entry at the manifest's size, turned upright and deskewed."""
    if entry["source_path"].lower().endswith(".pdf"):
        import pymupdf as fitz
        with fitz.open(entry["source_path"]) as doc:
            pix = doc[entry["page_index"]].get_pixmap(matrix=fitz.Matrix(RASTER_ZOOM, RASTER_ZOOM),
                                                      colorspace=fitz.csGRAY, alpha=False)
            gray = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()
    else:
        gray = cv2.imread(entry["source_path"], cv2.IMREAD_GRAYSCALE)
    gray = rotate90(gray, entry.get("rotate90", 0))
    h, w = entry["used_h"], entry["used_w"]
    if gray.shape != (h, w):
        gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA if gray.size > h * w else cv2.INTER_LINEAR)
    if entry["needs_deskew"]:
        angle = estimate_deskew_angle(gray)
        if needs_deskew(angle):
            gray = rotate_full_res(gray, angle, fill_value=255)
    return gray
