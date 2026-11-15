"""Pages of source files as grey canvas images."""

import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pymupdf

from .config import CONFIG
from .detector import fit

pymupdf.TOOLS.mupdf_display_errors(False)


def canvas_matrix(page, cfg=CONFIG):
    return pymupdf.Matrix(cfg.geometry.canvas_w / page.rect.width, cfg.geometry.canvas_h / page.rect.height)


def render_page(page, cfg=CONFIG):
    """Grey image of an open MuPDF page stretched to the canvas."""
    pm = page.get_pixmap(matrix=canvas_matrix(page, cfg), colorspace=pymupdf.csGRAY, alpha=False)
    return np.frombuffer(pm.samples, np.uint8).reshape(pm.h, pm.w).copy()


def render(path, number, cfg=CONFIG):
    with pymupdf.open(str(path)) as doc:
        return render_page(doc[number], cfg)


def render_other(path, number, cfg=CONFIG):
    """pdftoppm (a pdf; in its own process: MuPDF aborts the worker on some files) or PIL (an image, page 0), stretched to the canvas. None for page > 0 of an image;
    raises when the file cannot be read."""
    k = cfg.kaggle
    if str(path).lower().endswith(".pdf"):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    "pdftoppm",
                    "-gray",
                    "-f",
                    str(number + 1),
                    "-l",
                    str(number + 1),
                    "-scale-to-x",
                    str(cfg.geometry.canvas_w),
                    "-scale-to-y",
                    str(cfg.geometry.canvas_h),
                    str(path),
                    f"{tmp}/o",
                ],
                check=True,
                timeout=k.local_pdf_seconds,
                capture_output=True,
            )
            return cv2.imread(str(sorted(Path(tmp).glob("o*"))[0]), cv2.IMREAD_GRAYSCALE)
    if number == 0:
        from PIL import Image

        return fit(np.asarray(Image.open(path).convert("L")), cfg)
    return None
