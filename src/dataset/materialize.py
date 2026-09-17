"""Shared "manifest entry -> flat 1024x1024 uint8 grayscale page" helper,
used by both export_pages.py (100k .npz set) and export_visual_sample.py
(1k JPG spot-check). Plain uint8 grayscale (0=ink..255=paper), matching
the on-disk raster sources directly -- no INK/PAPER reflectance scaling,
since no downstream consumer of this flat set (checked src/synth/*.py and
scripts/export_uv_from_images.py) reads a plain flat page other than as a
uint8 matrix (export_uv_from_images.py's own `original` field, and its
_tex_from_matrix, both start from a uint8 matrix and rescale explicitly
where needed) -- callers that need INK/PAPER-scaled float32 apply
_tex_from_matrix themselves.
"""
import cv2
import numpy as np

from src.dataset.crop import crop_from_bbox
from src.dataset.deskew import estimate_deskew_angle, needs_deskew, rotate90, rotate_full_res


def _rasterize_pdf_page(source_path: str, page_index: int):
    """Same fixed zoom as build_manifest.RASTER_ZOOM -- used_h/used_w in
    the manifest were measured at this zoom, and materialize_page resizes
    to them afterward if they differ (covers the below-1024 upscale case)."""
    import fitz
    from src.dataset.build_manifest import RASTER_ZOOM
    with fitz.open(source_path) as doc:
        page = doc[page_index]
        pix = page.get_pixmap(matrix=fitz.Matrix(RASTER_ZOOM, RASTER_ZOOM), colorspace=fitz.csGRAY, alpha=False)
        return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()


def render_book_text_page(source_path: str, text_seed: int) -> np.ndarray:
    """-> uint8 (1024,1024) grayscale, deterministic in (source_path, text_seed)."""
    from src.synth import config as cfg
    from src.synth.text_texture import render_flat_text, sample_snippet, find_fonts
    rng = np.random.default_rng(text_seed)
    snippet = sample_snippet([source_path], rng)
    font_files = find_fonts()
    tex = render_flat_text(snippet, rng, cfg.TEXT_PAGE_PX, cfg.TEXT_PAGE_MM, font_files,
                           margin_mm=0.0, add_old_creases=False, add_noise=False)
    page01 = (tex - cfg.INK) / (cfg.PAPER - cfg.INK)
    gray = np.clip(page01 * 255.0, 0, 255).astype(np.uint8)
    if gray.shape != (1024, 1024):
        gray = cv2.resize(gray, (1024, 1024), interpolation=cv2.INTER_AREA)
    return gray


def materialize_page(entry: dict) -> np.ndarray:
    """-> uint8 (1024,1024) grayscale. All keep/reject filtering, and the
    90-degree orientation decision, already happened at manifest-build
    time on the full page; rotate90 here just replays that decision
    rather than re-detecting it, since a 90-degree rotation moves content
    enough that redoing detection after crop_from_bbox (on a much smaller
    window) is far less reliable. Deskew applies after, to the resulting
    square crop, so it can never change its shape -- no fit check needed."""
    if entry["kind"] == "book_text":
        return render_book_text_page(entry["source_path"], entry["text_seed"])

    used_h, used_w = entry["used_h"], entry["used_w"]
    if entry["source_path"].lower().endswith(".pdf"):
        gray = _rasterize_pdf_page(entry["source_path"], entry["page_index"])
    else:
        gray = cv2.imread(entry["source_path"], cv2.IMREAD_GRAYSCALE)

    gray = rotate90(gray, entry["rotate90"])

    h, w = gray.shape
    if (h, w) != (used_h, used_w):
        gray = cv2.resize(gray, (used_w, used_h),
                          interpolation=cv2.INTER_AREA if (h * w) > (used_h * used_w) else cv2.INTER_LINEAR)

    page = crop_from_bbox(gray, entry["bbox_y0"], entry["bbox_x0"], entry["bbox_h"], entry["bbox_w"],
                          entry["crop_y0"], size=entry["crop_size"])

    if entry["needs_deskew"]:
        angle = estimate_deskew_angle(page)
        if needs_deskew(angle):
            page = rotate_full_res(page, angle, fill_value=255)

    return page
