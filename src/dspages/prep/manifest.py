"""Source manifest: one JSON line per usable page of the layout corpora (PDF pages, images, book texts).

Fields: source_path, page_index, language, category, kind (raster | book_text), content_frac, needs_deskew,
rotate90, used_h, used_w (raster size the bbox refers to), bbox_y0, bbox_x0, bbox_h, bbox_w (content bbox).

python -m dspages.prep.manifest --out <paths.manifest>
"""
import argparse
import glob
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from ..conditions import script
from ..config import LayoutCfg, Paths
from ..layout.content import bbox, content_mask
from ..layout.deskew import detect_rotation, rotate90
from ..layout.sources import RASTER_ZOOM
from ..layout.text import _words, book_text
from .text_extract import DOMINANT_IMAGE_AREA_FRAC, book_language, script_language

LAYOUTS = Paths().layouts
BOOKS = Paths().books
MIN_CONTENT_AREA_FRAC = 0.01
MAX_MARGIN_FRAC = 0.50
MIN_CONTENT_FRAC = 0.01
MIN_BOOK_WORDS = 3000
BOOK_SCAN_LANGUAGES = {"Bakht Novel": "ur", "Dana Pani": "ur", "Yaaram novel": "ur", "Kandukondaen": "ta",
                       "Yoga Vasistam": "ta", "Nagaon_Ka_Rahasya": "hi", "kupdf.net_1-1-": "el", "Kusadikika": "sw"}
SCAN_CHARS_PER_PAGE = 300                  # book PDFs with less text per page are used as raster pages
PAGES_PER_PDF_CAP = 10                 # pages per PDF; all pages of rare-script documents
RARE_SCRIPTS = ("cyrillic", "cjk", "other")


def _init():
    cv2.setNumThreads(1)


def load_metadata(meta_path: str) -> dict:
    """local_path -> {language, bucket, category, pages}, keyed after
    normalizing the metadata's stale '/Layouts/test/corpus/...' prefix to
    the corpus's actual on-disk '/Layouts/corpus/...' location."""
    idx = {}
    if not os.path.exists(meta_path):
        return idx
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            p = d["local_path"].replace("/Layouts/test/corpus/", "/Layouts/corpus/")
            idx[p] = d
    return idx


def _seed(path: str) -> int:
    return hash(path) & 0xFFFFFFFF


def _page_indices(page_count: int, rng, cap) -> list:
    if cap is None or page_count <= cap:
        return list(range(page_count))
    return sorted(rng.choice(page_count, size=cap, replace=False).tolist())


# ---------------------------------------------------------------------------
# per-page processing, shared by pdf and image sources
# ---------------------------------------------------------------------------

def _bbox_keep(b, h, w):
    y0, y1, x0, x1 = b
    return ((y1 - y0 + 1) * (x1 - x0 + 1) >= MIN_CONTENT_AREA_FRAC * h * w
            and 1 - (y1 - y0 + 1) / h <= MAX_MARGIN_FRAC and 1 - (x1 - x0 + 1) / w <= MAX_MARGIN_FRAC)


def _bbox_fields(b, s, h, w):
    y0, y1, x0, x1 = (round(v / s) for v in b)
    return dict(bbox_y0=y0, bbox_x0=x0, bbox_h=min(h, y1 + 1) - y0, bbox_w=min(w, x1 + 1) - x0)


def _process_gray(gray: np.ndarray, rng):
    rot = detect_rotation(gray)
    gray = rotate90(gray, rot)
    mask, s = content_mask(gray)
    b = bbox(mask)
    if b is None or not _bbox_keep(b, *mask.shape) or mask.mean() < MIN_CONTENT_FRAC:
        return None
    return dict(content_frac=round(float(mask.mean()), 5), needs_deskew=True, rotate90=rot,
                used_h=gray.shape[0], used_w=gray.shape[1], **_bbox_fields(b, s, *gray.shape))


def _pdf_text_blocks_px(page, zoom: float) -> list:
    """Word-level boxes, not paragraph-level "blocks" -- a block's own
    bbox spans its full line height/spacing and reads as much denser
    content than it visually is, letting sparse pages clear the patch-
    coverage bar that the pixel-based raster path would reject them on."""
    return [(w[0] * zoom, w[1] * zoom, w[2] * zoom, w[3] * zoom) for w in page.get_text("words")]


def _occupancy(blocks: list, width: int, height: int, x_off: float = 0.0,
               y_off: float = 0.0, scale: float = 1.0) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for x0, y0, x1, y1 in blocks:
        gx0 = max(0, int((x0 - x_off) * scale))
        gy0 = max(0, int((y0 - y_off) * scale))
        gx1 = min(width, int(np.ceil((x1 - x_off) * scale)))
        gy1 = min(height, int(np.ceil((y1 - y_off) * scale)))
        if gx1 > gx0 and gy1 > gy0:
            mask[gy0:gy1, gx0:gx1] = True
    return mask


def _pdf_text_page_info(page, rng, zoom: float = RASTER_ZOOM):
    """Content bbox from the page's word boxes (no rasterization); None for landscape pages."""
    pw_pt, ph_pt = page.rect.width, page.rect.height
    if pw_pt > ph_pt:
        return None
    used_w, used_h = round(pw_pt * zoom), round(ph_pt * zoom)
    blocks = _pdf_text_blocks_px(page, zoom)
    if not blocks:
        return None
    s = min(1.0, 512 / max(used_w, used_h))
    mask = _occupancy(blocks, max(1, round(used_w * s)), max(1, round(used_h * s)), scale=s)
    b = bbox(mask)
    if b is None or not _bbox_keep(b, *mask.shape):
        return None
    return dict(content_frac=round(float(mask.mean()), 5), needs_deskew=False, used_h=used_h, used_w=used_w,
                **_bbox_fields(b, s, used_h, used_w))


def _pdf_page_is_scan(page) -> bool:
    """A page with a real text layer can still be a scanned image with an
    OCR text overlay -- its own text bbox says nothing about visual skew
    in that case, so it needs the raster/deskew path, not the vector one."""
    page_area = page.rect.width * page.rect.height
    if not page_area:
        return False
    return any((info["bbox"][2] - info["bbox"][0]) * (info["bbox"][3] - info["bbox"][1]) / page_area
               > DOMINANT_IMAGE_AREA_FRAC for info in page.get_image_info())


def _job_pdf(args):
    import pymupdf as fitz
    pdf_path, page_count_hint, language, category, seed = args
    rng = np.random.default_rng(seed)
    try:
        doc = fitz.open(pdf_path)
        rare = script(dict(language=language), LayoutCfg().scripts) in RARE_SCRIPTS
        page_indices = _page_indices(doc.page_count, rng, None if rare else PAGES_PER_PDF_CAP)
    except Exception:
        return []
    recs = []
    try:
        for pi in page_indices:
            if pi >= doc.page_count:
                continue
            page = doc[pi]
            has_text = bool(page.get_text().strip())
            info = has_text and not _pdf_page_is_scan(page) and _pdf_text_page_info(page, rng)
            if not info:
                pix = page.get_pixmap(matrix=fitz.Matrix(RASTER_ZOOM, RASTER_ZOOM),
                                      colorspace=fitz.csGRAY, alpha=False)
                gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()
                info = _process_gray(gray, rng)
            if info is None:
                continue
            recs.append(dict(source_path=pdf_path, page_index=pi, language=language,
                             category=category, kind="raster", **info))
    except Exception:
        return recs
    finally:
        doc.close()
    return recs


def _job_image(args):
    image_path, language, category, seed = args
    rng = np.random.default_rng(seed)
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return []
    info = _process_gray(gray, rng)
    if info is None:
        return []
    return [dict(source_path=image_path, page_index=0, language=language,
                 category=category, kind="raster", **info)]


# ---------------------------------------------------------------------------
# source enumeration
# ---------------------------------------------------------------------------

def corpus_pdf_jobs(meta_idx: dict):
    jobs = []
    for sub in ("pdf", "scanned", "forms_bulk"):
        for p in sorted(glob.glob(os.path.join(LAYOUTS, "corpus", sub, "**", "*.pdf"), recursive=True)):
            m = meta_idx.get(p)
            language = m["language"] if m else "unknown"
            category = m.get("category") if m else None
            jobs.append((p, m.get("pages") if m else None, language, category, _seed(p)))
    return jobs


def corpus_overflow_jobs():
    pdf_jobs, img_jobs = [], []
    for p in sorted(glob.glob(os.path.join(LAYOUTS, "corpus_overflow", "Images", "*.png"))):
        img_jobs.append((p, "unknown", None, _seed(p)))
    for p in sorted(glob.glob(os.path.join(LAYOUTS, "corpus_overflow", "pdf", "*"))):
        if p.lower().endswith(".pdf"):
            pdf_jobs.append((p, None, "unknown", None, _seed(p)))
        elif p.lower().endswith(".png"):
            img_jobs.append((p, "unknown", None, _seed(p)))
    return pdf_jobs, img_jobs


def pdf_png_jobs():
    return [(p, "en", None, _seed(p))
            for p in sorted(glob.glob(os.path.join(LAYOUTS, "Pdf", "*.png")))]


def xfund_funsd_jobs():
    """<lang>.<split>/*.png dirs -- language is the dir name's prefix
    before the dot (en = FUNSD, others = XFUND)."""
    root = os.path.join(LAYOUTS, "XFUND and FUNSD")
    jobs = []
    for p in sorted(glob.glob(os.path.join(root, "*", "*.png"))):
        lang = os.path.basename(os.path.dirname(p)).split(".")[0]
        jobs.append((p, lang, "form", _seed(p)))
    return jobs


def arxiv_jobs():
    return [(p, None, "en", "scientific_paper", _seed(p))
            for p in sorted(glob.glob(os.path.join(LAYOUTS, "scientific_paper", "arxiv_pdfs", "*.pdf")))]


# ---------------------------------------------------------------------------
# book-text filler (step 3)
# ---------------------------------------------------------------------------

def book_txt_paths():
    paths = sorted(glob.glob(os.path.join(BOOKS, "**", "*.txt"), recursive=True))
    return [p for p in paths if len(_words(p)) >= MIN_BOOK_WORDS and not _scan_text(p)]


def _scan_text(txt):
    pdf = os.path.join(os.path.dirname(BOOKS), os.path.relpath(txt, BOOKS))[:-4] + ".pdf"
    pdf = pdf if os.path.exists(pdf) else os.path.join(os.path.dirname(BOOKS), os.path.basename(pdf))
    if not os.path.exists(pdf):
        return False
    import pymupdf
    with pymupdf.open(pdf) as d:
        return len(book_text(txt)) < SCAN_CHARS_PER_PAGE * d.page_count


def book_scan_jobs():
    """Book PDFs without a usable text layer (scans, music scores) as raster PDF jobs. Language: BOOK_SCAN_LANGUAGES
    (checked by eye), else the name's code (name.<code>.pdf), else a non-Latin script in the name, else "unknown"."""
    import pymupdf
    jobs = []
    for pdf in sorted(glob.glob(os.path.join(os.path.dirname(BOOKS), "**", "*.pdf"), recursive=True)):
        rel = os.path.relpath(pdf, os.path.dirname(BOOKS))
        topic = os.path.dirname(rel) or "uncategorized"
        txt = os.path.join(BOOKS, topic, os.path.splitext(os.path.basename(rel))[0] + ".txt")
        try:
            with pymupdf.open(pdf) as d:
                n = d.page_count
        except Exception:
            continue
        if os.path.exists(txt) and len(book_text(txt)) >= SCAN_CHARS_PER_PAGE * n:
            continue
        name = os.path.basename(rel)
        code = re.search(r"\.([a-z]{2})(?:_\w+)?\.pdf$", name)
        other = script_language("".join(c for c in name if not c.isascii()))
        lang = next((v for k, v in BOOK_SCAN_LANGUAGES.items() if name.startswith(k)), None) or (
            code.group(1) if code else other if other not in ("latin", "unknown") else "unknown")
        jobs.append((pdf, n, lang, "book_scan", _seed(pdf)))
    return jobs


def _book_jobs():
    return [(p, book_language(p, LayoutCfg().scripts)) for p in book_txt_paths()]


def _job_book_text(args):
    """One entry per book; the plan draws snippets at distinct offsets of its `chars` characters."""
    txt_path, language = args
    return [dict(source_path=txt_path, page_index=0, language=language, category="book", kind="book_text",
                 chars=len(book_text(txt_path)))]


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def _run_group(name, jobs, job_fn, workers, out_f, counts, t0):
    print(f"[{name}] {len(jobs)} jobs", flush=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as pool:
        for done, recs in enumerate(pool.map(job_fn, jobs, chunksize=8), 1):
            for rec in recs:
                out_f.write(json.dumps(rec) + "\n")
                counts[rec["language"]] = counts.get(rec["language"], 0) + 1
            if done % 1000 == 0:
                print(f"[{name}] {done}/{len(jobs)} files, entries={sum(counts.values())} "
                      f"elapsed={time.time() - t0:.0f}s", flush=True)


def main(out_path: str, workers: int, limit_files: int = None):
    """Every usable page: PDF pages (PAGES_PER_PDF_CAP per document, all of rare-script ones), every image, and
    one entry per book text."""
    meta_idx = load_metadata(os.path.join(LAYOUTS, "corpus", "metadata.jsonl"))
    overflow_pdf_jobs, overflow_img_jobs = corpus_overflow_jobs()
    groups = [
        ("corpus_pdf", corpus_pdf_jobs(meta_idx), _job_pdf),
        ("overflow_pdf", overflow_pdf_jobs, _job_pdf),
        ("overflow_img", overflow_img_jobs, _job_image),
        ("pdf_png_en", pdf_png_jobs(), _job_image),
        ("xfund_funsd", xfund_funsd_jobs(), _job_image),
        ("arxiv_en", arxiv_jobs(), _job_pdf),
        ("book_scans", book_scan_jobs(), _job_pdf),
        ("books", _book_jobs(), _job_book_text),
    ]
    if limit_files:
        groups = [(name, jobs[:limit_files], fn) for name, jobs, fn in groups]
    t0, counts = time.time(), {}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out_f:
        for name, jobs, fn in groups:
            _run_group(name, jobs, fn, workers, out_f, counts, t0)
    print(f"TOTAL {sum(counts.values())} entries, wall={time.time() - t0:.0f}s", flush=True)
    print("language distribution:", json.dumps(counts, indent=2), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit_files", type=int, default=None,
                    help="cap per-source-group file count, for a fast dry run")
    args = ap.parse_args()
    main(args.out, args.workers, args.limit_files)
