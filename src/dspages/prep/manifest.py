"""Source manifest: one JSON line per usable page of the layout corpora (PDF pages, images, book texts).

Raster fields: source_path, page_index, language, category, kind, rotate90, deskew_deg, used_h, used_w (raster
size before deskew), bbox_y0, bbox_x0, bbox_h, bbox_w (content box after deskew), pictures (boxes), fill and
picture_share of the layout's window check. Book texts: one entry each with chars. Rejected pages (and the check
they failed) go to <out>.rejected.jsonl.

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
from ..config import LayoutCfg, Paths, get_preset, page_px
from ..layout.content import bbox, content_mask, place_window, window_limits
from ..layout.deskew import detect_rotation, estimate_deskew_angle, needs_deskew, rotate90, rotate_full_res
from ..layout.pictures import picture_mask, pictures
from ..layout.sources import RASTER_ZOOM
from ..layout.text import _words, book_text
from .text_extract import DOMINANT_IMAGE_AREA_FRAC, book_language, script_language

LAYOUTS = Paths().layouts
BOOKS = Paths().books
MIN_CONTENT_AREA_FRAC = 0.01
MAX_MARGIN_FRAC = 0.50
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


# ---------------------------------------------------------------------------
# per-page processing, shared by pdf and image sources
# ---------------------------------------------------------------------------

_LAYOUT = get_preset("full").layout


def _reject(path, pi, language, reason, **kw):
    return dict(kind="rejected", source_path=path, page_index=pi, language=language, reason=reason, **kw)


def _page_record(gray, path, pi, language, category, scan):
    """Manifest record of a rasterized page (RASTER_ZOOM for PDFs), or a rejected record naming the first failed
    check, cheapest first: content box, size (a window reaches the sheet width at 1:1, within max_zoom for PDF
    pages), fill without pictures, then pictures (the detector runs only on pages that passed the rest). Scans
    are turned upright and deskewed first (the stored angles are replayed by raster_page); born-digital PDF
    pages are taken as they are."""
    c = _LAYOUT
    pw, ph = page_px(c.sheet_mm, c.canvas_px)
    rot = detect_rotation(gray) if scan else 0
    gray = rotate90(gray, rot)
    used_h, used_w = gray.shape
    angle = estimate_deskew_angle(gray) if scan else 0.0
    angle = angle if needs_deskew(angle) else 0.0
    if angle:
        gray = rotate_full_res(gray, angle, fill_value=255)
    mask, s = content_mask(gray)
    b = bbox(mask)
    if b is None or not _bbox_keep(b, *mask.shape):
        return _reject(path, pi, language, "content box")
    y0, y1, x0, x1 = (v / s for v in b)
    box = (x0, y0, x1 - x0 + 1 / s, y1 - y0 + 1 / s)
    pdf = path.lower().endswith(".pdf")
    cap = c.max_zoom if pdf else 1.0
    if window_limits(box[2], box[3], pw / ph, c.margin_frac[1])[1] * cap < pw:
        return _reject(path, pi, language, "too small")
    args = (pw / ph, c.margin_frac[1], pw / cap, mask, s)
    none = np.zeros_like(mask)
    _, fill, _, ok = place_window(np.random.default_rng(0), box, *args, none, c.min_fill, c.fill_grid, 1.0)
    if not ok:
        return _reject(path, pi, language, "fill", fill=round(fill, 3))
    boxes = pictures(gray, c.pictures)
    _, fill, share, ok = place_window(np.random.default_rng(0), box, *args, picture_mask(boxes, mask.shape, s),
                                      c.min_fill, c.fill_grid, c.max_image)
    if not ok:
        return _reject(path, pi, language, "pictures", fill=round(fill, 3), picture_share=round(share, 3))
    y0, x0 = round(y0), round(x0)
    return dict(source_path=path, page_index=pi, language=language, category=category, kind="raster",
                rotate90=rot, deskew_deg=round(angle, 2), used_h=used_h, used_w=used_w,
                bbox_y0=y0, bbox_x0=x0, bbox_h=round(box[3]), bbox_w=round(box[2]),
                pictures=np.round(boxes, 1).tolist(), fill=round(fill, 3), picture_share=round(share, 3))


def _bbox_keep(b, h, w):
    y0, y1, x0, x1 = b
    return ((y1 - y0 + 1) * (x1 - x0 + 1) >= MIN_CONTENT_AREA_FRAC * h * w
            and 1 - (y1 - y0 + 1) / h <= MAX_MARGIN_FRAC and 1 - (x1 - x0 + 1) / w <= MAX_MARGIN_FRAC)


def _pdf_page_is_scan(page) -> bool:
    """A page dominated by an image (a scan, possibly with an OCR text layer) needs upright/deskew handling."""
    page_area = page.rect.width * page.rect.height
    if not page_area:
        return False
    return any((info["bbox"][2] - info["bbox"][0]) * (info["bbox"][3] - info["bbox"][1]) / page_area
               > DOMINANT_IMAGE_AREA_FRAC for info in page.get_image_info())


def _job_pdf(args):
    """Pages in random order until PAGES_PER_PDF_CAP pass, or all pages of a chunk (start, end) of a rare-script
    document; each page rendered once. Rejected pages come back as kind="rejected" records."""
    import pymupdf as fitz
    pdf_path, _, language, category, seed, *chunk = args
    try:
        doc = fitz.open(pdf_path)
        order = np.arange(*chunk[0]) if chunk else np.random.default_rng(seed).permutation(doc.page_count)
    except Exception:
        return []
    cap = None if chunk else PAGES_PER_PDF_CAP
    recs, n = [], 0
    try:
        for pi in order.tolist():
            if cap is not None and n >= cap:
                break
            page = doc[pi]
            pix = page.get_pixmap(matrix=fitz.Matrix(RASTER_ZOOM, RASTER_ZOOM), colorspace=fitz.csGRAY, alpha=False)
            gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()
            rec = _page_record(gray, pdf_path, pi, language, category, _pdf_page_is_scan(page))
            recs.append(rec)
            n += rec["kind"] == "raster"
    except Exception:
        return recs
    finally:
        doc.close()
    return recs


def _job_image(args):
    image_path, language, category, _ = args
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return [_reject(image_path, 0, language, "unreadable")]
    return [_page_record(gray, image_path, 0, language, category, True)]


RARE_CHUNK = 50                       # pages per job of rare-script documents (all pages are used)


def _chunk_rare(jobs):
    """PDF jobs with rare-script documents split into RARE_CHUNK-page jobs, so one long book does not hold a
    worker for the whole run."""
    import pymupdf as fitz
    out = []
    for j in jobs:
        if script(dict(language=j[2]), LayoutCfg().scripts) not in RARE_SCRIPTS:
            out.append(j)
            continue
        try:
            with fitz.open(j[0]) as d:
                n = d.page_count
        except Exception:
            continue
        out += [(*j, (a, min(a + RARE_CHUNK, n))) for a in range(0, n, RARE_CHUNK)]
    return out


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

def _run_group(name, jobs, job_fn, workers, out_f, rej_f, counts, t0):
    print(f"[{name}] {len(jobs)} jobs", flush=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as pool:
        for done, recs in enumerate(pool.map(job_fn, jobs, chunksize=8), 1):
            for rec in recs:
                if rec["kind"] == "rejected":
                    rej_f.write(json.dumps(rec) + "\n")
                    continue
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
    groups = [(name, _chunk_rare(jobs) if fn is _job_pdf else jobs, fn) for name, jobs, fn in groups]
    t0, counts = time.time(), {}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out_f, \
            open(Path(out_path).with_suffix(".rejected.jsonl"), "w", encoding="utf-8") as rej_f:
        for name, jobs, fn in groups:
            _run_group(name, jobs, fn, workers, out_f, rej_f, counts, t0)
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
