"""Source manifest: one JSON line per usable page of the layout corpora (PDF pages, images, book texts).

Raster fields: source_path, page_index, language, category, kind, rotate90, deskew_deg, used_h, used_w (raster
size), bbox_w, bbox_h (content box), fill and picture_share of the window check. Book texts: one entry each
with chars. Rejected pages (and the check they failed) go to <out>.rejected.jsonl.

python -m dspages.prep.manifest --out <paths.manifest>
"""
import argparse
import glob
import json
import os
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from ..config import Paths, get_preset
from ..layout.content import bbox, content_mask, place_window, window_limits
from ..layout.deskew import detect_rotation, estimate_deskew_angle, needs_deskew, rotate90
from ..layout.pictures import picture_mask, pictures
from ..layout.sources import RASTER_ZOOM
from .text_extract import name_language

LAYOUTS = Paths().layouts
BOOKS = Paths().books
MIN_CONTENT_AREA_FRAC = 0.01
MAX_MARGIN_FRAC = 0.50
PAGES_PER_PDF_CAP = 10                 # accepted pages per PDF
CHECK_PX = 800                         # longest side of the PDF render the checks run on (none needs more)


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
    return zlib.crc32(path.encode())


# ---------------------------------------------------------------------------
# per-page processing, shared by pdf and image sources
# ---------------------------------------------------------------------------

_LAYOUT = get_preset("full").layout


def _reject(path, pi, language, reason, **kw):
    return dict(kind="rejected", source_path=path, page_index=pi, language=language, reason=reason, **kw)


def _page_record(gray, path, pi, language, category, scan, f=1.0):
    """Manifest record of a rasterized page (RASTER_ZOOM for PDFs), or a rejected record naming the first failed
    check: rotation (scans only), content box, size and fill of a sheet window (cropped or padded with margins),
    pictures (the detector runs only on pages that passed the rest). An accepted scan's skew angle is then
    estimated and stored (raster_page replays rotation and deskew). gray may be a reduced render: f page px per
    gray px; boxes and sizes are stored in page px."""
    c = _LAYOUT
    pw, ph = c.page_px
    rot = detect_rotation(gray) if scan else 0
    if rot:
        gray = rotate90(gray, rot)
    mask, s = content_mask(gray)
    b = bbox(mask)
    if b is None or not _bbox_keep(b, *mask.shape):
        return _reject(path, pi, language, "content box")
    s /= f
    box = _box(b, s)
    cap = c.max_zoom if path.lower().endswith(".pdf") else 1.0
    if window_limits(box[2], box[3], pw / ph, c.margin_frac[1])[1] * cap < pw:
        return _reject(path, pi, language, "too small")

    def window(pics, max_image):
        return place_window(np.random.default_rng(0), box, pw / ph, c.margin_frac[1], pw / cap, mask, s, pics,
                            c.min_fill, c.fill_grid, max_image)[1:]

    fill, _, ok = window(np.zeros_like(mask), 1.0)
    if not ok:
        return _reject(path, pi, language, "fill", fill=round(fill, 3))
    boxes = pictures(gray, c.pictures) * f if category != "sheet_music" else np.zeros((0, 4))  # staves read as pictures
    fill, share, ok = window(picture_mask(boxes, mask.shape, s), c.max_image)
    if not ok:
        return _reject(path, pi, language, "pictures", fill=round(fill, 3), picture_share=round(share, 3))
    angle = estimate_deskew_angle(gray) if scan else 0.0
    return dict(source_path=path, page_index=pi, language=language, category=category, kind="raster",
                rotate90=rot, deskew_deg=round(angle, 2) if needs_deskew(angle) else 0.0, used_h=round(gray.shape[0] * f),
                used_w=round(gray.shape[1] * f), bbox_w=round(box[2]), bbox_h=round(box[3]),
                pictures=np.round(boxes, 1).tolist(), fill=round(fill, 3), picture_share=round(share, 3))


def _is_scan(category, has_text):
    """Pages that get rotation and deskew: document PDF pages without a text layer. Book scans and images don't."""
    return category != "book_scan" and not has_text


def _box(b, s):
    """(x0, y0, w, h) in page px of a mask bbox (y0, y1, x0, x1) at scale s."""
    y0, y1, x0, x1 = (v / s for v in b)
    return x0, y0, x1 - x0 + 1 / s, y1 - y0 + 1 / s


def _bbox_keep(b, h, w):
    y0, y1, x0, x1 = b
    return ((y1 - y0 + 1) * (x1 - x0 + 1) >= MIN_CONTENT_AREA_FRAC * h * w
            and 1 - (y1 - y0 + 1) / h <= MAX_MARGIN_FRAC and 1 - (x1 - x0 + 1) / w <= MAX_MARGIN_FRAC)


def _job_pdf(args):
    """Pages in random order, each rendered once, until PAGES_PER_PDF_CAP pass; the rest of the document is not
    touched. Rejected pages come back as kind="rejected" records."""
    import pymupdf as fitz
    pdf_path, _, language, category, seed = args
    language = name_language(pdf_path) or language
    try:
        doc = fitz.open(pdf_path)
        order = np.random.default_rng(seed).permutation(doc.page_count)
    except Exception:
        return []
    recs, n = [], 0
    try:
        for pi in order.tolist():
            if n >= PAGES_PER_PDF_CAP:
                break
            page = doc[pi]
            z = min(RASTER_ZOOM, CHECK_PX / max(page.rect.width, page.rect.height))
            pix = page.get_pixmap(matrix=fitz.Matrix(z, z), colorspace=fitz.csGRAY, alpha=False)
            gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()
            rec = _page_record(gray, pdf_path, pi, language, category,
                               _is_scan(category, bool(page.get_text().strip())), RASTER_ZOOM / z)
            recs.append(rec)
            n += rec["kind"] == "raster"
    except Exception:
        return recs
    finally:
        doc.close()
    return recs


def _job_image(args):
    image_path, language, category, _ = args
    language = name_language(image_path) or language
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return [_reject(image_path, 0, language, "unreadable")]
    return [_page_record(gray, image_path, 0, language, category, _is_scan(category, True))]


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


def sheet_music_jobs():
    return [(p, None, "xx", "sheet_music", _seed(p))
            for p in sorted(glob.glob(os.path.join(LAYOUTS, "sheet_music", "*.pdf")))]


def arxiv_jobs():
    return [(p, None, "en", "scientific_paper", _seed(p))
            for p in sorted(glob.glob(os.path.join(LAYOUTS, "scientific_paper", "arxiv_pdfs", "*.pdf")))]


# ---------------------------------------------------------------------------
# books: read from the index written once by prep.books
# ---------------------------------------------------------------------------

def book_index():
    path = os.path.join(BOOKS, "index.jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def book_scan_jobs(index=None):
    return [(e["pdf"], e["pages"], e["language"], "book_scan", _seed(e["pdf"])) for e in index or book_index()
            if e["kind"] == "scan"]


def book_text_record(e):
    """The index line of a text book as its manifest entry (the plan draws snippets at distinct offsets)."""
    return dict(source_path=e["txt"], page_index=0, language=e["language"], category="book", kind="book_text",
                chars=e["chars"])


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def _run_task(task):
    fn, args = task
    return fn(args)


def main(out_path: str, workers: int, limit_files: int = None):
    """Usable pages: up to PAGES_PER_PDF_CAP per PDF, every image, and one entry per book text."""
    meta_idx = load_metadata(os.path.join(LAYOUTS, "corpus", "metadata.jsonl"))
    books = book_index()
    overflow_pdf_jobs, overflow_img_jobs = corpus_overflow_jobs()
    groups = [
        (corpus_pdf_jobs(meta_idx), _job_pdf),
        (overflow_pdf_jobs, _job_pdf),
        (overflow_img_jobs, _job_image),
        (pdf_png_jobs(), _job_image),
        (xfund_funsd_jobs(), _job_image),
        (arxiv_jobs(), _job_pdf),
        (sheet_music_jobs(), _job_pdf),
        (book_scan_jobs(books), _job_pdf),
    ]
    tasks = [(fn, job) for jobs, fn in groups for job in jobs[:limit_files]]
    np.random.default_rng(0).shuffle(tasks)                     # mixes heavy groups (book scans) into the run
    texts = [book_text_record(e) for e in books if e["kind"] == "text"][:limit_files]
    t0, counts = time.time(), {}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out_f, \
            open(Path(out_path).with_suffix(".rejected.jsonl"), "w", encoding="utf-8") as rej_f:

        def write(rec):
            if rec["kind"] == "rejected":
                rej_f.write(json.dumps(rec) + "\n")
                return
            out_f.write(json.dumps(rec) + "\n")
            counts[rec["language"]] = counts.get(rec["language"], 0) + 1

        for rec in texts:
            write(rec)
        print(f"{len(tasks)} files, {len(texts)} book texts", flush=True)
        with ProcessPoolExecutor(max_workers=workers, initializer=_init) as pool:
            for done, recs in enumerate(pool.map(_run_task, tasks, chunksize=1), 1):
                for rec in recs:
                    write(rec)
                if done % 1000 == 0:
                    print(f"{done}/{len(tasks)} files, entries={sum(counts.values())} "
                          f"elapsed={time.time() - t0:.0f}s", flush=True)
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
