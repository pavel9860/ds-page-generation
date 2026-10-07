"""Source manifest: one JSON line per usable page of the layout corpora (PDF pages, images).

Raster fields: source_path, page_index, language, category, kind, rotate90, deskew_deg, used_h, used_w (raster
size), window (x, y, w, h in page px: the window that passed the checks), margin, pictures, fill, picture_share.
Restyled pages (kind "refont", digital-born PDF pages set again in other fonts): seed, margin, font_pt (body), scale (other
sizes) and fill of the sheet (generation renders exactly that page). Rejected pages (and the check they failed) go to <out>.rejected.jsonl. No source
document holds more than DOC_SHARE of its script group.

python -m dspages.prep.manifest --out <paths.manifest>
"""
import argparse
import glob
import json
import os
import re
import time
import zlib
from collections import Counter
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
MIN_CONTENT_AREA_FRAC = 0.01
MAX_MARGIN_FRAC = 0.50
PAGES_PER_PDF_CAP = 10                 # accepted pages per PDF
TRIES_PER_PDF = 30                     # pages checked per PDF
DOC_SHARE = 0.02                       # max share of a script group's pages from one source document
CHECK_PX = 800                         # longest side of the PDF render the checks run on (none needs more)


def _init():
    cv2.setNumThreads(1)


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
                            c.min_fill, c.fill_grid, max_image)

    _, fill, _, ok = window(np.zeros_like(mask), 1.0)
    if not ok:
        return _reject(path, pi, language, "fill", fill=round(fill, 3))
    boxes = pictures(gray, c.pictures) * f if category != "sheet_music" else np.zeros((0, 4))  # staves read as pictures
    win, fill, share, ok = window(picture_mask(boxes, mask.shape, s), c.max_image)
    if not ok:
        return _reject(path, pi, language, "pictures", fill=round(fill, 3), picture_share=round(share, 3))
    angle = estimate_deskew_angle(gray) if scan else 0.0
    return dict(source_path=path, page_index=pi, language=language, category=category, kind="raster",
                rotate90=rot, deskew_deg=round(angle, 2) if needs_deskew(angle) else 0.0, used_h=round(gray.shape[0] * f),
                used_w=round(gray.shape[1] * f), window=[round(v, 1) for v in win], margin=round(_margin(box, win), 4),
                pictures=np.round(boxes, 1).tolist(), fill=round(fill, 3), picture_share=round(share, 3))


def _margin(box, win):
    """Widest blank margin of the window around the content box, as a fraction of the window width."""
    (x0, y0, bw, bh), (x, y, w, h) = box, win
    return float(max(x0 - x, x + w - x0 - bw, y0 - y, y + h - y0 - bh, 0.0) / w)


def _is_scan(has_text):
    """Pages that get rotation and deskew: PDF pages without a text layer and images."""
    return not has_text


def _box(b, s):
    """(x0, y0, w, h) in page px of a mask bbox (y0, y1, x0, x1) at scale s."""
    y0, y1, x0, x1 = (v / s for v in b)
    return x0, y0, x1 - x0 + 1 / s, y1 - y0 + 1 / s


def _bbox_keep(b, h, w):
    y0, y1, x0, x1 = b
    return ((y1 - y0 + 1) * (x1 - x0 + 1) >= MIN_CONTENT_AREA_FRAC * h * w
            and 1 - (y1 - y0 + 1) / h <= MAX_MARGIN_FRAC and 1 - (x1 - x0 + 1) / w <= MAX_MARGIN_FRAC)


_FONTS = []


def _refont_record(path, pi, language, category, seed):
    """A digital-born page restyled: the settings drawn (margin, body size, scale of the other sizes, a seed for the
    fonts), the sheet set and checked for fill and glyphs."""
    from ..layout import refont
    from ..layout.generate import sheet_fill
    c = _LAYOUT
    if not _FONTS:
        _FONTS.append(refont.font_set(Paths().font_dirs))
        refont.load_glyphs()
    rng = np.random.default_rng([seed, pi])
    m, pt, scale = rng.uniform(*c.margin_frac), rng.uniform(*c.font_pt), rng.uniform(*c.role_scale)
    sd = int(rng.integers(1 << 31))
    gray, info = refont.sheet(path, pi, sd, _FONTS[0], pt, scale, m, c.page_px)
    if info["unresolved"]:
        return _reject(path, pi, language, "glyphs")
    fill = sheet_fill(gray, c.fill_grid)
    if fill < c.min_fill:
        return _reject(path, pi, language, "fill", fill=round(fill, 3))
    return dict(source_path=path, page_index=pi, language=language, category=category, kind="refont", seed=sd,
                margin=float(m), font_pt=float(pt), scale=float(scale), fill=fill)


def _job_pdf(args):
    """Pages in random order, each handled once, until PAGES_PER_PDF_CAP pass; the rest of the document is not
    touched. A digital-born page of a restyled source is set again; other pages (scans, garbled text, many
    columns) are rasterized. Rejected pages come back as kind="rejected" records."""
    import pymupdf as fitz
    from ..layout import refont
    pdf_path, language, category, seed, digital = args
    try:
        doc = fitz.open(pdf_path)
        order = np.random.default_rng(seed).permutation(doc.page_count)[:TRIES_PER_PDF]
    except Exception:
        return []
    recs, n = [], 0
    try:
        for pi in order.tolist():
            if n >= PAGES_PER_PDF_CAP:
                break
            page = doc[pi]
            if digital and refont.usable(page):
                rec = _refont_record(pdf_path, pi, language, category, seed)
            else:
                z = min(RASTER_ZOOM, CHECK_PX / max(page.rect.width, page.rect.height))
                pix = page.get_pixmap(matrix=fitz.Matrix(z, z), colorspace=fitz.csGRAY, alpha=False)
                gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()
                rec = _page_record(gray, pdf_path, pi, language, category, _is_scan(bool(page.get_text().strip())),
                                   RASTER_ZOOM / z)
            recs.append(rec)
            n += rec["kind"] != "rejected"
    except Exception:
        return recs
    finally:
        doc.close()
        if _FONTS:
            refont.save_glyphs()
    return recs


def _job_image(args):
    image_path, language, category, _ = args
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return [_reject(image_path, 0, language, "unreadable")]
    return [_page_record(gray, image_path, 0, language, category, _is_scan(True))]


# ---------------------------------------------------------------------------
# source enumeration
# ---------------------------------------------------------------------------

DIGITAL = ("books_pdf", "forms", "other_docs", "magazines", "scientific_paper", "exams", "bureaucracy", "corpus_add/pdf")
RASTER = ("filled_forms", "flyers", "sheet_music", "overflow", "corpus_add/Images")
IMAGES = (".png", ".jpg", ".jpeg")


def _language(path):
    """Language code of a file named <hash>_<topic>_<code>[_p<page>].<ext>, else the old name.<code>.<ext>, else unknown."""
    code = re.sub(r"_p\d+$", "", os.path.splitext(os.path.basename(path))[0]).rsplit("_", 1)[-1]
    return code if re.fullmatch(r"[a-z]{2,3}", code) else name_language(path) or "unknown"


def _files(sub, exts):
    return sorted(p for p in glob.glob(os.path.join(LAYOUTS, sub, "**", "*"), recursive=True) if p.lower().endswith(exts))


def pdf_jobs():
    """(path, language, category, seed, digital): the digital-born sources' PDFs are restyled where they can be."""
    return [(p, _language(p), sub.split("/")[0], _seed(p), digital)
            for subs, digital in ((DIGITAL, True), (RASTER, False)) for sub in subs for p in _files(sub, (".pdf",))]


def image_jobs():
    return [(p, _language(p), sub.split("/")[0], _seed(p)) for sub in DIGITAL + RASTER for p in _files(sub, IMAGES)]


def balance(recs, scripts, rng):
    """Per script group at most DOC_SHARE of its pages (and at least PAGES_PER_PDF_CAP) from one source document,
    a random subset when over."""
    from ..conditions import script
    group = Counter(script(r, scripts) for r in recs)
    by_doc = {}
    for r in recs:
        by_doc.setdefault(r["source_path"], []).append(r)
    out = []
    for doc in by_doc.values():
        cap = max(PAGES_PER_PDF_CAP, round(DOC_SHARE * group[script(doc[0], scripts)]))
        out += [doc[i] for i in sorted(rng.permutation(len(doc))[:cap])]
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def _run_task(task):
    fn, args = task
    return fn(args)


def main(out_path: str, workers: int, limit_files: int = None):
    """Usable pages: up to PAGES_PER_PDF_CAP per PDF, every image; then balanced so no
    source document holds more than DOC_SHARE of its script group."""
    groups = [(pdf_jobs(), _job_pdf), (image_jobs(), _job_image)]
    rng = np.random.default_rng(0)
    tasks = [(fn, jobs[i]) for jobs, fn in groups for i in (rng.permutation(len(jobs))[:limit_files] if limit_files else range(len(jobs)))]
    rng.shuffle(tasks)                     # mixes heavy documents into the run
    t0, recs, rejected = time.time(), [], []
    print(f"{len(tasks)} files", flush=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as pool:
        for done, rs in enumerate(pool.map(_run_task, tasks, chunksize=1), 1):
            for r in rs:
                (rejected if r["kind"] == "rejected" else recs).append(r)
            if done % 1000 == 0:
                print(f"{done}/{len(tasks)} files, entries={len(recs)} elapsed={time.time() - t0:.0f}s", flush=True)
    kept = balance(recs, _LAYOUT.scripts, np.random.default_rng(0))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in kept)
    with open(Path(out_path).with_suffix(".rejected.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rejected)
    from ..conditions import script
    per = Counter((script(r, _LAYOUT.scripts), r["kind"]) for r in kept)
    print(f"TOTAL {len(kept)} entries ({len(recs) - len(kept)} over the per-document share), "
          f"wall={time.time() - t0:.0f}s", flush=True)
    print("per script group and kind:", json.dumps({f"{g} {k}": v for (g, k), v in sorted(per.items())}, indent=1),
          flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit_files", type=int, default=None,
                    help="cap per-source-group file count, for a fast dry run")
    args = ap.parse_args()
    main(args.out, args.workers, args.limit_files)
