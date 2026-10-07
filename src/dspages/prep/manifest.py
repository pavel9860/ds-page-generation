"""Source manifest: one JSON line per usable page of the layout corpora (PDF pages, images).

Raster fields: source_path, page_index, language, category, kind, rotate90, deskew_deg, used_h, used_w (raster
size), window (x, y, w, h in page px: the window that passed the checks), margin, pictures, fill, picture_share.
Restyled pages (kind "refont": pages of text set in the PDF, set again in other fonts, sizes and margins drawn per
sample at generation): source_path, page_index, language, category. Rejected pages (and the check they failed) go to
<out>.rejected.jsonl.

python -m dspages.prep.manifest --out <paths.manifest>
"""
import argparse
import glob
import json
import math
import os
import re
import time
import zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pymupdf as fitz

from ..conditions import script
from ..config import Paths, get_preset
from ..layout.content import bbox, content_mask, place_window, window_limits
from ..layout import refont
from ..layout.deskew import detect_rotation, estimate_deskew_angle, needs_deskew, rotate90
from ..layout.pictures import picture_mask, pictures
from ..layout.sources import RASTER_ZOOM

LAYOUTS = Paths().layouts


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


def _box(b, s):
    """(x0, y0, w, h) in page px of a mask bbox (y0, y1, x0, x1) at scale s."""
    y0, y1, x0, x1 = (v / s for v in b)
    return x0, y0, x1 - x0 + 1 / s, y1 - y0 + 1 / s


def _bbox_keep(b, h, w):
    y0, y1, x0, x1 = b
    return ((y1 - y0 + 1) * (x1 - x0 + 1) >= _LAYOUT.min_content_area * h * w
            and 1 - (y1 - y0 + 1) / h <= _LAYOUT.max_blank_margin and 1 - (x1 - x0 + 1) / w <= _LAYOUT.max_blank_margin)


def _job_pdf(args):
    """The pages of a PDF in random order, until `cap` are accepted or the tries are used. A page of text set in the PDF
    is restyled at generation; any other (a scan, garbled text, many columns) is rasterized."""
    path, language, category, digital, cap = args
    c = _LAYOUT
    try:
        doc = fitz.open(path)
    except fitz.FileDataError:
        return [_reject(path, 0, language, "unreadable")]
    recs, n = [], 0
    with doc:
        for pi in np.random.default_rng(_seed(path)).permutation(doc.page_count)[:c.tries_per_page * cap].tolist():
            page = doc[pi]
            if digital and refont.usable(page):
                rec = dict(source_path=path, page_index=pi, language=language, category=category, kind="refont")
            else:
                z = min(RASTER_ZOOM, c.check_px / max(page.rect.width, page.rect.height))
                pix = page.get_pixmap(matrix=fitz.Matrix(z, z), colorspace=fitz.csGRAY, alpha=False)
                gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
                rec = _page_record(gray, path, pi, language, category, not page.get_text().strip(), RASTER_ZOOM / z)
            recs.append(rec)
            n += rec["kind"] != "rejected"
            if n == cap:
                break
    return recs


def _job_image(args):
    path, language, category, _, _ = args
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return [_reject(path, 0, language, "unreadable")]
    return [_page_record(gray, path, 0, language, category, True)]


# ---------------------------------------------------------------------------
# source files
# ---------------------------------------------------------------------------

DIGITAL = ("books_pdf", "bureaucracy", "corpus_add/pdf", "exams", "other_docs", "scientific_paper")  # restyled
RASTER = ("certificates", "corpus_add/Images")                                                       # as images
BOOKS = "books_pdf"


def _language(path):
    """Language code of a file named <hash>_<topic>_<code>[_p<page>].<ext>."""
    code = re.sub(r"_p\d+$", "", os.path.splitext(os.path.basename(path))[0]).rsplit("_", 1)[-1]
    return code if re.fullmatch(r"[a-z]{2,3}", code) else "unknown"


def list_files(path):
    """The source files (path, language, category, digital) of the listing file, made when missing."""
    if not os.path.exists(path):
        rows = [dict(path=p, language=_language(p), category=sub.split("/")[0], digital=digital and p.lower().endswith(".pdf"))
                for subs, digital in ((DIGITAL, True), (RASTER, False)) for sub in subs
                for p in sorted(glob.glob(os.path.join(LAYOUTS, sub, "**", "*"), recursive=True))
                if p.lower().endswith((".pdf", ".png", ".jpg", ".jpeg"))]
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def _run(row):
    fn = _job_pdf if row["path"].lower().endswith(".pdf") else _job_image
    return fn((row["path"], row["language"], row["category"], row["digital"], row["cap"]))


def _batch(queue, group, got, quota):
    """The files of one round, each with the pages it may give: every language of a script group with files left asks
    for what it lacks to an equal share of the group's quota (a language without files leaves its share to the others)."""
    c = _LAYOUT
    batch = []
    for g, q in quota.items():
        langs = [l for l in queue if group[l] == g and queue[l]]
        for l in langs:
            lack, used = q / len(langs) - got[l], 0
            for f in queue[l]:
                if lack <= 0:
                    break
                cap = max(c.pages_per_pdf, round(c.doc_share * q)) if f["category"] == BOOKS else c.pages_per_pdf
                batch.append(dict(f, cap=min(cap, math.ceil(lack))))
                lack -= batch[-1]["cap"]
                used += 1
            queue[l] = queue[l][used:]
    return batch


def main(out_path, workers, pages=None):
    """Pages by the targets: each script group gets its share of `pages`, split equally between its languages. A
    language's files are taken in random order, pages_per_pdf pages of each (a book: its share of the group), a round
    after another until the shares are met. A page is taken once, restyled or not."""
    c = _LAYOUT
    files = list_files(Paths().files)
    quota = {g: round((pages or c.manifest_pages) * w) for g, w in c.script_mix}
    group = {f["language"]: script(f, c.scripts) for f in files}
    queue = {l: [] for l in group}
    for i in np.random.default_rng(0).permutation(len(files)):
        queue[files[i]["language"]].append(files[i])
    got, recs, rejected, t0 = Counter(), [], [], time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as pool:
        while batch := _batch(queue, group, got, quota):
            for rs in pool.map(_run, batch, chunksize=1):
                for r in rs:
                    (rejected if r["kind"] == "rejected" else recs).append(r)
                    got[r["language"]] += r["kind"] != "rejected"
            print(f"entries={len(recs)} elapsed={time.time() - t0:.0f}s", flush=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in recs)
    with open(Path(out_path).with_suffix(".rejected.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rejected)
    print(f"TOTAL {len(recs)} entries, {len(rejected)} rejected, wall={time.time() - t0:.0f}s", flush=True)
    print("per language:", dict(got), "quota:", quota, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--pages", type=int, default=None, help="pages to collect (default: the config's manifest_pages)")
    args = ap.parse_args()
    main(args.out, args.workers, args.pages)
