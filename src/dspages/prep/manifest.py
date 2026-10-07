"""Source manifest: one JSON line per usable page of the layout corpora (PDF pages, images).

Raster fields: source_path, page_index, language, category, kind, rotate90, deskew_deg, used_h, used_w (raster
size), window (x, y, w, h in page px: the window that passed the checks), margin, fill, picture_share.
Restyled pages (kind "refont": pages of text set in the PDF, set again in other fonts, sizes and margins drawn per
sample at generation, their sheet filled to min_fill with pictures up to max_image, else rasterized): source_path, page_index, language, category,
fill. Rejected pages (and the check they failed) go to
<out>.rejected.jsonl.

python -m dspages.prep.manifest --out <paths.manifest>
"""
import argparse
import json
import time
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pymupdf as fitz

from ..conditions import script
from ..config import Paths, get_preset
from ..layout.content import bbox, content_mask, place_window, window_limits, window_stats
from ..layout import refont
from ..layout.deskew import detect_rotation, estimate_deskew_angle, needs_deskew, rotate90
from ..layout.pictures import picture_area
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


def _page_record(img, path, pi, language, category, scan, f=1.0):
    """Manifest record of a rasterized page (RASTER_ZOOM for PDFs), or a rejected record naming the first failed
    check: rotation (scans only), content box, size and fill of a sheet window (cropped or padded with margins),
    pictures (the detector runs only on pages that passed the rest). An accepted scan's skew angle is then
    estimated and stored (raster_page replays rotation and deskew). img (RGB) may be a reduced render: f page px per
    img px; sizes are stored in page px."""
    c = _LAYOUT
    pw, ph = c.page_px
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    rot = detect_rotation(gray) if scan else 0
    if rot:
        img, gray = rotate90(img, rot), rotate90(gray, rot)
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
    pics = picture_area(img, mask.shape, c.pictures) if category != "sheet_music" else np.zeros_like(mask)  # staves
    win, fill, share, ok = window(pics, c.max_image)
    if not ok:
        return _reject(path, pi, language, "pictures", fill=round(fill, 3), picture_share=round(share, 3))
    angle = estimate_deskew_angle(gray) if scan else 0.0
    return dict(source_path=path, page_index=pi, language=language, category=category, kind="raster",
                rotate90=rot, deskew_deg=round(angle, 2) if needs_deskew(angle) else 0.0, used_h=round(gray.shape[0] * f),
                used_w=round(gray.shape[1] * f), window=[round(v, 1) for v in win], margin=round(_margin(box, win), 4),
                fill=round(fill, 3), picture_share=round(share, 3))


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


def _restyled_ok(img, box, z):
    """Whether the sheet a restyled page makes (its content box (pt) width-fitted and top-aligned on a window of the
    sheet's aspect, measured on the RGB img at z px per pt) has the fill and picture share of a raster window;
    -> (ok, fill). Pictures are looked for only on pages that fill."""
    c = _LAYOUT
    pw, ph = c.page_px
    w = max(box.width, box.height * pw / ph)
    win = [((box.x0 + box.x1 - w) / 2 * z, box.y0 * z, w * z, w * ph / pw * z)]
    mask, s = content_mask(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    ink = cv2.integral(mask.astype(np.uint8)).astype(np.int64)
    fill = float(window_stats(ink, np.zeros_like(ink), s, win, c.fill_grid)[0][0])
    if fill < c.min_fill:
        return False, fill
    pics = cv2.integral(picture_area(img, mask.shape, c.pictures).astype(np.uint8)).astype(np.int64)
    return bool(window_stats(ink, pics, s, win, c.fill_grid)[1][0] <= c.max_image), fill


def _body_pt(src, pi, box):
    """Body size of a restyled page on the sheet (its content box fitted, no margin)."""
    c = _LAYOUT
    pw, ph = c.page_px
    return refont.body_size(refont.page_blocks(src, pi, True)) * min(pw, ph * box.width / box.height) / box.width * c.sheet_mm[0] / 25.4 * 72 / pw


def _job_pdf(args):
    """The pages of a PDF in random order, until `cap` are accepted or the tries are used. A page of text set in the PDF
    is restyled at generation; any other (a scan, garbled text, many columns) is rasterized."""
    path, language, category, digital, cap = args
    c = _LAYOUT
    try:
        doc = fitz.open(path)
    except fitz.FileDataError:
        return [_reject(path, 0, language, "unreadable")]
    latin = script(dict(language=language), c.scripts) in ("latin", "unknown")
    recs, n = [], 0
    with doc:
        src = refont.Source(doc)
        for pi in np.random.default_rng(_seed(path)).permutation(doc.page_count)[:c.tries_per_page * cap].tolist():
            page = doc[pi]
            z = min(RASTER_ZOOM, c.check_px / max(page.rect.width, page.rect.height))
            pix = page.get_pixmap(matrix=fitz.Matrix(z, z), colorspace=fitz.csRGB, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            box = refont.content_box(page)
            body = (_body_pt(src, pi, box) if digital and category not in c.raster_only and refont.usable(src, pi, latin)
                    else 0.0)
            ok, fill = _restyled_ok(img, box, z) if body else (False, 0.0)
            if ok:
                rec = dict(source_path=path, page_index=pi, language=language, category=category, kind="refont",
                           fill=round(fill, 3), body_pt=round(body, 1))
            else:
                rec = _page_record(img, path, pi, language, category, not page.get_text().strip(), RASTER_ZOOM / z)
            recs.append(rec)
            n += rec["kind"] != "rejected"
            if n == cap:
                break
    return recs


def _job_image(args):
    path, language, category, _, _ = args
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return [_reject(path, 0, language, "unreadable")]
    return [_page_record(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), path, 0, language, category, True)]


# ---------------------------------------------------------------------------
# target selection
# ---------------------------------------------------------------------------

BOOKS = "books_pdf"


def _water(total, sizes):
    """Equal shares of total between the keys of sizes, a key with less than its share giving the rest to the others."""
    out = {}
    for i, (k, size) in enumerate(sorted(sizes.items(), key=lambda kv: kv[1])):
        out[k] = min(size, total // (len(sizes) - i))
        total -= out[k]
    return out


def _pages(files):
    return sum(f["cap"] for f in files)


def _take(files, n):
    """Files popped from the list with the pages they give, n pages together."""
    taken = []
    while n > 0 and files:
        f = files.pop()
        taken.append(dict(f, cap=min(f["cap"], n)))
        n -= taken[-1]["cap"]
    return taken


def _allocate(tree, need):
    """The files of a round with their pages: each script group's need split equally between its languages, a language's
    between its topics, a topic's over its files."""
    batch = []
    for g, langs in tree.items():
        by_lang = {l: sum(_pages(fs) for fs in topics.values()) for l, topics in langs.items()}
        for l, n in _water(need[g], by_lang).items():
            for t, m in _water(n, {t: _pages(fs) for t, fs in langs[l].items()}).items():
                batch += _take(langs[l][t], m)
    return batch


def _rank(f):
    """Which files fill a target first: PDFs of text set in them (0), then the other PDFs (1), then images (2)."""
    return 0 if f["digital"] else 1 if f["path"].lower().endswith(".pdf") else 2


def _tree(files, c, quota):
    """script group -> language -> topic -> files, the first to take last: by _rank, in random order within a rank. Each
    file has the pages it may give: pages_per_pdf, a book its doc_share of its group's quota, an image one."""
    tree = {g: defaultdict(lambda: defaultdict(list)) for g, _ in c.script_mix}
    for i in sorted(np.random.default_rng(0).permutation(len(files)), key=lambda i: -_rank(files[i])):
        f = files[i]
        g = script(f, c.scripts)
        cap = max(c.pages_per_pdf, round(c.doc_share * quota[g])) if f["category"] == BOOKS else c.pages_per_pdf
        tree[g][f["language"]][f["topic"]].append(dict(f, cap=min(cap, f["pages"])))
    return tree


def _run(row):
    fn = _job_pdf if row["path"].lower().endswith(".pdf") else _job_image
    return fn((row["path"], row["language"], row["category"], row["digital"], row["cap"]))


def main(out_path, workers, pages=None):
    """Pages by the targets: each script group gets its share of `pages`, split equally between its languages and each
    language between its topics (a language or topic short of files leaves its share to the others). The listing gives
    the files; a round after another takes pages_per_pdf of each until the quotas are met. A page is taken once,
    restyled or not."""
    c = _LAYOUT
    quota = {g: round((pages or c.manifest_pages) * w) for g, w in c.script_mix}
    with open(Paths().files, encoding="utf-8") as f:
        tree = _tree([json.loads(line) for line in f], c, quota)
    got, recs, rejected, t0 = Counter(), [], [], time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as pool:
        while batch := _allocate(tree, {g: max(q - got[g], 0) for g, q in quota.items()}):
            for rs in pool.map(_run, batch, chunksize=1):
                for r in rs:
                    (rejected if r["kind"] == "rejected" else recs).append(r)
                    got[script(r, c.scripts)] += r["kind"] != "rejected"
            print(f"entries={len(recs)} elapsed={time.time() - t0:.0f}s", flush=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in recs)
    with open(Path(out_path).with_suffix(".rejected.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rejected)
    print(f"TOTAL {len(recs)} entries, {len(rejected)} rejected, wall={time.time() - t0:.0f}s", flush=True)
    print("per script group:", dict(got), "quota:", quota, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--pages", type=int, default=None, help="pages to collect (default: the config's manifest_pages)")
    args = ap.parse_args()
    main(args.out, args.workers, args.pages)
