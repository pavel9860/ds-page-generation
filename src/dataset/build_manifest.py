"""Fields: source_path, page_index, language, category, content_frac,
needs_deskew, used_h, used_w, bbox_y0, bbox_x0, bbox_h,
bbox_w, crop_y0, crop_size, kind.

Run:
    .venv/bin/python -m src.dataset.build_manifest \\
        --out /run/media/me/D/ML_DS/UVTM/Layouts/manifest_100k.jsonl --n 100000
"""
import argparse
import glob
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

LAYOUTS = "/run/media/me/D/ML_DS/UVTM/Layouts"
PAGES_PER_PDF_CAP = 3
RASTER_ZOOM = 2.0

EXCLUDED_PAGES = {
    (f"{LAYOUTS}/Pdf/6HPMFPOTKN7J772QGZBHKGKYSNEYTF3I_p87.png", 0),
    (f"{LAYOUTS}/corpus/forms_bulk/64b3012ead024750.pdf", 6),
    (f"{LAYOUTS}/corpus/forms_bulk/9e3e89ed7d9dac2a.pdf", 1),
}


def _init():
    cv2.setNumThreads(1)


def _load_metadata(meta_path: str) -> dict:
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


def _page_indices(page_count: int, cap: int = PAGES_PER_PDF_CAP) -> list:
    if page_count <= cap:
        return list(range(page_count))
    return sorted({int(round(i * (page_count - 1) / (cap - 1))) for i in range(cap)})


# ---------------------------------------------------------------------------
# per-page processing, shared by pdf and image sources
# ---------------------------------------------------------------------------

def _process_gray(gray: np.ndarray, rng):
    from src.dataset.content_filter import has_enough_content
    from src.dataset.crop import select_crop_1024

    if gray.shape[1] > gray.shape[0]:
        gray = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)

    bbox, content_frac, keep = has_enough_content(gray)
    if not keep:
        return None

    crop = select_crop_1024(gray, rng, bbox=bbox)
    if crop is None:
        return None
    return dict(content_frac=round(content_frac, 5), needs_deskew=True,
               used_h=gray.shape[0], used_w=gray.shape[1], crop_size=1024, **crop)


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
    """Content bbox + crop window from the page's own text-block
    coordinates -- no get_pixmap() call, so a born-digital page with a
    real text layer never gets rasterized during manifest building.
    None for a landscape page (falls back to the raster path). Shares
    its keep-check and window search with the pixel-based raster path
    (content_filter.bbox_keep, crop.select_window_in_mask) -- only the
    mask source (text-block occupancy vs. BlackHat/TopHat) differs."""
    from src.dataset.content_filter import _bbox_from_mask, bbox_keep
    from src.dataset.crop import CROP_SIZE, select_window_in_mask

    pw_pt, ph_pt = page.rect.width, page.rect.height
    if pw_pt > ph_pt:
        return None
    used_w, used_h = round(pw_pt * zoom), round(ph_pt * zoom)
    blocks = _pdf_text_blocks_px(page, zoom)
    if not blocks:
        return None

    coarse_scale = min(1.0, 512 / max(used_w, used_h))
    cw, ch = max(1, round(used_w * coarse_scale)), max(1, round(used_h * coarse_scale))
    coarse_mask = _occupancy(blocks, cw, ch, scale=coarse_scale)
    bbox = _bbox_from_mask(coarse_mask)
    if bbox is None:
        return None
    y0, y1, x0, x1 = bbox
    inv = 1.0 / coarse_scale
    by0, by1 = int(y0 * inv), min(used_h - 1, int(y1 * inv) + 1)
    bx0, bx1 = int(x0 * inv), min(used_w - 1, int(x1 * inv) + 1)
    bbox_h, bbox_w = by1 - by0 + 1, bx1 - bx0 + 1

    if not bbox_keep((by0, by1, bx0, bx1), used_h, used_w):
        return None

    size = CROP_SIZE
    strip_scale = size / bbox_w
    sh = max(1, round(bbox_h * strip_scale))
    strip_mask = _occupancy(blocks, size, sh, x_off=bx0, y_off=by0, scale=strip_scale)
    crop_y0 = select_window_in_mask(strip_mask, rng, size)
    if crop_y0 is None:
        return None

    return dict(content_frac=round(float(coarse_mask.mean()), 5), needs_deskew=False,
               used_h=used_h, used_w=used_w, crop_size=size,
               bbox_y0=by0, bbox_x0=bx0, bbox_h=bbox_h, bbox_w=bbox_w, crop_y0=crop_y0)


def _pdf_page_is_scan(page) -> bool:
    """A page with a real text layer can still be a scanned image with an
    OCR text overlay -- its own text bbox says nothing about visual skew
    in that case, so it needs the raster/deskew path, not the vector one."""
    from scripts.text_extract import DOMINANT_IMAGE_AREA_FRAC
    page_area = page.rect.width * page.rect.height
    if not page_area:
        return False
    return any((info["bbox"][2] - info["bbox"][0]) * (info["bbox"][3] - info["bbox"][1]) / page_area
              > DOMINANT_IMAGE_AREA_FRAC for info in page.get_image_info())


def _job_pdf(args, all_pages: bool = False):
    import fitz
    pdf_path, page_count_hint, language, category, seed = args
    rng = np.random.default_rng(seed)
    try:
        doc = fitz.open(pdf_path)
        page_indices = list(range(doc.page_count)) if all_pages else _page_indices(doc.page_count)
    except Exception:
        return []
    recs = []
    try:
        for pi in page_indices:
            if pi >= doc.page_count or (pdf_path, pi) in EXCLUDED_PAGES:
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


def _job_pdf_all(args):
    return _job_pdf(args, all_pages=True)


def _job_image(args):
    image_path, language, category, seed = args
    if (image_path, 0) in EXCLUDED_PAGES:
        return []
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

def _corpus_pdf_jobs(meta_idx: dict):
    jobs = []
    for sub in ("pdf", "pdf_flat", "scanned", "forms_bulk"):
        for p in sorted(glob.glob(os.path.join(LAYOUTS, "corpus", sub, "**", "*.pdf"), recursive=True)):
            m = meta_idx.get(p)
            language = m["language"] if m else "unknown"
            category = m.get("category") if m else None
            jobs.append((p, m.get("pages") if m else None, language, category, _seed(p)))
    return jobs


def _corpus_overflow_jobs():
    pdf_jobs, img_jobs = [], []
    for p in sorted(glob.glob(os.path.join(LAYOUTS, "corpus_overflow", "Images", "*.png"))):
        img_jobs.append((p, "unknown", None, _seed(p)))
    for p in sorted(glob.glob(os.path.join(LAYOUTS, "corpus_overflow", "pdf", "*"))):
        if p.lower().endswith(".pdf"):
            pdf_jobs.append((p, None, "unknown", None, _seed(p)))
        elif p.lower().endswith(".png"):
            img_jobs.append((p, "unknown", None, _seed(p)))
    return pdf_jobs, img_jobs


def _pdf_png_jobs():
    return [(p, "en", None, _seed(p))
           for p in sorted(glob.glob(os.path.join(LAYOUTS, "Pdf", "*.png")))]


def _xfund_funsd_jobs():
    """<lang>.<split>/*.png dirs -- language is the dir name's prefix
    before the dot (en = FUNSD, others = XFUND)."""
    root = os.path.join(LAYOUTS, "XFUND and FUNSD")
    jobs = []
    for p in sorted(glob.glob(os.path.join(root, "*", "*.png"))):
        lang = os.path.basename(os.path.dirname(p)).split(".")[0]
        jobs.append((p, lang, "form", _seed(p)))
    return jobs


def _arxiv_jobs():
    return [(p, None, "en", "scientific_paper", _seed(p))
           for p in sorted(glob.glob(os.path.join(LAYOUTS, "scientific_paper", "arxiv_pdfs", "*.pdf")))]


# ---------------------------------------------------------------------------
# book-text filler (step 3)
# ---------------------------------------------------------------------------

def _book_txt_paths():
    """Excludes files too short to fill a page -- same word-count bar
    text_texture.load_corpus already uses, applied here too since the
    filler pool is built independently of that function."""
    from src.synth.text_texture import _words, _SNIPPET_WORD_BUDGET
    paths = sorted(glob.glob(os.path.join(LAYOUTS, "books", "Texts", "**", "*.txt"), recursive=True))
    return [p for p in paths if len(_words(p)) >= _SNIPPET_WORD_BUDGET]


def _classify_book_language(path: str) -> str:
    """Coarse en / cyr / other, reusing export_uv_from_images.classify_language's
    approach (script + English-stopword density) collapsed to match the
    corpus's own language tags loosely (fine-grained EU languages in the
    real corpus can't be told apart this way, so filler books map only to
    "en"/"cyr"/"other" and get distributed by that coarse bucket)."""
    from scripts.export_uv_from_images import classify_language
    return classify_language(path)


def _build_filler_jobs(target_lang_counts: dict, n_needed: int, seed0: int):
    paths = _book_txt_paths()
    if not paths:
        return []
    buckets = {"en": [], "eu": [], "cyr": []}
    for p in paths:
        buckets[_classify_book_language(p)].append(p)
    # coarse-map real-corpus language tags to en/eu(other)/cyr buckets
    coarse = {"en": 0, "cyr": 0, "eu": 0}
    for lang, c in target_lang_counts.items():
        coarse["cyr" if lang == "cyr" else ("en" if lang == "en" else "eu")] += c
    total = sum(coarse.values()) or 1
    jobs = []
    idx = 0
    for bucket, frac_count in coarse.items():
        n_bucket = round(n_needed * frac_count / total)
        pool = buckets.get(bucket) or paths
        for _ in range(n_bucket):
            jobs.append((pool[idx % len(pool)], bucket, seed0 + idx))
            idx += 1
    return jobs


def _job_book_text(args):
    from src.synth import config as cfg

    txt_path, bucket, seed = args
    return [dict(source_path=txt_path, page_index=0, language=bucket, category="book_filler",
               kind="book_text", content_frac=None, needs_deskew=False,
               used_h=cfg.TEXT_PAGE_PX, used_w=cfg.TEXT_PAGE_PX, crop_y0=0, crop_x0=0,
               crop_size=cfg.TEXT_PAGE_PX, text_seed=seed)]


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def _run_group(name, jobs, job_fn, workers, out_f, kept, target, t0, lang_counts):
    if kept[0] >= target or not jobs:
        print(f"[{name}] skipped ({kept[0]}/{target} already, {len(jobs)} jobs)", flush=True)
        return
    print(f"[{name}] {len(jobs)} jobs", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as pool:
        for recs in pool.map(job_fn, jobs, chunksize=8):
            done += 1
            for rec in recs:
                if kept[0] >= target:
                    break
                out_f.write(json.dumps(rec) + "\n")
                lang_counts[rec["language"]] = lang_counts.get(rec["language"], 0) + 1
                kept[0] += 1
            if done % 1000 == 0 or kept[0] >= target:
                el = time.time() - t0
                print(f"[{name}] {done}/{len(jobs)} files, kept={kept[0]} elapsed={el:.0f}s", flush=True)
            if kept[0] >= target:
                break
    print(f"[{name}] done: kept={kept[0]}", flush=True)


SCIENTIFIC_PAPER_QUOTA_FRAC = 0.02


def main(out_path: str, n: int, workers: int, limit_files: int = None):
    meta_idx = _load_metadata(os.path.join(LAYOUTS, "corpus", "metadata.jsonl"))
    overflow_pdf_jobs, overflow_img_jobs = _corpus_overflow_jobs()
    groups = [
        ("corpus_pdf", _corpus_pdf_jobs(meta_idx), _job_pdf_all),
        ("overflow_pdf", overflow_pdf_jobs, _job_pdf_all),
        ("overflow_img", overflow_img_jobs, _job_image),
        ("pdf_png_en", _pdf_png_jobs(), _job_image),
        ("xfund_funsd", _xfund_funsd_jobs(), _job_image),
        ("arxiv_en", _arxiv_jobs(), _job_pdf),
    ]
    if limit_files:
        groups = [(name, jobs[:limit_files], fn) for name, jobs, fn in groups]

    kept = [0]
    t0 = time.time()
    lang_counts = {}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out_f:
        for name, jobs, fn in groups:
            target = (min(n, kept[0] + round(n * SCIENTIFIC_PAPER_QUOTA_FRAC))
                     if name == "arxiv_en" else n)
            _run_group(name, jobs, fn, workers, out_f, kept, target, t0, lang_counts)

        if kept[0] < n and not limit_files:
            filler_jobs = _build_filler_jobs(lang_counts, round((n - kept[0]) * 1.2), seed0=12345)
            _run_group("book_filler", filler_jobs, _job_book_text, workers, out_f, kept, n, t0,
                      lang_counts)

    print(f"TOTAL kept={kept[0]} wall={time.time() - t0:.0f}s", flush=True)
    print("language distribution:", json.dumps(lang_counts, indent=2), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit_files", type=int, default=None,
                    help="cap per-source-group file count, for a fast dry run")
    args = ap.parse_args()
    main(args.out, args.n, args.workers, args.limit_files)
