"""Fields: source_path, page_index, language, category, content_frac,
needs_deskew, deskew_angle_deg, used_h, used_w, bbox_y0, bbox_x0, bbox_h,
bbox_w, crop_y0, crop_size, kind.

Run:
    .venv/bin/python -m src.dataset.build_manifest \\
        --out /run/media/me/D/ML_DS/UVTM/Layouts/manifest_100k.jsonl --n 100000
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

LAYOUTS = "/run/media/me/D/ML_DS/UVTM/Layouts"
PAGES_PER_PDF_CAP = 3
RASTER_ZOOM = 2.0


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
    return dict(content_frac=round(content_frac, 5), needs_deskew=False, deskew_angle_deg=0.0,
               used_h=gray.shape[0], used_w=gray.shape[1], crop_size=1024, **crop)


_CYR_RE = re.compile("[Ѐ-ӿ]")
_STOP_EN = set("the and of to in is was that he she it with as for on at by "
              "an be this his her they were are not".split())


def _pdf_gray_pages(pdf_path: str, page_indices, zoom=RASTER_ZOOM):
    import fitz
    out = []
    with fitz.open(pdf_path) as doc:
        for pi in page_indices:
            if pi >= doc.page_count:
                continue
            pix = doc[pi].get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csGRAY, alpha=False)
            gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()
            out.append((pi, gray))
    return out


def _job_pdf(args):
    pdf_path, page_count_hint, language, category, seed = args
    rng = np.random.default_rng(seed)
    try:
        import fitz
        with fitz.open(pdf_path) as doc:
            page_indices = _page_indices(doc.page_count)
    except Exception:
        return []
    recs = []
    try:
        for pi, gray in _pdf_gray_pages(pdf_path, page_indices):
            info = _process_gray(gray, rng)
            if info is None:
                continue
            recs.append(dict(source_path=pdf_path, page_index=pi, language=language,
                             category=category, kind="raster", **info))
    except Exception:
        return recs
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

def _corpus_pdf_jobs(meta_idx: dict):
    jobs = []
    for sub in ("pdf", "pdf_flat", "scanned", "forms_bulk"):
        for p in sorted(glob.glob(os.path.join(LAYOUTS, "corpus", sub, "**", "*.pdf"), recursive=True)):
            m = meta_idx.get(p)
            language = m["language"] if m else "unknown"
            category = m.get("category") if m else None
            jobs.append((p, m.get("pages") if m else None, language, category, hash(p) & 0xFFFFFFFF))
    return jobs


def _corpus_overflow_jobs():
    pdf_jobs, img_jobs = [], []
    for p in sorted(glob.glob(os.path.join(LAYOUTS, "corpus_overflow", "Images", "*.png"))):
        img_jobs.append((p, "unknown", None, hash(p) & 0xFFFFFFFF))
    for p in sorted(glob.glob(os.path.join(LAYOUTS, "corpus_overflow", "pdf", "*"))):
        if p.lower().endswith(".pdf"):
            pdf_jobs.append((p, None, "unknown", None, hash(p) & 0xFFFFFFFF))
        elif p.lower().endswith(".png"):
            img_jobs.append((p, "unknown", None, hash(p) & 0xFFFFFFFF))
    return pdf_jobs, img_jobs


def _pdf_png_jobs():
    return [(p, "en", None, hash(p) & 0xFFFFFFFF)
           for p in sorted(glob.glob(os.path.join(LAYOUTS, "Pdf", "*.png")))]


def _xfund_funsd_jobs():
    """<lang>.<split>/*.png dirs -- language is the dir name's prefix
    before the dot (en = FUNSD, others = XFUND)."""
    root = os.path.join(LAYOUTS, "XFUND and FUNSD")
    jobs = []
    for p in sorted(glob.glob(os.path.join(root, "*", "*.png"))):
        lang = os.path.basename(os.path.dirname(p)).split(".")[0]
        jobs.append((p, lang, "form", hash(p) & 0xFFFFFFFF))
    return jobs


def _arxiv_jobs():
    return [(p, None, "en", "scientific_paper", hash(p) & 0xFFFFFFFF)
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
               kind="book_text", content_frac=None, needs_deskew=False, deskew_angle_deg=0.0,
               used_h=cfg.TEXT_PAGE_PX, used_w=cfg.TEXT_PAGE_PX, crop_y0=0, crop_x0=0,
               crop_size=cfg.TEXT_PAGE_PX, text_seed=seed)]


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def _run_group(name, jobs, job_fn, workers, out_f, kept, target, t0, lang_counts, returns_list=True):
    if kept[0] >= target or not jobs:
        print(f"[{name}] skipped ({kept[0]}/{target} already, {len(jobs)} jobs)", flush=True)
        return
    print(f"[{name}] {len(jobs)} jobs", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as pool:
        for result in pool.map(job_fn, jobs, chunksize=8):
            done += 1
            recs = result if returns_list else [result]
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
    corpus_pdf_jobs = _corpus_pdf_jobs(meta_idx)
    overflow_pdf_jobs, overflow_img_jobs = _corpus_overflow_jobs()
    pdf_png_jobs = _pdf_png_jobs()
    xfund_funsd_jobs = _xfund_funsd_jobs()
    arxiv_jobs = _arxiv_jobs()

    if limit_files:
        corpus_pdf_jobs = corpus_pdf_jobs[:limit_files]
        overflow_pdf_jobs = overflow_pdf_jobs[:limit_files]
        overflow_img_jobs = overflow_img_jobs[:limit_files]
        pdf_png_jobs = pdf_png_jobs[:limit_files]
        xfund_funsd_jobs = xfund_funsd_jobs[:limit_files]
        arxiv_jobs = arxiv_jobs[:limit_files]

    kept = [0]
    t0 = time.time()
    lang_counts = {}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out_f:
        _run_group("corpus_pdf", corpus_pdf_jobs, _job_pdf, workers, out_f, kept, n, t0, lang_counts)
        _run_group("overflow_pdf", overflow_pdf_jobs, _job_pdf, workers, out_f, kept, n, t0, lang_counts)
        _run_group("overflow_img", overflow_img_jobs, _job_image, workers, out_f, kept, n, t0, lang_counts)
        _run_group("pdf_png_en", pdf_png_jobs, _job_image, workers, out_f, kept, n, t0, lang_counts)
        _run_group("xfund_funsd", xfund_funsd_jobs, _job_image, workers, out_f, kept, n, t0, lang_counts)

        arxiv_quota_target = min(n, kept[0] + round(n * SCIENTIFIC_PAPER_QUOTA_FRAC))
        _run_group("arxiv_en", arxiv_jobs, _job_pdf, workers, out_f, kept, arxiv_quota_target, t0, lang_counts)

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
