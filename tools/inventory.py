"""What the source folders hold against a plan's quotas: pages per script group and language, what to download.

python tools/inventory.py [--preset full] [--n 100000] [--workers 12]
Counts without rendering: PDF pages (PAGES_PER_PDF_CAP per document, all of rare-script ones), one page per image,
book pages of book_page_chars. Upper bounds: pages that later fail the content and size checks are included.
Quota of a group = n * its weight; raster pages cover it first, book pages the rest.
"""
import argparse
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

from dspages.conditions import script
from dspages.config import get_preset
from dspages.prep import manifest as M
from dspages.prep.text_extract import classify_language


def _pdf_pages(args):
    path, lang = args
    import pymupdf
    try:
        with pymupdf.open(path) as d:
            n = d.page_count
    except Exception:
        return lang, 0
    rare = script(dict(language=lang), M.LayoutCfg().scripts) in M.RARE_SCRIPTS
    return lang, n if rare else min(n, M.PAGES_PER_PDF_CAP)


def _book_pages(args):
    path, page_chars = args
    return classify_language(path), os.path.getsize(path) // page_chars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="full")
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    P = get_preset(a.preset)
    sc = P.layout.scripts
    meta = M.load_metadata(os.path.join(M.LAYOUTS, "corpus", "metadata.jsonl"))
    over_pdf, over_img = M.corpus_overflow_jobs()
    pdfs = [(j[0], j[2]) for j in M.corpus_pdf_jobs(meta) + over_pdf + M.arxiv_jobs()]
    images = [(j[0], j[1]) for j in over_img + M.pdf_png_jobs() + M.xfund_funsd_jobs()]
    books = [(p, P.layout.book_page_chars) for p in M.book_txt_paths()]

    pages = defaultdict(Counter)                   # (group, language) -> {pdf, image, book}
    with ProcessPoolExecutor(a.workers) as pool:
        for lang, n in pool.map(_pdf_pages, pdfs, chunksize=16):
            pages[(script(dict(language=lang), sc), lang)]["pdf"] += n
        for lang, n in pool.map(_book_pages, books, chunksize=16):
            pages[(script(dict(language=lang), sc), lang)]["book"] += n
    for _, lang in images:
        pages[(script(dict(language=lang), sc), lang)]["image"] += 1

    print(f"sources: {len(pdfs)} PDFs, {len(images)} images, {len(books)} books\n")
    print(f"{'group':10s} {'quota':>8s} {'raster':>8s} {'books':>8s} {'covered':>8s} {'short':>8s}  raster share")
    need = {}
    for g, w in P.layout.script_mix:
        raster = sum(c["pdf"] + c["image"] for (gg, _), c in pages.items() if gg == g)
        book = sum(c["book"] for (gg, _), c in pages.items() if gg == g)
        q = round(a.n * w)
        cov = min(q, raster + book)
        need[g] = q - cov
        print(f"{g:10s} {q:8d} {raster:8d} {book:8d} {cov:8d} {q - cov:8d}  {min(raster, q) / q:.0%}")
    print("\npages per language")
    for (g, lang), c in sorted(pages.items(), key=lambda kv: (kv[0][0], -sum(kv[1].values()))):
        print(f"  {g:9s} {str(lang):8s} " + "  ".join(f"{k}={v}" for k, v in sorted(c.items())))
    print("\nto download (pages, raster preferred):")
    for g, v in need.items():
        if v > 0:
            print(f"  {g}: {v} more pages")
    if not any(v > 0 for v in need.values()):
        print("  nothing: every quota is covered")


if __name__ == "__main__":
    main()
