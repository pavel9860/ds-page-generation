"""What the source folders hold against a plan's quotas: pages per script group and language, what to download.

python tools/inventory.py [--preset full] [--n 100000] [--workers 12]
Counts without rendering: PDF pages (PAGES_PER_PDF_CAP per document, all of rare-script ones), one page per image,
distinct book pages of book_page_chars characters. Upper bounds: pages that later fail the content and size checks
are included. Groups of use_all take all their raster pages; the others share the rest of n by script_mix, raster
pages first, then book snippets (repeating text beyond the distinct book pages, in other formatting).
Lists the configured languages without book text.
"""
import argparse
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

from dspages.conditions import script
from dspages.config import get_preset
from dspages.prep import manifest as M
from dspages.layout.text import book_text
from dspages.prep.text_extract import book_language


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
    return book_language(path, M.LayoutCfg().scripts), len(book_text(path)) // page_chars


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
    kinds = (("pdf", "image"), ("book",))
    tot = {g: [sum(c[k] for (gg, _), c in pages.items() if gg == g for k in ks) for ks in kinds]
           for g in {g for g, _ in pages} | {g for g, _ in P.layout.script_mix}}
    whole = {g: tot[g][0] for g in P.layout.use_all if tot.get(g, [0])[0]}
    rest = max(0, a.n - sum(whole.values()))
    wsum = sum(w for g, w in P.layout.script_mix if g not in whole)
    print(f"{'group':10s} {'quota':>8s} {'raster':>8s} {'book pg':>8s} {'repeats':>8s} {'short':>8s}")
    for g, w in P.layout.script_mix:
        r, b = tot[g]
        q = whole[g] if g in whole else round(rest * w / wsum)
        need = max(0, q - r)
        print(f"{g:10s} {q:8d} {r:8d} {b:8d} {need / b if b else 0:8.1f} {need if not b else 0:8d}")
    print("\npages per language")
    for (g, lang), c in sorted(pages.items(), key=lambda kv: (kv[0][0], -sum(kv[1].values()))):
        print(f"  {g:9s} {str(lang):8s} " + "  ".join(f"{k}={v}" for k, v in sorted(c.items())))
    have = {lang for (_, lang), c in pages.items() if c["book"]}
    print("\nconfigured languages without book text:",
          {g: [x for x in ls if x not in have and x and x not in ("unknown", "cyr")]
           for g, ls in P.layout.scripts.items() if g not in P.layout.use_all})
    print("book texts with no text (scans, need OCR):",
          [os.path.basename(p) for p in M.glob.glob(os.path.join(M.BOOKS, "**", "*.txt"), recursive=True)
           if p not in set(M.book_txt_paths())])


if __name__ == "__main__":
    main()
