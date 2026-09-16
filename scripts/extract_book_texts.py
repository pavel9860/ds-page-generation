"""Extracts and cleans text from every books/<topic>/*.pdf (the topic subdirs from the LLM
classification pass — see docs/manifest_notes.md, "Round 3") and persists it as one .txt file
per book, mirrored into the same topic subdirs under books/Texts/.

Why persist at all: build_manifest.py previously re-ran extract_pdf_page_text() on every PDF
page on every single build (by design, to keep the CSV small and rendering reproducible from
just file+page+seed) — correct for final rendering, but wasteful for the build step itself,
which re-does the same PDF parsing/cleaning from scratch each time just to measure page-text
length and font coverage. Caching it here lets build_manifest.py read a flat .txt instead.

One .txt per book, pages joined with text_extract.PAGE_BREAK so per-page structure round-trips.
"""
import glob
import os
import sys
import time

import fitz  # PyMuPDF

sys.path.insert(0, os.path.dirname(__file__))
from text_extract import extract_pdf_page_text, is_degenerate_repetition, join_pages

L = "/run/media/me/D/ML_DS/UVTM/Layouts"
BOOKS = f"{L}/books"
OUT_ROOT = f"{BOOKS}/Texts"
SKIP_TOPICS = {"Texts"}   # the pre-existing parallel-translation corpus itself, not a book topic


UNCATEGORIZED = "uncategorized"   # topic for *.pdf placed directly under books/, no subfolder


def main() -> None:
    topics = sorted(d for d in os.listdir(BOOKS)
                     if os.path.isdir(os.path.join(BOOKS, d)) and d not in SKIP_TOPICS)
    print(f"topics: {topics}")

    n_books = n_pages_total = n_failed = 0
    for topic in topics + [UNCATEGORIZED]:
        pdfs = (sorted(glob.glob(f"{BOOKS}/*.pdf")) if topic == UNCATEGORIZED
               else sorted(glob.glob(f"{BOOKS}/{topic}/*.pdf")))
        if not pdfs:
            continue
        out_dir = os.path.join(OUT_ROOT, topic)
        os.makedirs(out_dir, exist_ok=True)
        for pdf_path in pdfs:
            base = os.path.basename(pdf_path)
            stem = os.path.splitext(base)[0]
            out_path = os.path.join(out_dir, f"{stem}.txt")
            t0 = time.time()
            try:
                doc = fitz.open(pdf_path)
                pages = [extract_pdf_page_text(doc[p]) for p in range(doc.page_count)]
                doc.close()
            except Exception as e:
                print(f"  [FAILED] {topic}/{base}: {e}", file=sys.stderr)
                n_failed += 1
                continue
            if is_degenerate_repetition(join_pages(pages)):
                print(f"  [DEGENERATE] {topic}/{base}: repeated content, dropped", file=sys.stderr)
                n_failed += 1
                if os.path.exists(out_path):
                    os.remove(out_path)
                continue
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write(join_pages(pages))
            n_books += 1
            n_pages_total += len(pages)
            print(f"  {topic}/{base}: {len(pages)} pages in {time.time()-t0:.1f}s")

    print(f"\ndone. {n_books} books, {n_pages_total} pages extracted to {OUT_ROOT}/<topic>/ "
          f"({n_failed} failed).")


if __name__ == "__main__":
    main()
