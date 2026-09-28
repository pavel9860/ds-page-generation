"""Book PDFs -> cleaned text: every <layouts>/books/<topic>/*.pdf becomes <paths.books>/<topic>/<name>.txt.

python -m dspages.prep.books
"""
import glob
import os
import sys
import time

import pymupdf as fitz

from ..config import Paths
from .text_extract import extract_pdf_page_text, is_degenerate_repetition, join_pages

BOOKS = os.path.join(Paths().layouts, "books")
OUT_ROOT = Paths().books
SKIP_TOPICS = {os.path.basename(OUT_ROOT)}
UNCATEGORIZED = "uncategorized"
NUSX = str.maketrans("abgdevzTiklmnopJrstufqRySCcZwWxjh", "აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ")


def _georgian(name, text):
    """Georgian text typed in a legacy Nusx font (Latin code points) back to Unicode, for Georgian-named files."""
    if not any("\u10d0" <= c <= "\u10ff" for c in name):
        return text
    letters = [c for c in text if c.isalpha()]
    return text.translate(NUSX) if sum("\u10d0" <= c <= "\u10ff" for c in letters) < 0.05 * len(letters) else text


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
                fh.write(_georgian(stem, join_pages(pages)))
            n_books += 1
            n_pages_total += len(pages)
            print(f"  {topic}/{base}: {len(pages)} pages in {time.time() - t0:.1f}s")

    print(f"\ndone. {n_books} books, {n_pages_total} pages extracted to {OUT_ROOT}/<topic>/ "
          f"({n_failed} failed).")


if __name__ == "__main__":
    main()
