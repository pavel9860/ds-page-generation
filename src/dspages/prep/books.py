"""Book PDFs -> cleaned text, run once, independent of the manifest: every <layouts>/books/<topic>/*.pdf becomes
<paths.books>/<topic>/<name>.txt, and <paths.books>/index.jsonl lists each book once:
kind "text" (txt, language, chars) for books with >= MIN_BOOK_CHARS of text and >= SCAN_CHARS_PER_PAGE per page,
kind "scan" (pdf, language, pages) for books without a usable text layer (their pages are used as images);
books with too little content are left out.

python -m dspages.prep.books
"""
import glob
import json
import os
import re
import sys
import time

import pymupdf as fitz

from ..config import Paths
from ..config import LayoutCfg
from ..layout.text import book_text
from .text_extract import book_language, extract_pdf_page_text, is_degenerate_repetition, join_pages, script_language

BOOKS = os.path.join(Paths().layouts, "books")
OUT_ROOT = Paths().books
SKIP_TOPICS = {os.path.basename(OUT_ROOT)}
UNCATEGORIZED = "uncategorized"
MIN_BOOK_CHARS = 30000
SCAN_CHARS_PER_PAGE = 300
SCAN_LANGUAGES = {"Bakht Novel": "ur", "Dana Pani": "ur", "Yaaram novel": "ur", "Kandukondaen": "ta",
                  "Yoga Vasistam": "ta", "Nagaon_Ka_Rahasya": "hi", "kupdf.net_1-1-": "el", "Kusadikika": "sw"}


def _scan_language(name):
    """SCAN_LANGUAGES (checked by eye), else the name's code (name.<code>.pdf), else a non-Latin script in the
    name, else "unknown"."""
    code = re.search(r"\.([a-z]{2})(?:_\w+)?\.pdf$", name)
    other = script_language("".join(c for c in name if not c.isascii()))
    return next((v for k, v in SCAN_LANGUAGES.items() if name.startswith(k)), None) or (
        code.group(1) if code else other if other not in ("latin", "unknown") else "unknown")


def _index_entry(pdf_path, txt_path, n_pages):
    chars = len(book_text(txt_path)) if os.path.exists(txt_path) else 0
    if chars >= max(MIN_BOOK_CHARS, SCAN_CHARS_PER_PAGE * n_pages):
        return dict(kind="text", txt=txt_path, language=book_language(txt_path, LayoutCfg().scripts), chars=chars)
    if chars < SCAN_CHARS_PER_PAGE * n_pages:
        return dict(kind="scan", pdf=pdf_path, language=_scan_language(os.path.basename(pdf_path)), pages=n_pages)
    return None


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
    index = []
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
            if os.path.exists(out_path) and os.path.getmtime(out_path) >= os.path.getmtime(pdf_path):
                with fitz.open(pdf_path) as d:
                    e = _index_entry(pdf_path, out_path, d.page_count)
                if e:
                    index.append(e)
                continue
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
            e = _index_entry(pdf_path, out_path, len(pages))
            if e:
                index.append(e)
            n_books += 1
            n_pages_total += len(pages)
            print(f"  {topic}/{base}: {len(pages)} pages in {time.time() - t0:.1f}s")

    with open(os.path.join(OUT_ROOT, "index.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(json.dumps(e, ensure_ascii=False) + "\n" for e in index)
    print(f"index: {sum(e['kind'] == 'text' for e in index)} text books, {sum(e['kind'] == 'scan' for e in index)} "
          f"scans")
    print(f"\ndone. {n_books} books, {n_pages_total} pages extracted to {OUT_ROOT}/<topic>/ "
          f"({n_failed} failed).")


if __name__ == "__main__":
    main()
