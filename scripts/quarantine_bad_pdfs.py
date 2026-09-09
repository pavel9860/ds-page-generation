import glob
import os
import shutil
import sys

import fitz

sys.path.insert(0, os.path.dirname(__file__))
from page_source import render_pdf_page

L = "/run/media/me/D/ML_DS/UVTM/Layouts"
ARCHIVE_DIR = os.path.join(L, "archive", "quarantined_pdfs")

SOURCES = [
    "scientific_paper/arxiv_pdfs/*.pdf",
    "Forms/**/*.pdf",
    "dictionares/*.pdf",
    "Pdf/**/*.pdf",
    "books/*/*.pdf",
]


def pdf_should_convert(doc: fitz.Document) -> bool:
    """True if no page has any text object at all — a scanned/image-only PDF."""
    for p in range(doc.page_count):
        if doc[p].get_text("text").strip():
            return False
    return True


def convert_to_images(path: str) -> int:
    doc = fitz.open(path)
    n = doc.page_count
    doc.close()
    out_dir = os.path.dirname(path)
    stem = os.path.splitext(os.path.basename(path))[0]
    saved = 0
    for p in range(n):
        try:
            img = render_pdf_page(path, p)
        except Exception as e:
            print(f"    [skip page {p}] {path}: {e}")
            continue
        img.save(os.path.join(out_dir, f"{stem}_p{p}.png"))
        saved += 1
    return saved


def main() -> None:
    n_checked = n_bad = n_unopenable = n_pages = 0
    for pattern in SOURCES:
        for f in sorted(glob.glob(f"{L}/{pattern}", recursive=True)):
            rel = os.path.relpath(f, L)
            n_checked += 1
            try:
                doc = fitz.open(f)
            except Exception as e:
                print(f"[unopenable] {rel}: {e}")
                n_unopenable += 1
                archive_path = os.path.join(ARCHIVE_DIR, rel)
                os.makedirs(os.path.dirname(archive_path), exist_ok=True)
                shutil.move(f, archive_path)
                continue
            try:
                bad = pdf_should_convert(doc)
            except Exception:
                bad = True
            doc.close()
            if not bad:
                continue

            n_bad += 1
            print(f"[image-only, converting] {rel}")
            n_pages += convert_to_images(f)

            archive_path = os.path.join(ARCHIVE_DIR, rel)
            os.makedirs(os.path.dirname(archive_path), exist_ok=True)
            shutil.move(f, archive_path)

    print(f"checked {n_checked} pdfs, unopenable {n_unopenable}, converted {n_bad} "
          f"({n_pages} pages to images)")


if __name__ == "__main__":
    main()
