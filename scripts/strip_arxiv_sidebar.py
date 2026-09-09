"""Batch-rewrite scientific_paper/arxiv_pdfs IN PLACE: delete arXiv's injected vertical sidebar
identifier block (not just paint over it — apply_redactions removes the underlying text/vector
content, then fills the gap white) on every page. Each PDF is rewritten to a temp file first and
only then moved over the original, so a crash mid-file never leaves a half-written PDF in place.
A page that PyMuPDF itself can't parse (malformed content stream) is left untouched and counted
as a failure rather than aborting the whole run.

Usage: python strip_arxiv_sidebar.py [--dry-run]
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

import fitz  # PyMuPDF

from text_bounds import arxiv_sidebar_bbox

LAYOUTS = Path("/run/media/me/D/ML_DS/UVTM/Layouts")
SRC_DIR = LAYOUTS / "scientific_paper" / "arxiv_pdfs"


def strip_pdf_inplace(path: Path) -> int:
    """Returns the number of pages a sidebar block was removed from. Writes to a temp file in
    the same directory, then atomically replaces the original — the original is untouched if
    anything raises before the replace."""
    doc = fitz.open(path)
    n_stripped = 0
    for page in doc:
        bbox = arxiv_sidebar_bbox(page)
        if bbox is None:
            continue
        page.add_redact_annot(fitz.Rect(*bbox), fill=(1, 1, 1))
        page.apply_redactions()
        n_stripped += 1

    if n_stripped == 0:
        doc.close()
        return 0

    fd, tmp_name = tempfile.mkstemp(suffix=".pdf", dir=path.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        doc.save(tmp_path, garbage=4, deflate=True)
        doc.close()
        os.replace(tmp_path, path)   # atomic on the same filesystem
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return n_stripped


def count_sidebar_pages(path: Path) -> int:
    doc = fitz.open(path)
    n = 0
    for page in doc:
        try:
            if arxiv_sidebar_bbox(page) is not None:
                n += 1
        except Exception:
            pass
    doc.close()
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="scan and report only, write nothing")
    args = ap.parse_args()

    pdfs = sorted(SRC_DIR.glob("*.pdf"))
    print(f"{len(pdfs)} PDFs in {SRC_DIR} (overwriting in place)" if not args.dry_run
          else f"{len(pdfs)} PDFs in {SRC_DIR} (dry run)")

    total_pages_stripped = 0
    n_failed = 0
    for i, path in enumerate(pdfs, 1):
        try:
            n = count_sidebar_pages(path) if args.dry_run else strip_pdf_inplace(path)
        except Exception as e:
            n_failed += 1
            print(f"  [{i}/{len(pdfs)}] FAILED {path.name}: {e}", file=sys.stderr)
            continue
        total_pages_stripped += n
        if i % 50 == 0 or i == len(pdfs):
            print(f"  [{i}/{len(pdfs)}] ... {total_pages_stripped} pages stripped so far, "
                  f"{n_failed} files failed")

    print(f"done. {total_pages_stripped} pages had a sidebar removed across {len(pdfs)} PDFs "
          f"({n_failed} files failed).")


if __name__ == "__main__":
    main()
