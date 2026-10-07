"""Listing of the source files for the manifest: one JSON line per file with what the target selection needs, so no file
is opened again for it. Fields: path, language, topic, category (the folder), digital (PDF with text set in it, restyled),
pages, fonts (base fonts of the middle page), body_pt (its most common font size).

python -m dspages.prep.files [--workers 12]   -> paths.files
"""
import argparse
import glob
import json
import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import pymupdf as fitz

from ..config import Paths

LAYOUTS = Paths().layouts
DIGITAL = ("books_pdf", "bureaucracy", "corpus_add/pdf", "exams", "other_docs", "scientific_paper")  # restyled
RASTER = ("certificates", "corpus_add/Images")                                                       # as images
IMAGES = (".png", ".jpg", ".jpeg")


def name_parts(path):
    """(topic, language) of a file named <hash>_<topic>_<code>[_p<page>].<ext>; unknown for any other name."""
    parts = re.sub(r"_p\d+$", "", os.path.splitext(os.path.basename(path))[0]).split("_")
    ok = len(parts) >= 3 and re.fullmatch(r"[a-z]{2,3}", parts[-1])
    return ("_".join(parts[1:-1]), parts[-1]) if ok else ("unknown", "unknown")


def describe(args):
    """The listing row of a file; a PDF is opened for its pages, and fonts and body size of its middle page."""
    path, category, digital = args
    topic, language = name_parts(path)
    row = dict(path=path, language=language, topic=topic, category=category, digital=False, pages=1, fonts=[], body_pt=0.0)
    if not path.lower().endswith(".pdf"):
        return row
    with fitz.open(path) as doc:
        mid = doc[doc.page_count // 2]
        sizes = Counter()
        for s in (s for b in mid.get_text("dict", flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES)["blocks"] for l in b.get("lines", []) for s in l["spans"]):
            sizes[round(s["size"] * 2) / 2] += len(s["text"])
        return dict(row, digital=digital and bool(sizes), pages=doc.page_count,
                    fonts=sorted({f[3] for f in mid.get_fonts()}), body_pt=sizes.most_common(1)[0][0] if sizes else 0.0)


def main(workers):
    files = [(p, sub.split("/")[0], digital) for subs, digital in ((DIGITAL, True), (RASTER, False)) for sub in subs
             for p in sorted(glob.glob(os.path.join(LAYOUTS, sub, "**", "*"), recursive=True))
             if p.lower().endswith((".pdf",) + IMAGES)]
    with ProcessPoolExecutor(workers) as pool, open(Paths().files, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in pool.map(describe, files, chunksize=64))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    main(ap.parse_args().workers)
