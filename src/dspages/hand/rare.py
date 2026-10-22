"""Handwritten A4 pages of rare scripts from real pages: a page of a real prose document is read by OCR with its structure (headings,
paragraphs, list items, blank lines, columns), checked for the script and the OCR confidence, and written by hand with a font that has every
character. Nothing is dropped from the text; a page is not used when its text is not good enough. The original page is rendered beside it.

python -m dspages.hand rare --out DIR [--langs hi bn ...] [--seed 1] [--tries 300]
"""
import argparse
import random
from pathlib import Path

import cv2
import pymupdf

from . import corpus, page
from . import labelled as FL
from . import ocr as OL
from .structure import to_items

LANGS = "hi bn ta te mr gu kn ml pa ne si km my lo or as ur am sa th".split()
PROSE = {"plain", "formatted", "scientific", "magazines", "poems", "bureaucracy"}


def prose_docs():
    """Documents whose page type (existing embedding labels, p >= 0.6) is a text page: not forms, tables, flyers or maps."""
    return {d for d, (lab, p) in corpus.page_types(0.6).items() if lab in PROSE}


def docs_of(lang, ok):
    db = corpus.connect()
    lg = {d for d, l in corpus.languages(db, 0.9).items() if l == lang}
    return [(path, n) for d, path, n in corpus.pdf_documents(db) if d in lg and d in ok]


def good_page(blocks):
    """A prose page: enough text, mostly in long paragraphs (not a form, a schedule or a flyer of fragments)."""
    chars = sum(len(b[1]) for b in blocks)
    long_ = [b for b in blocks if b[0] == "p" and len(b[1]) >= 120]
    return len(blocks) >= 6 and 700 <= chars <= 3500 and len(long_) >= 3 and sum(len(b[1]) for b in long_) >= 0.55 * chars


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--langs", nargs="*", default=LANGS)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tries", type=int, default=300)
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ok = prose_docs()
    rows = []
    for k, lg in enumerate(a.langs):
        docs = docs_of(lg, ok)
        rng.shuffle(docs)
        for path, n in docs[:a.tries]:
            no = rng.randrange(n)
            try:
                blocks = OL.blocks(path, no, lg)
            except Exception:
                continue
            if not blocks or not good_page(blocks):
                continue
            items = to_items(blocks)
            plan = dict(i=k, seed=700 + 13 * k, wseed=300 + 5 * k, persona=rng.choice(FL.PERSONS), style=rng.choice(["casual", "accurate", "educated", "professional"]),
                        pen=rng.choice(FL.PENS), paper=rng.choice(FL.PAPERS), fmt="a4", dense=False, xh=rng.uniform(2.2, 2.8), xh_ratio=rng.uniform(0.3, 0.4),
                        fill=0.97, dy=0.2, strict=True, prefer_font={"Arabic": "NotoNastaliqUrdu[wght].ttf"} if lg == "ur" else {})
            try:
                img, meta = page.make(plan, items)
            except Exception as e:
                print(lg, "write failed", e)
                continue
            if meta["uncovered_chars"] or meta["lines"] < 0.8 * meta["lines_want"]:
                continue                                     # no font for all characters, or the text is cut short
            cv2.imwrite(str(out / f"{k:02d}_{lg}_hw.jpg"), img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
            pymupdf.open(path)[no].get_pixmap(dpi=90).save(str(out / f"{k:02d}_{lg}_orig.png"))
            rows.append(FL.row(k, lg, path, no, items, plan, meta))
            print(lg, "ok", path.name, no, len(blocks), "blocks", plan["paper"], meta["lines"], "lines", flush=True)
            break
        else:
            print(lg, "NO PAGE FOUND")
    FL.sheet(out, rows, "rare_pages.html")


if __name__ == "__main__":
    main()
