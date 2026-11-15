"""Handwritten A4 pages of rare scripts from real pages: a page of a real prose document is read by OCR with its structure (headings,
paragraphs, list items, blank lines, columns), checked for the script and the OCR confidence, and written by hand with a font that has every
character. Nothing is dropped from the text; a page is not used when its text is not good enough. The original page is rendered beside it.

python -m dspages.hand rare --out DIR [--langs hi bn ...] [--per-lang 1] [--seed 1] [--tries 300]
A language with no document at all gets Google-translated English pages. Source pages: farthest-first in the 16-d head space (dspages.hand.spread), one per document. Every page has its own source page, writer, pen, paper (all kinds), sheet format (A4 / A5 / B5) and density.
"""

import argparse
import collections
import random
from pathlib import Path

import cv2
import pymupdf

from ..fillpic.paths import doc_hash
from ..layout import legacy
from . import corpus, page, spread
from . import labelled as FL
from . import texts as TX
from .config import LOST, PER_DOC  # noqa: F401
from .langs import EXPECT, NO_TEXT_LAYER, script_ok
from .paper import FORMATS, KINDS
from .structure import to_items

DONOR = "en"  # a language without documents is translated from English pages
LANGS = "hi bn ta te mr gu kn ml pa ne si km my lo or as ur am sa th".split()
PROSE = {"plain", "formatted", "scientific", "magazines", "poems", "bureaucracy"}


def prose_docs():
    """Documents whose page type (existing embedding labels, p >= 0.6) is a text page: not forms, tables, flyers or maps."""
    return {d for d, (lab, p) in corpus.page_types(0.6).items() if lab in PROSE}


def docs_of(lang, ok):
    db = corpus.connect()
    lg = {d for d, l in corpus.languages(db, 0.9).items() if l == lang}
    return [(path, n) for d, path, n in corpus.pdf_documents(db) if d in lg and d in ok]


def upright(pdf, no):
    """The page, or, when most of its text runs at 90 degrees (a landscape table set on a portrait page), a copy with the content turned back upright
    (the reader takes such lines for pictures)."""
    pg = pdf[no]
    dirs = [l["dir"] for b in pg.get_text("dict")["blocks"] for l in b.get("lines", ())]
    turn = [d for d in dirs if abs(d[1]) > 0.9]
    if len(turn) <= 0.5 * max(len(dirs), 1):
        return pg
    up = pymupdf.open()
    r = pg.rect
    new = up.new_page(width=r.height, height=r.width)
    new.show_pdf_page(new.rect, pdf, no, rotate=-90 if sum(d[1] for d in turn) < 0 else 90)
    return new


def _valid(x):
    if isinstance(x, str):
        return x.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
    if isinstance(x, dict):
        return {k: _valid(v) for k, v in x.items()}
    return type(x)(map(_valid, x)) if isinstance(x, (list, tuple)) else x


def page_blocks(path, no, lg):
    """(blocks, how): the digital text layer of the page read by the restyling reader (texts.blocks_of: text repaired from the font programs, Indic
    and RTL order, tables, columns). The documents are digital-born: no OCR, and no page is skipped for its text: how says what the check found
    (text layer; a garbled layer is the translation's: translated()). Only a page without any text, or a language without a usable layer (NO_TEXT_LAYER), gives None."""
    if lg in NO_TEXT_LAYER:
        return None, "no usable text layer"

    def read(lang):
        with (
            pymupdf.open(path) as pdf,
            legacy.language(lang),
        ):  # legacy-font runs are read by glyph shape (layout.legacy)
            return _valid(
                TX.blocks_of(upright(pdf, no))
            )  # a lone surrogate (a wrong decode), in a text or a table cell: U+FFFD

    bl = read(lg)
    if bl and (
        EXPECT.get(lg, "Latin") == "Latin" or not script_ok(" ".join(b[1] for b in bl), lg, 0.5)
    ):  # a wrong label (legacy codes read as Latin
        with pymupdf.open(path) as pdf:  # letters, or another script): read in the script its fonts draw
            drawn = legacy.drawn_scripts(upright(pdf, no))
        for sc in drawn:
            alt = next(l for l, s in legacy.LANG_SCRIPT.items() if s == sc)
            other = read(alt)
            if (
                other
                and script_ok(" ".join(b[1] for b in other), alt, 0.5)
                and legacy.glot(" ".join(b[1] for b in other), sc) > legacy.glot(" ".join(b[1] for b in bl))
            ):  # it reads as a language more surely
                bl = other
                break
    if not bl:
        return None, "no text"
    return bl, "text layer"  # garbled: its translation loses more than LOST of the tags


def list_page(blocks):
    """A list-like page: most of the text in list items (bullets, numbers), at least 5 of them."""
    li = [b for b in blocks if b[0] == "li"]
    chars = sum(len(b[1]) for b in blocks)
    return len(li) >= 5 and 350 <= chars <= 3500 and sum(len(b[1]) for b in li) >= 0.5 * chars


def translated(blocks, lg, sl=DONOR):
    """The blocks with their texts translated by Google Translate (dspages.multilang.gtranslate), structure and boxes kept; None when the text is garbled (more than LOST of the blocks lost)."""
    from dspages.multilang import gtranslate as GT

    rows = [
        b[4].get("rows") if len(b) > 4 and isinstance(b[4], dict) else None for b in blocks
    ]  # the cells of a table block are its text
    src = [b[1] for b in blocks] + [c for r in rows if r for row in r for c in row]
    tr = GT.lines(src, lg, sl)
    if len(tr) != len(src) or sum(t is None for t in tr) > LOST * len(
        tr
    ):  # garbled: Google dropped the tags of more than LOST of the blocks
        return None
    tr = [t or s for s, t in zip(src, tr)]  # a line Google gives nothing for stays as it is
    cells = iter(tr[len(blocks) :])
    out = []
    for b, t, r in zip(blocks, tr, rows):
        if r:
            b = b[:4] + ({**b[4], "rows": [[next(cells) for _ in row] for row in r]},) + tuple(b[5:])
        out.append((b[0], t) + tuple(b[2:]))
    return out


def plan_of(lg, k, rng, fix=None):
    """The page plan write() writes with: a random writer, pen, paper, sheet; the factors in fix chosen by the caller."""
    plan = dict(
        i=k,
        seed=700 + 13 * k,
        wseed=300 + 5 * k,
        persona=rng.choice(FL.PERSONS),
        style=rng.choice(["casual", "accurate", "educated", "professional"]),
        pen=rng.choice(FL.PENS),
        paper=rng.choice(KINDS),
        fmt=rng.choice(list(FORMATS)),
        dense=rng.random() < 0.3,
        xh=rng.uniform(2.2, 2.8),
        xh_ratio=rng.uniform(0.3, 0.4),
        fill=0.97,
        dy=0.2,
        strict=False,
        prefer_font={"Arabic": "NotoNastaliqUrdu[wght].ttf"} if lg == "ur" else {},
    )
    plan.update(fix or {})  # writer factors chosen by the caller (a balanced design)
    return plan


def write(blocks, lg, k, rng, fix=None):
    """(img, meta, plan, items, None) of the blocks written by hand (a random writer, pen, paper, sheet), the same page again with smaller writing while the
    text is cut short (then returned with that reason); characters no hand font has are left out (meta uncovered_chars counts them); (None, None, None, None, reason) when it cannot be written."""
    items, plan = to_items(blocks), plan_of(lg, k, rng, fix)
    for _ in range(4):
        img, meta = page.make(plan, items)
        if meta["lines"] >= 0.8 * meta["lines_want"]:
            break
        plan = dict(plan, xh=plan["xh"] * 0.85)
    if meta["lines"] < 0.8 * meta["lines_want"]:
        return (
            img,
            meta,
            plan,
            items,
            "text cut short",
        )  # the page as written: a caller that has no more text may keep it
    return img, meta, plan, items, None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--langs", nargs="*", default=LANGS)
    ap.add_argument("--per-lang", type=int, default=1)
    ap.add_argument(
        "--translate", action="store_true", help="skip the language's own documents: Google-translated English pages"
    )
    ap.add_argument(
        "--kind", choices=["prose", "list"], default="prose", help="prose pages or list-like pages (items of lists)"
    )
    ap.add_argument("--start", type=int, default=0, help="number of the first page (to add pages to a directory)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tries", type=int, default=300)
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ok = prose_docs()
    rows = []
    k = a.start
    donor_used = set()
    for lg in a.langs:
        got = 0
        drops = collections.Counter()
        for src in (
            (DONOR,) if a.translate else (lg, DONOR)
        ):  # no usable page of the language at all: Google Translate of English pages
            if got:
                break
            docs = docs_of(src, ok)
            if src != lg:
                docs = [d for d in docs if d[0] not in donor_used]
            by = {doc_hash(p): (p, n) for p, n in docs}
            n_doc = {}  # at most PER_DOC pages tried per document, in farthest-first rank order
            order = [
                (by[h][0], pg)
                for h, pg in spread.ranked(set(by), 5 * a.tries)
                if n_doc.__setitem__(h, n_doc.get(h, 0) + 1) or n_doc[h] <= PER_DOC
            ][: a.tries]  # pages spread over the 16-d head space
            if not order:  # documents outside the space: random pages
                rng.shuffle(docs)
                order = [(p, rng.randrange(n)) for p, n in docs[: a.tries]]
            used = set()
            for path, no in order:
                if got >= a.per_lang:
                    break
                if path in used:
                    continue
                try:
                    blocks, how = page_blocks(path, no, src)
                except Exception:
                    blocks, how = None, "unreadable"
                if not blocks:
                    drops[how] += 1
                    continue
                if a.kind == "list" and not list_page(blocks):
                    drops["not a list"] += 1
                    continue
                if src != lg:
                    blocks = translated(blocks, lg)
                    if not blocks:
                        drops["no translation"] += 1
                        continue
                img, meta, plan, items, bad = write(blocks, lg, k, rng)
                if bad:
                    drops[bad] += 1
                    continue
                cv2.imwrite(str(out / f"{k:02d}_{lg}_hw.jpg"), img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
                pymupdf.open(path)[no].get_pixmap(dpi=90).save(str(out / f"{k:02d}_{lg}_orig.png"))
                rows.append(FL.row(k, lg, path, no, items, plan, meta))
                print(
                    lg,
                    "ok",
                    how if src == lg else "translated " + how,
                    path.name,
                    no,
                    len(blocks),
                    "blocks",
                    plan["paper"],
                    plan["fmt"],
                    meta["lines"],
                    "lines",
                    flush=True,
                )
                used.add(path)
                donor_used.add(path)
                got += 1
                k += 1
        print(lg, "pages", got, "dropped", dict(drops), flush=True)
        if got < a.per_lang:
            print(lg, f"only {got} of {a.per_lang} pages found")
    FL.sheet(out, rows, f"rare_pages{a.start or ''}.html")


if __name__ == "__main__":
    main()
