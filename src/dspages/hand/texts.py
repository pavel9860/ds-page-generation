"""Text passages for handwritten pages, cut from digital-born PDFs (local, parallel).

Documents: digital text layer (plan.cls == text), language confident, page type from the existing embedding
classification (emb_out/doc_types.jsonl) reliable and prose-like (not forms, tables, flyers, maps, blueprints, music),
or the folder is books / scientific papers. Passages are prose blocks of one page, with the script of the language.

python -m dspages.hand texts --out texts.jsonl [--per-lang 400] [--workers 11]
"""

import argparse
import json
import os
import random
import re
import unicodedata
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np
import pymupdf as fitz
import regex
from hwsynth.scripts import script

from ..ingest.lang import garbled
from ..layout import legacy
from ..layout import refont as R
from . import corpus, pagekind
from .config import (
    PATHS,
    SYMBOL_BULLET,  # noqa: F401
)
from .structure import to_items
from .util import walk_text

LAYOUTS = PATHS.layouts
PROSE = {"plain", "formatted", "scientific", "magazines", "bureaucracy", "poems", "recieps"}
FOLDER_KIND = {
    "books_pdf": "book",
    "scientific_paper": "paper",
    "text": "article",
    "bureaucracy": "letter",
    "exams": "exam",
    "magazines": "magazine",
}
LABEL_KIND = {
    "plain": "article",
    "formatted": "article",
    "scientific": "paper",
    "magazines": "magazine",
    "bureaucracy": "letter",
    "poems": "poem",
    "recieps": "recipe",
}
SCRIPT = {"en": "Latin"}


def candidates(min_p=0.7, min_conf=0.95, min_chars=1500):
    """Documents with a digital text layer, a confident language and prose pages (page type from the embeddings, or a book / paper folder)."""
    db = corpus.connect()
    dt = corpus.page_types()
    lang = corpus.languages(db, min_conf, min_chars)
    out = []
    for doc, typ, cls, path, pages in db.execute("select doc,type,cls,new_path,pages from plan"):
        if cls != "text" or doc not in lang or not path.lower().endswith(".pdf"):
            continue
        lg = lang[doc]
        lab, p = dt.get(doc, (None, 0))
        if typ in ("books_pdf", "scientific_paper"):
            kind = FOLDER_KIND[typ]
        elif lab in PROSE and p >= min_p:
            kind = LABEL_KIND[lab]
        else:
            continue
        out.append(dict(doc=doc, lang=lg, kind=kind, path=f"{LAYOUTS}/{path}", pages=pages or 1))
    return out


BULLET = regex.compile(r"^(?:(?:[•▪●◦‣⁃■□➢►\-–—*·])\s+|\(?[a-zA-Z][.)]\s)|^\(?\d{1,3}[.)](?:\s+|(?=\p{L}))")
STOP = regex.compile(r"[.!?。…:;।۔؟։]\W*$")


SYMBOL = regex.compile(r"[\uf020-\uf0ff]")  # a symbol font's codes: ASCII below 0x80, bullets and arrows above


def _symbol(m):
    b = ord(m.group()) - 0xF000
    return chr(b) if b < 0x80 else SYMBOL_BULLET.get(b, "\u2022")


PRESENTATION = regex.compile(r"[\ufb50-\ufdff\ufe70-\ufefc]+")
ARABIC_LETTERS = str.maketrans(
    "ھیکۀ", "هيكة"
)  # an Arabic text set in a Persian / Urdu font: its heh, yeh, kaf take those codes
HOMOGLYPH = {
    "Cyrillic": dict(zip("ABCEHKMOPTXYaceopxyi", "АВСЕНКМОРТХУасеорхуі")),
    "Greek": dict(zip("ABEHIKMNOPTXYZovu", "ΑΒΕΗΙΚΜΝΟΡΤΧΥΖονυ")),
}


def _lam_alef(w, lang):
    """Arabic-script word read back from a visual-order line: a lam-alef ligature (one glyph) comes out as alef-lam (the line is reversed, the pair is not).
    The occurrences of alef-lam are turned back to lam-alef where that gives the commoner word."""
    try:
        from wordfreq import zipf_frequency
    except ImportError:
        return w
    spots = [i for i in range(len(w) - 1) if w[i] in "\u0627\u0623\u0625\u0622" and w[i + 1] == "\u0644"][
        :6
    ]  # alef, hamza alefs, madda alef
    if not spots:
        return w
    best, top = w, zipf_frequency(w, lang)
    for mask in range(1, 1 << len(spots)):
        v = list(w)
        for k, i in enumerate(spots):
            if mask >> k & 1:
                v[i], v[i + 1] = "\u0644", w[i]
        v = "".join(v)
        z = zipf_frequency(v, lang)
        if z > top + 0.5:
            best, top = v, z
    return best


def _word(w, lang=None):
    """A word with Latin look-alikes of its own script's letters (a Latin A inside a Cyrillic word) written with the script's letters."""
    sc = Counter(script(c) for c in w if c.isalpha())
    main = sc.most_common(1)[0][0] if sc else None
    if main == "Arabic" and lang in ("ar", "fa", "ur"):
        w = _lam_alef(w, lang)
    if main == "Arabic" and lang == "ar":
        w = w.translate(ARABIC_LETTERS)
    return (
        "".join(HOMOGLYPH[main].get(c, c) if main in HOMOGLYPH and script(c) == "Latin" else c for c in w)
        if main in HOMOGLYPH and sc["Latin"]
        else w
    )


def _glue(t):
    """A space before a dependent sign (vowel sign, virama, accent) is a positioning gap of the glyph pieces, not a word break."""
    return "".join(
        c
        for i, c in enumerate(t)
        if not (c == " " and 0 < i < len(t) - 1 and unicodedata.category(t[i + 1])[0] == "M" and t[i - 1] != " ")
    )


def _spans_text(spans):
    """The spans of a line joined, with a space where two of them stand apart by more than pdfminer's word margin (x the size) and neither has one
    (words set as lines of their own)."""
    out = []
    for a, b in zip([None] + spans[:-1], spans):
        if (
            a is not None
            and out
            and not out[-1].endswith(" ")
            and not b["text"].startswith(" ")
            and b["bbox"].x0 - a["bbox"].x1 > R.C.word_margin * b["size"]
        ):
            out.append(" ")
        out.append(b["text"])
    return "".join(out)


def _line_text(line, lang=None):
    t = _glue(_spans_text(line["spans"]))
    t = SYMBOL.sub(_symbol, t)
    t = PRESENTATION.sub(lambda m: unicodedata.normalize("NFKC", m.group()), t)  # Arabic presentation forms -> letters
    return " ".join(_word(w, lang) for w in re.sub(r"\s+", " ", t).strip().split(" "))


def page_lines(page, lang):
    """[{text, bbox, size, spans}] of every text line of a page as the restyling reader reads it (legacy-font runs and outline text decoded for lang,
    header and footer kept): the one text reader of the page for every tool."""
    with legacy.language(lang):
        blocks = R.page_blocks(R.Source(page.parent), page.number, True)
    return [
        dict(text=t, bbox=tuple(l["bbox"]), size=b["size"], spans=l["spans"])
        for b in blocks
        for l in b["lines"]
        if (t := _line_text(l, lang))
    ]


def page_paragraphs(page, lang):
    """[{text, bbox, size, bold, role}] of a page (blocks_of read for lang: legacy fonts decoded; role h / p / li / l / v)."""
    with legacy.language(lang):
        bl = blocks_of(page)
    return [dict(text=t, bbox=box, size=size, bold=bool(e.get("bold")), role=k) for k, t, box, size, e in bl]


def page_words(page, lang, lines=None):
    """[(x0, y0, x1, y1, word)] of a page (its page_lines, or lines read already): a span's width shared among its words by their characters."""
    out = []
    for ln in page_lines(page, lang) if lines is None else lines:
        for sp in ln["spans"]:
            t, (x0, y0, x1, y1) = sp["text"], tuple(sp["bbox"])
            step = (x1 - x0) / max(len(t), 1)
            out += [(x0 + m.start() * step, y0, x0 + m.end() * step, y1, m.group()) for m in re.finditer(r"\S+", t)]
    return out


def joined(texts):
    t = ""
    for ln in texts:
        ln = ln.strip()
        if t.endswith("-") and ln[:1].islower():
            t = t[:-1] + ln
        else:
            t += (" " if t else "") + ln
    return t


def blocks_of(page, poem=False, blocks=None):
    """Items of one page in reading order: (kind, text, box, size, {align, lvl}); kind h heading, p paragraph, li list item, l a line on its
    own (names, addresses, form and table rows, contents entries), v verse line. The page is read by the restyling reader
    (layout.refont: text repaired from the font programs where the text map is wrong, Indic and RTL order, rows, columns, paragraphs,
    alignment); everything it reads is kept, only the header / footer zones of the page are left out."""
    blocks = R.page_blocks(R.Source(page.parent), page.number, False) if blocks is None else blocks
    if not blocks:
        return []
    body = R.body_size(blocks) or max(b["size"] for b in blocks)
    out = []
    for b in sorted(blocks, key=lambda b: (b["bbox"].y0, b["bbox"].x0)):
        lines = [(l, _line_text(l, legacy.LANG.get())) for l in b["lines"]]
        lines = [(l, t) for l, t in lines if t]
        if not lines:
            continue
        align = b["align"] if b["align"] in ("center", "right") else None
        spans = [s for l, _ in lines for s in l["spans"]]
        text = joined(t for _, t in lines)
        bold = all(s["bold"] for s in spans)
        extra = {**(dict(align=align) if align else {}), **(dict(bold=True) if bold else {})}
        box = tuple(b["bbox"])
        tabbed = any(l.get("tabbed") for l, _ in lines)
        if tabbed or (len(lines) >= 2 and align):
            out += [
                ("l", t, tuple(l["bbox"]), b["size"], extra) for l, t in lines
            ]  # rows of a form or table, a centred title: as set
        elif poem and len(lines) >= 2:
            out += [("v", t, tuple(l["bbox"]), b["size"], extra) for l, t in lines]
        elif BULLET.match(text) or b["lines"][0].get("marked") or _glyph_marker(lines[0][0]):
            text = re.sub(r"^[\ue000-\uf8ff]\s*", "", text)  # a bullet of a symbol font: the writer's own marker
            col0 = b["colspan"][0] if b.get("colspan") else box[0]
            out.append(("li", text, box, b["size"], dict(extra, lvl=min(int(max(box[0] - col0, 0) / (1.5 * body)), 3))))
        elif len(lines) <= 2 and len(text) <= 110 and (b["size"] >= 1.12 * body or bold) and not STOP.search(text):
            out.append(("h", text, box, b["size"], extra))
        elif len(lines) == 1:
            out.append(("l", text, box, b["size"], extra))
        else:
            out.append(("p", text, box, b["size"], extra))
    return _tables(_hanging(out))


def _glyph_marker(line):
    """The line starts with a picture glyph (a bullet of a symbol font, an icon) or a private-use character."""
    sp = line["spans"][0]
    return sp.get("glyph") is not None or sp.get("icon") or bool(re.match(r"\s*[\ue000-\uf8ff]", sp["text"]))


def _hanging(items):
    """A line under a list item that starts right of its marker (or under its text) and follows closely goes on with the item."""
    out = []
    for it in items:
        o = out[-1] if out else None
        if (
            o
            and o[0] == "li"
            and it[0] in ("l", "p")
            and not BULLET.match(it[1])
            and abs(it[3] - o[3]) < 0.5
            and 0 <= it[2][1] - o[2][3] < 0.6 * it[3]
            and o[2][0] + 0.5 * it[3] < it[2][0] < o[2][0] + 4 * it[3]
        ):
            out[-1] = ("li", joined([o[1], it[1]]), (o[2][0], o[2][1], max(o[2][2], it[2][2]), it[2][3]), o[3], o[4])
        else:
            out.append(it)
    return out


def _tables(items):
    """Rows of short items side by side (cells of a table, ruled or not) become one table item: bands of items that overlap vertically,
    two or more items apart in a band, two or more such bands one under the other; columns from the left edges of the cells."""
    order = sorted(range(len(items)), key=lambda i: items[i][2][1])
    bands = []
    for i in order:
        k, t, (x0, y0, x1, y1), sz, _ = items[i]
        for bd in bands[-3:]:
            by0, by1 = bd["y"]
            if min(by1, y1) - max(by0, y0) > 0.5 * min(by1 - by0, y1 - y0) and all(
                x1 <= items[j][2][0] or x0 >= items[j][2][2] for j in bd["ix"]
            ):
                bd["ix"].append(i)
                bd["y"] = (min(by0, y0), max(by1, y1))
                break
        else:
            bands.append(dict(ix=[i], y=(y0, y1)))
    short = lambda i: items[i][0] in ("l", "p", "h") and items[i][2][3] - items[i][2][1] <= 3.5 * items[i][3]
    groups, cur = [], []
    for bd in bands:
        cells = len(bd["ix"]) >= 2 and all(short(i) for i in bd["ix"])
        sz = max(items[i][3] for i in bd["ix"])
        if cells and cur and bd["y"][0] - cur[-1]["y"][1] < 2.5 * sz:
            cur.append(bd)
        else:
            if len(cur) >= 2:
                groups.append(cur)
            cur = [bd] if cells else []
    if len(cur) >= 2:
        groups.append(cur)
    gone, tables = set(), []
    for g in groups:
        ix = [i for bd in g for i in bd["ix"]]
        sz = float(np.median([items[i][3] for i in ix]))
        lefts = sorted(items[i][2][0] for i in ix)
        cols = [lefts[0]]
        for x in lefts[1:]:
            if x - cols[-1] > 1.5 * sz:
                cols.append(x)
        if len(cols) < 2:
            continue
        rows = []
        for bd in g:
            row = [""] * len(cols)
            for i in sorted(bd["ix"], key=lambda i: items[i][2][0]):
                c = max(j for j, x in enumerate(cols) if x <= items[i][2][0] + 1.5 * sz)
                row[c] = (row[c] + " " + items[i][1]).strip()
            rows.append(row)
        box = (
            min(items[i][2][0] for i in ix),
            min(items[i][2][1] for i in ix),
            max(items[i][2][2] for i in ix),
            max(items[i][2][3] for i in ix),
        )
        tables.append(("table", " ".join(c for r in rows for c in r if c), box, sz, dict(rows=rows)))
        gone |= set(ix)
    return sorted([it for i, it in enumerate(items) if i not in gone] + tables, key=lambda it: (it[2][1], it[2][0]))


def passage(job):
    """One passage of a document: consecutive items of a random page (headings, paragraphs, list items, lines, verse) with the blank lines,
    alignment and columns of the page."""
    c, seed = job
    rng = random.Random(seed)
    legacy.LANG.set(c["lang"])  # legacy-font runs are decoded by the tables / glyph shapes of layout.legacy
    try:
        pdf = fitz.open(c["path"])
        n = len(pdf)
        lo = int(0.1 * n) if n >= 20 else 0
        hi = n - int(0.08 * n) if n >= 20 else n
        pages = (
            [c["page"] - 1] if c.get("page") else list(range(lo, max(hi, lo + 1)))
        )  # a given page (seed layouts) or any page of the document
        rng.shuffle(pages)
        for pg in pages[:4]:
            bl = blocks_of(pdf[pg], c["kind"] == "poem")
            loose = c.get("layout")  # a page of seed layouts: lines, list items and headings are a page too
            starts = [i for i, b in enumerate(bl) if (loose or b[0] in ("h", "p", "v")) and not b[1][:1].islower()]
            if not starts or not (loose or any(b[0] in ("p", "v") for b in bl)):
                continue
            heads = [i for i in starts if bl[i][0] == "h"]
            s0 = rng.choice(heads if heads and rng.random() < 0.4 else starts)
            cut = rng.choice((1500, 2500, 4000, 6000))
            items, size, text = [], 0, ""
            for q in range(
                pg, min(pg + c.get("follow", 4), n)
            ):  # the passage goes on over the next pages until a page can be filled
                if q > pg:
                    bl = blocks_of(pdf[q], c["kind"] == "poem")
                    s = next((i for i, b in enumerate(bl) if loose or b[0] in ("h", "p", "v")), None)
                    if s is None:
                        break
                else:
                    s = s0
                win = []
                for b in bl[s:]:
                    win.append(b)
                    size += len(b[1])
                    if size >= cut:
                        break
                while win and win[-1][0] == "h":
                    win.pop()
                part = to_items(win)
                if part and items:
                    part[0]["gap"] = 1
                items += part
                if size >= cut:
                    break
            if size < 120 or not (loose or any(i["k"] in ("p", "v", "cols") for i in items)):
                continue
            text = "\n".join(walk_text(items))
            pk, _ = pagekind.kind(pdf[pg])
            return dict(
                c, items=items, text=text, page=pg, pages=n, pagekind=pk, path=os.path.relpath(c["path"], LAYOUTS)
            )
    except Exception as e:
        print("passage failed:", c["path"], repr(e)[:120], flush=True)
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-lang", type=int, default=400)
    ap.add_argument("--en", type=int, default=0, help="cap of English documents (default per-lang)")
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    cand = candidates()
    by = defaultdict(list)
    for c in cand:
        by[c["lang"]].append(c)
    print({k: len(v) for k, v in sorted(by.items(), key=lambda kv: -len(kv[1]))[:70]})
    jobs = []
    for lg, cs in by.items():
        if len(cs) < 40:
            continue
        kinds = defaultdict(list)
        for c in cs:
            kinds[c["kind"]].append(c)
        for v in kinds.values():
            rng.shuffle(v)
        order = []  # kinds in turn: every source kind is represented
        while any(kinds.values()) and len(order) < int((a.en if lg == "en" and a.en else a.per_lang) * 1.4):
            for k in list(kinds):
                if kinds[k]:
                    order.append(kinds[k].pop())
        jobs += [(c, rng.getrandbits(32)) for c in order]
        if len(order) < a.per_lang:  # small pools: a second page of every document
            jobs += [(c, rng.getrandbits(32)) for c in order]
    print(len(jobs), "documents")
    n = 0
    with Pool(a.workers) as pool, open(a.out, "w") as f:
        for r in pool.imap_unordered(passage, jobs, chunksize=8):
            if r and not garbled(r["text"]):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
    print(n, "passages ->", a.out)


if __name__ == "__main__":
    main()
