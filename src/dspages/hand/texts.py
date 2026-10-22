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
from collections import Counter, defaultdict
from multiprocessing import Pool

import pymupdf as fitz
import regex

from . import corpus
from .config import PATHS
from .langs import script_ok

LAYOUTS = PATHS.layouts
PROSE = {"plain", "formatted", "scientific", "magazines", "bureaucracy", "poems", "recieps"}
FOLDER_KIND = {"books_pdf": "book", "scientific_paper": "paper", "text": "article", "bureaucracy": "letter",
               "exams": "exam", "magazines": "magazine"}
LABEL_KIND = {"plain": "article", "formatted": "article", "scientific": "paper", "magazines": "magazine",
              "bureaucracy": "letter", "poems": "poem", "recieps": "recipe"}
SCRIPT = {"en": "Latin"}
URL = regex.compile(r"https?:|www\.|@\w|\.{4,}|_{3,}|©|\bISBN\b|\bdoi\b", regex.I)


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


BULLET = regex.compile(r"^(?:[\u2022\u25aa\u25cf\u25e6\u2023\u2043\-\u2013\u2014*\u00b7]|\(?\d{1,2}[.)]|\(?[a-zA-Z][.)])\s+")
STOP = regex.compile(r"[.!?\u3002\u2026:;\u0964\u06d4\u061f\u0589]\W*$")
WORDS = regex.compile(r"\p{L}")


def joined(lines):
    t = ""
    for ln in lines:
        ln = ln.strip()
        if t.endswith("-") and ln[:1].islower():
            t = t[:-1] + ln
        else:
            t += (" " if t else "") + ln
    return re.sub(r"\s+", " ", t).strip()


def blocks_of(page, poem, with_box=False):
    """Items of one page in reading order: (kind, text); kind h heading, p paragraph, li list item, v verse line."""
    d = page.get_text("dict", sort=True)
    sizes = Counter()
    for b in d["blocks"]:
        for ln in b.get("lines", ()):
            for sp in ln["spans"]:
                sizes[round(sp["size"])] += len(sp["text"])
    if not sizes:
        return []
    body = sizes.most_common(1)[0][0]
    out = []
    for b in d["blocks"]:
        if b["type"] != 0:
            continue
        lines = [("".join(sp["text"] for sp in ln["spans"]), ln["spans"]) for ln in b["lines"]]
        lines = [(t, sp) for t, sp in lines if t.strip()]
        if not lines:
            continue
        t = joined(t for t, _ in lines)
        n = len(t.replace(" ", ""))
        letters = sum(c.isalpha() for c in t)
        if n < 3 or URL.search(t) or letters < 0.5 * n:
            continue
        spans = [sp for _, ss in lines for sp in ss if sp["text"].strip()]
        size = sum(sp["size"] * len(sp["text"]) for sp in spans) / max(sum(len(sp["text"]) for sp in spans), 1)
        bold = all(sp["flags"] & 16 for sp in spans)
        if len(lines) <= 2 and n <= 110 and (size >= 1.12 * body or bold) and not STOP.search(t) and letters >= 0.6 * n:
            out.append(("h", t, b["bbox"], size))
        elif BULLET.match(t) and n <= 600:
            out.append(("li", t, b["bbox"], size))
        elif len(lines) >= 3 and max(len(x.strip()) for x, _ in lines) <= 80 and sum(bool(STOP.search(x.strip())) for x, _ in lines) >= 0.7 * len(lines) \
                and not poem:                                   # a list whose items end with a full stop: one item per line
            out += [("li", x.strip(), b["bbox"], size) for x, _ in lines if x.strip()]
        elif poem and len(lines) >= 2 and max(len(x) for x, _ in lines) <= 70:
            out += [("v", x.strip(), b["bbox"], size) for x, _ in lines]
        elif n >= 40 and letters >= 0.8 * n and sum(c.isdigit() for c in t) <= 0.06 * n:
            if not (sum(c.isupper() for c in t) > 0.35 * letters and t.isascii()):
                out.append(("p", t, b["bbox"], size))
    merged = []                                         # a paragraph cut by a column or page break goes on
    for k, t, bb, sz in out:
        if merged and k == "p" and merged[-1][0] == "p" and not STOP.search(merged[-1][1]) and t[:1].islower():
            o = merged[-1]
            merged[-1] = ("p", o[1] + " " + t, (min(o[2][0], bb[0]), min(o[2][1], bb[1]), max(o[2][2], bb[2]), max(o[2][3], bb[3])), o[3])
        else:
            merged.append((k, t, bb, sz))
    return merged if with_box else [(k, t) for k, t, _, _ in merged]


def passage(job):
    """One passage of a document: consecutive items of a random page (headings, paragraphs, list items, verse)."""
    c, seed = job
    rng = random.Random(seed)
    try:
        pdf = fitz.open(c["path"])
        n = len(pdf)
        lo = int(0.1 * n) if n >= 20 else 0
        hi = n - int(0.08 * n) if n >= 20 else n
        pages = list(range(lo, max(hi, lo + 1)))
        rng.shuffle(pages)
        for pg in pages[:4]:
            bl = blocks_of(pdf[pg], c["kind"] == "poem")
            if c.get("native"):                           # bilingual documents: only the blocks in the language's own script
                bl = [(k, t) for k, t in bl if script_ok(t, c["lang"])]
            starts = [i for i, (k, t) in enumerate(bl) if k in ("h", "p", "v") and not t[:1].islower()]
            if not starts or sum(k == "p" or k == "v" for k, _ in bl) < 1:
                continue
            s = rng.choice([i for i in starts if bl[i][0] == "h"] if rng.random() < 0.4 and any(bl[i][0] == "h" for i in starts) else starts)
            items, size, cut = [], 0, rng.choice((300, 600, 1000, 1500, 2000))
            for k, t in bl[s:]:
                if k == "h" and size > 0.5 * cut and items and items[-1][0] == "h":
                    break
                items.append(dict(k=k, t=t))
                size += len(t)
                if size >= cut:
                    break
            if size < 120 or not any(i["k"] in ("p", "v") for i in items):
                continue
            while items and items[-1]["k"] == "h":      # no heading at the end of a page
                items.pop()
            text = "\n".join(i["t"] for i in items)
            return dict(c, items=items, text=text, page=pg, pages=n, path=os.path.relpath(c["path"], LAYOUTS))
    except Exception:
        pass
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
        order = []                          # kinds in turn: every source kind is represented
        while any(kinds.values()) and len(order) < int((a.en if lg == "en" and a.en else a.per_lang) * 1.4):
            for k in list(kinds):
                if kinds[k]:
                    order.append(kinds[k].pop())
        jobs += [(c, rng.getrandbits(32)) for c in order]
        if len(order) < a.per_lang:                      # small pools: a second page of every document
            jobs += [(c, rng.getrandbits(32)) for c in order]
    print(len(jobs), "documents")
    n = 0
    with Pool(a.workers) as pool, open(a.out, "w") as f:
        for r in pool.imap_unordered(passage, jobs, chunksize=8):
            if r:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
    print(n, "passages ->", a.out)


if __name__ == "__main__":
    main()
