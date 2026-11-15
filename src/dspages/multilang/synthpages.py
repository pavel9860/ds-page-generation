"""Synthetic pages of every type of the language dataset from a language's own text (the text pool: real lines of its documents; the translated documents for the languages without a pool): plain and formatted text, tables, forms, magazine pages (headline, two columns, a pull quote), flyers (headline, bands, a box, short blocks), handwritten pages and, from a form written to a PDF, filled forms."""
import hashlib
import json
import random
import re
from pathlib import Path

import pymupdf
from PIL import Image

from . import langdoc as D
from .config import LANGDS as C

_E = D._e


def part_of(text):
    """'val' for a fifth of the texts (by hash), else 'train'"""
    return "val" if int(hashlib.sha1(text.encode()).hexdigest(), 16) % 5 == 0 else "train"


def load_pools(lang, out, trans_dir=None, part=None):
    """{p, h, li}: the real (or agent-written) paragraphs"""
    f = Path(out) / "paras" / f"{C.alias.get(lang, lang)}.json"
    pool = json.loads(f.read_text()) if f.exists() else {}
    from .langcheck import clean_ok, repair
    lang = C.alias.get(lang, lang)
    pool = {k: [x.strip() for x in map(repair, pool.get(k, [])) if x.strip() and clean_ok(x, lang)] for k in ("p", "h", "li")}
    if len(pool["p"]) < 40 and trans_dir:
        for g in sorted(Path(trans_dir).glob(f"out_{lang}_?.json")):
            for v in json.loads(g.read_text()).values():
                pool["p"] += [x.strip() for x in re.sub(r"(?<=[.。।؟!?։]) +|(?<=[。।])", "\n", v).splitlines()
                              if len(x.strip()) >= 30 and "_" not in x and clean_ok(x, lang, 0.25)]
    if part:
        pool = {k: [x for x in v if part_of(x) == part] for k, v in pool.items()}
    pool.update(field_pool(lang, trans_dir))
    return pool


def field_pool(lang, trans_dir):
    """Form labels, table headings and cell values, titles"""
    out = dict(labels=[], heads=[], cells=[], titles_table=[], titles_form=[])
    src = json.loads((Path(trans_dir) / "src.json").read_text())
    for doc in (3, 4, 5):
        f = Path(trans_dir) / f"out_{lang}_{doc}.json"
        tr = json.loads(f.read_text()) if f.exists() else {}
        cells = []
        for k, b in enumerate(src[doc]["blocks"]):
            t = tr.get(f"{doc}_{k}", b["text"] if lang == "en" else "").strip()
            if not t:
                continue
            if b["role"] == "cell":
                cells.append(t)
            elif b["role"] == "h" and doc != 5:
                out["titles_table"].append(t.rstrip(":"))
            elif doc == 5 and "_" in t:
                out["labels"].append(re.sub(r"[\s_:：]+$", "", t.split("_")[0]).strip())
        if doc != 5:
            out["heads"] += cells[:4]
            out["cells"] += [c for c in cells[4:] if not re.fullmatch(r"[\d\s.,:/%-]+", c)]
        else:
            out["titles_form"] += [t for t in (tr.get("5_0", ""),) if t]
    out["labels"] = [x for x in dict.fromkeys(out["labels"]) if x]
    return out


class Text:
    """Pieces of text of one language from whole paragraphs, headings and list items (never line fragments)."""

    def __init__(self, pool, rng, script):
        self.pool, self.rng, self.sep = pool, rng, "" if script in C.nospace else " "
        self.used = set()

    def _cut(self, t, lo, hi):
        if not self.sep:
            return t[:self.rng.randint(lo * 3, hi * 3)]
        return " ".join(t.split()[:self.rng.randint(lo, hi)])

    def heading(self):
        return self.rng.choice(self.pool["h"] or self.pool["li"] or self.pool["p"]) if (self.pool["h"] or self.pool["li"]) else self.words(2, 6)

    def item(self):
        return self.rng.choice(self.pool["li"] or self.pool["h"]) if (self.pool["li"] or self.pool["h"]) else self.words(3, 8)

    def words(self, lo, hi):
        src = self.pool["h"] + self.pool["li"] or self.pool["p"]
        return self._cut(self.rng.choice(src), lo, hi).strip(" :;,.")

    def para(self, lo=1, hi=1):
        k = self.rng.randint(lo, hi)
        free = [x for x in self.pool["p"] if x not in self.used] or self.pool["p"]
        got = self.rng.sample(free, min(k, len(free)))
        self.used.update(got)
        return self.sep.join(got)

    def pick(self, key, k=1):
        """k distinct entries of the vocabulary `key` (the whole of it, shuffled, when it is shorter)."""
        v = self.pool[key]
        return [v[i] for i in sorted(self.rng.sample(range(len(v)), min(k, len(v))))]

    def num(self):
        r = self.rng
        return r.choice([f"{r.randint(1, 999)}", f"{r.randint(1, 99)}.{r.randint(0, 99):02d}", f"{r.randint(1, 28):02d}.{r.randint(1, 12):02d}.{r.randint(1990, 2024)}"])


def common_chars(pool, share=0.995):
    """The characters that make up `share` of the pool's text, as a string"""
    from collections import Counter
    c = Counter(ch for k in ("p", "h", "li") for x in pool[k] for ch in x if not ch.isspace())
    out, tot, run = [], sum(c.values()), 0
    for ch, n in c.most_common():
        out.append(ch)
        run += n
        if run >= share * tot:
            break
    return "".join(out)


def body_of(kind, T, n):
    """HTML of a page of `kind` built from the text source T with `n` units (blocks, rows, paragraphs)"""
    r = T.rng
    if kind == "text":
        out = []
        for _ in range(n):
            k = r.random()
            out.append(f"<h2>{_E(T.heading())}</h2>" if k < 0.15 else f'<p class="li">{r.choice(["•", "–", f"{r.randint(1, 9)}."])} {_E(T.item())}</p>'
                       if k < 0.35 else f"<p>{_E(T.para())}</p>")
        return "".join(out)
    if kind == "table":
        cols = r.randint(3, 5)
        heads = T.pick("heads", cols) if T.pool["heads"] else [T.words(1, 2) for _ in range(cols)]
        num = {c for c in range(len(heads)) if r.random() < 0.4}
        head = "".join(f"<td><b>{_E(h)}</b></td>" for h in heads)
        cell = (lambda: r.choice(T.pool["cells"])) if T.pool["cells"] else (lambda: T.words(1, 3))
        body = "".join("<tr>" + "".join(f"<td>{_E(T.num() if c in num else cell())}</td>" for c in range(len(heads))) + "</tr>" for _ in range(n))
        title = r.choice(T.pool["titles_table"]) if T.pool["titles_table"] else T.heading()
        return f"<h2>{_E(title)}</h2><table><tr>{head}</tr>{body}</table>"
    if kind == "form":
        title = r.choice(T.pool["titles_form"]) if T.pool["titles_form"] else T.heading()
        labels = T.pick("labels", n) if T.pool["labels"] else [T.words(1, 4) for _ in range(n)]
        rows = "".join(f'<tr><td class="l">{_E(x)}:</td><td class="line">&nbsp;</td></tr>' for x in labels)
        notes = "".join(f"<p>{_E(T.para())}</p>" for _ in range(n - len(labels)))
        return f'<h2>{_E(title)}</h2><p>{_E(T.para())}</p><table class="f">{rows}</table>{notes}'
    if kind == "magazine":
        cols = ["".join(f"<p>{_E(T.para())}</p>" for _ in range(n)) for _ in range(2)]
        quote = f'<table class="box"><tr><td class="bx"><b>{_E(T.item())}</b></td></tr></table>'
        return (f"<h1>{_E(T.heading())}</h1><p><i>{_E(T.item())}</i></p><table><tr><td class=\"col\">{cols[0]}{quote}</td>"
                f"<td class=\"col\">{cols[1]}</td></tr></table>")
    if kind == "flyer":
        items = "".join(f'<p class="li">• {_E(T.item())}</p>' for _ in range(n))
        return (f'<table class="box"><tr><td class="band"><h1>{_E(T.heading())}</h1></td></tr></table><h2>{_E(T.heading())}</h2>'
                f'<p>{_E(T.para())}</p>{items}<table class="box"><tr><td class="bx"><b>{_E(T.item())}</b><br>{_E(T.num())} {_E(T.num())}</td></tr></table>'
                f"<p>{_E(T.item())}</p>")
    raise ValueError(kind)


def line_boxes(pdf):
    """Boxes of the printed lines of a rendered page in the 1024 px window"""
    from hwsynth.scripts import text_script
    k = C.side / C.pt
    out = []
    with pymupdf.open("pdf", pdf) as doc:
        for b in doc[0].get_text("dict")["blocks"]:
            for ln in b.get("lines", []):
                t = "".join(sp["text"] for sp in ln["spans"]).strip()
                if t:
                    out.append(dict(k="print", b=[round(v * k) for v in ln["bbox"]], script=text_script(t)))
    return out


def make_synth(job):
    """(kind, lang, seed, out_dir, pool, split) -> row dict or None"""
    kind, lang, seed, out, pool, split = job
    from hwsynth.scripts import text_script
    from hwsynth.writer import _index
    rng = random.Random(f"{split}{kind}{lang}{seed}")
    try:
        script = text_script(" ".join(pool["p"][:6])[:600] or "a")
        T = Text(pool, rng, script)
        rid = hashlib.sha1(f"{split}{kind}{lang}{seed}".encode()).hexdigest()[:12]
        d = Path(out) / "synth" / split / kind / lang
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"{rid}.jpg"
        row = dict(id=rid, file=f"synth/{split}/{kind}/{lang}/{rid}.jpg", lang=lang, type=kind, scan=False, lang_status="synthetic",
                   split=split, script=script)
        if kind == "handwritten":
            rl = [("h", T.heading())] if rng.random() < 0.5 else []
            rl += [("p", T.para()) for _ in range(30)]
            import numpy as np
            hb = []
            page = D.hand_page(rl, D.writer_of(rng), rng, boxes=hb, rtl=lang in D.RTL)
            if not hb:
                return None
            Image.fromarray(np.asarray(page).clip(0, 255).astype("uint8")).save(f, quality=88)
            return dict(row, boxes=[dict(b, script=script) for b in hb])
        index = _index()
        font = rng.choice(D.T.fonts_for(common_chars(pool), index, 4))
        pool = {k: [x for x in v if index.covers(font, x)] if k in ("p", "h", "li") else v for k, v in pool.items()}
        if len(pool["p"]) < 3:
            return None
        lo, hi = C.font_pt[kind]
        fs = rng.uniform(lo + (1 if script in C.complex_scripts else 0), hi + (1 if script in C.complex_scripts else 0))
        m, lead = rng.uniform(0, C.margin_max) * C.pt, rng.uniform(1.15, 1.45)
        fixed = (round(fs, 1), m, lead)
        best = None
        for n in range(1, C.max_units[kind] + 1):
            pdf = D.page_pdf(body_of(kind, Text(pool, random.Random(f"{rid}"), script), n), font, lang, rng, None, fixed=fixed, shrink=False)
            if pdf is None:
                break
            best = (n, pdf)
        if best is None:
            return None
        n = max(1, round(best[0] * rng.uniform(0.85, 1.0)))
        pdf = D.page_pdf(body_of(kind, Text(pool, random.Random(f"{rid}"), script), n), font, lang, rng, None, fixed=fixed, shrink=False)
        with pymupdf.open("pdf", pdf) as d:
            pix = d[0].get_pixmap(dpi=C.dpi, colorspace=pymupdf.csGRAY)
            Image.frombytes("L", (pix.width, pix.height), pix.samples).resize((C.side, C.side)).save(f, quality=88)
        return dict(row, boxes=line_boxes(pdf), font_pt=fixed[0], margin_pt=round(m, 1))
    except Exception:
        import os
        if os.environ.get("HWDEBUG"):
            raise
        return None
