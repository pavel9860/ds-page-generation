"""The same documents in every language, as clean 1024 px pages, for grouping languages by how their pages look."""

import io
import json
import os
import random
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pymupdf
import regex
from PIL import Image

from . import translate as T
from .config import KINDS  # noqa: F401
from .config import LANGDS as C

TRAIN_DOCS, VAL_DOCS = {0, 1, 3, 5}, {2, 4, 6}
RTL = T.RTL


def writer_of(rng):
    """A natural writer without the random crossed-out typos (they read as fake corrections)."""
    from .bank import new_writer

    w = new_writer(rng)
    w.p["cross"] = 0.0
    return w


def table_html(blocks, texts):
    """A table from the cell blocks"""
    head = "".join(f"<h2>{_e(t)}</h2>" for b, t in zip(blocks, texts, strict=True) if b["role"] == "h")
    items = sorted(
        ((b, t) for b, t in zip(blocks, texts, strict=True) if b["role"] != "h"),
        key=lambda x: (x[0]["bbox"][1] + x[0]["bbox"][3]) / 2,
    )
    rows, cur, last = [], [], None
    for b, t in items:
        yc = (b["bbox"][1] + b["bbox"][3]) / 2
        if last is not None and yc - last > 0.7 * b["size"] * 1.5:
            rows.append(cur)
            cur = []
        cur.append((b["bbox"][0], t))
        last = yc
    rows.append(cur)
    xs = sorted({round(x / 25) for r in rows for x, _ in r})
    col = {x: i for i, x in enumerate(xs)}
    html = [head, "<table>"]
    for r in rows:
        cells = [""] * len(xs)
        for x, t in sorted(r):
            i = col[round(x / 25)]
            cells[i] = (cells[i] + " " + t).strip()
        html.append("<tr>" + "".join(f"<td>{_e(c)}</td>" for c in cells) + "</tr>")
    return "".join(html) + "</table>"


def _e(t):
    return t.replace("&", "&amp;").replace("<", "&lt;")


def form_html(blocks, texts, fills):
    """A form: a short block is a label with its blank beside it (a ruled cell, or the handwritten fill)"""
    out, rows, k = [], [], 0

    def flush():
        if rows:
            out.append('<table class="f">' + "".join(rows) + "</table>")
            rows.clear()

    for b, t in zip(blocks, texts, strict=True):
        t = re.sub(r"_{3,}|\.{5,}", " ", t).strip()
        if b["role"] == "h":
            flush()
            out.append(f"<h2>{_e(t)}</h2>")
        elif len(t.split()) <= 9 and b["role"] != "li":
            fill = f'<img class="g{k % len(fills)}" src="{fills[k % len(fills)]}">' if fills else "&nbsp;"
            k += 1
            rows.append(f'<tr><td class="l">{_e(t)}</td><td class="line">{fill}</td></tr>')
        else:
            flush()
            out.append(f"<p>{_e(t)}</p>")
    flush()
    return "".join(out)


def text_html(rl):
    """Headings, paragraphs and list items (marker or clause number kept in the text, hanging indent) in reading order."""
    return "".join(
        {"h": f"<h2>{_e(t)}</h2>", "li": f'<p class="li">{_e(t)}</p>', "p": f"<p>{_e(t)}</p>"}[role] for role, t in rl
    )


def page_pdf(body, font, lang, rng, images=None, fs_max=22, fixed=None, with_size=False, shrink=True):
    """Fits `body` (HTML) into the window at the largest font size (leading and margin vary) and returns the PDF bytes; with `fixed` (size, margin, leading) at that size, or smaller unless `shrink` is False (then None when it does not fit)."""
    font = Path(font)
    arch = pymupdf.Archive(str(font.parent))
    for name, (data, _, _) in (images or {}).items():
        arch.add(data, name)
    m, lead = fixed[1:] if fixed else (rng.uniform(16, 30), rng.uniform(1.15, 1.45))
    where = pymupdf.Rect(m, m, C.pt - m, C.pt - m)
    for fs in [fixed[0]] if fixed and not shrink else np.arange(fixed[0] if fixed else fs_max, 5.4, -0.4):
        css = (
            f"@font-face {{font-family: tf; src: url({font.name});}} body {{font-family: tf; font-size: {fs:.1f}pt; "
            f"line-height: {lead}; direction: {'rtl' if lang in RTL else 'ltr'};}} p {{margin: 0 0 {fs * 0.5:.1f}pt 0;}} "
            f"h2 {{font-size: {fs * 1.3:.1f}pt; margin: {fs * 0.4:.1f}pt 0 {fs * 0.4:.1f}pt 0;}} p.li {{margin-left: {fs * 1.4:.1f}pt;}} "
            f"img {{height: {fs * 1.9:.1f}pt;}} "
            f"table {{border-collapse: collapse; width: 100%;}} td {{border: 0.6pt solid black; padding: 2pt;}} "
            f"h1 {{font-size: {fs * 2.2:.1f}pt; text-align: center; margin: {fs * 0.3:.1f}pt 0 {fs * 0.6:.1f}pt 0;}} "
            f"td.col {{width: 48%; vertical-align: top; padding: 2pt 7pt; border: none;}} table.box {{border: 1.6pt solid black; width: 100%;}} "
            f"td.band {{background-color: #b8b8b8; padding: {fs * 0.4:.1f}pt; border: none;}} td.bx {{padding: {fs * 0.4:.1f}pt 8pt; border: none;}} "
            f"table.f {{width: 100%;}} td.l {{width: 45%; padding: {fs * 0.25:.1f}pt 4pt;}} "
            f"td.line {{border-bottom: 0.7pt solid black; width: 55%;}} "
            + "".join(
                f"img.g{i} {{width: {w * 72 / C.dpi * fs / 11:.1f}pt;}} "
                for i, (_, (_, w, _)) in enumerate((images or {}).items())
            )
            + ""
        )
        story = pymupdf.Story(html=f"<body>{body}</body>", user_css=css, archive=arch)
        more, filled = story.place(where)
        if not more and filled[2] <= where.x1 + 0.5:
            buf = io.BytesIO()
            w = pymupdf.DocumentWriter(buf)
            dev = w.begin_page(pymupdf.Rect(0, 0, C.pt, C.pt))
            story.draw(dev)
            w.end_page()
            w.close()
            return (buf.getvalue(), float(fs)) if with_size else buf.getvalue()
    return (None, None) if with_size else None


def hand_png(text, writer, width_mm):
    """A handwritten text as a transparent PNG (dark ink on alpha) and its ink array."""
    import hwsynth as H

    s = H.write(text, writer, width_mm=width_mm, dpi=C.dpi, ruled=False, scan=False, image=False)
    ink = np.clip(s.ink, 0, 1)
    ys, xs = np.nonzero(ink > 0.05)
    ink = ink[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    a = np.zeros((*ink.shape, 4), np.uint8)
    a[..., :3] = 25
    a[..., 3] = (ink * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "PNG")
    return buf.getvalue(), ink


def _cut(t, limit=40):
    """A block of at most `limit` words: whole, or cut after the last full stop, semicolon or colon before the limit."""
    w = t.split()
    if len(w) <= limit:
        return t
    head = " ".join(w[:limit])
    k = max(head.rfind(c) for c in ".;:。")
    return head[: k + 1] if k > len(head) * 0.4 else head


def hand_page(rl, writer, rng, boxes=None, rtl=False):
    """A handwritten page: the blocks one under the other"""
    page = np.full((C.side, C.side), 255, np.float32)
    m = int(rng.uniform(0, C.margin_max) * C.side)
    y = m
    for role, t in rl:
        if not writer.fonts.writes(_cut(t)):
            continue
        try:
            _, ink = hand_png(_cut(t), writer, (C.side - 2 * m) / C.dpi * 25.4 * (1 if role != "li" else 0.92))
        except Exception:
            continue
        sc = 1.3 if role == "h" else 1.0
        if sc != 1.0:
            ink = (
                np.asarray(
                    Image.fromarray((ink * 255).astype(np.uint8)).resize(
                        (int(ink.shape[1] * sc), int(ink.shape[0] * sc))
                    )
                )
                / 255
            )
        indent = 45 if role == "li" else 0
        h, w = ink.shape[0], min(ink.shape[1], C.side - 2 * m - indent)
        x = C.side - m - w if rtl else m + indent
        if y + h > C.side - m:
            if len(t.split()) <= 12:
                break
            continue
        page[y : y + h, x : x + w] *= 1 - 0.9 * ink[:h, :w]
        if boxes is not None:
            boxes.append(dict(k="hand", b=[x, y, x + w, y + h], role=role))
        y += h + 28
        if role == "h":
            page[y - 14 : y - 12, x : x + w] *= 0.35
        if C.side - m - y < 90:
            break
    return page


def make(args):
    """(work, lang, doc, mode, v, split, out) -> path or None; mode print | filled | hand."""
    work, lang, doc, mode, v, split, out = args
    from hwsynth.writer import _index

    rng = random.Random(f"{lang}{doc}{mode}{v}")
    try:
        work = Path(work)
        src = json.loads((work / "src.json").read_text())[doc]
        tr = json.loads((work / f"out_{lang}_{doc}.json").read_text())
        pairs = [(b, tr.get(f"{doc}_{k}", b["text"])) for k, b in enumerate(src["blocks"])]
        pairs = [(b, t) for b, t in pairs if t.strip()]
        blocks, texts = [b for b, _ in pairs], [t for _, t in pairs]
        kind = KINDS[doc]
        index = _index()
        if mode == "hand":
            rl = [(b["role"], t) for b, t in zip(blocks, texts, strict=True)]
            if kind == "plain":
                rl = [("p", t) for _, t in rl]
            img = hand_page(rl, writer_of(rng), rng)
            arr = img
        else:
            font = rng.choice(T.fonts_for("".join(texts)[:400], index))
            images, fills = {}, None
            if mode == "filled":
                words = [w for w in " ".join(texts).split() if regex.fullmatch(r"(?:\p{L}\p{M}*){3,}", w)] or ["name"]
                wr = writer_of(rng)
                fills = []
                for i in range(6):
                    s = rng.randrange(max(1, len(words) - 3))
                    name = f"h{i}.png"
                    png, ink = hand_png(" ".join(words[s : s + rng.randint(1, 3)]), wr, 110)
                    images[name] = (png, ink.shape[1], ink.shape[0])
                    fills.append(name)
            body = (
                table_html(blocks, texts)
                if kind == "table"
                else form_html(blocks, texts, fills)
                if kind == "form"
                else text_html([(b["role"], t) for b, t in zip(blocks, texts, strict=True)])
            )
            pdf = page_pdf(
                body, font, lang, rng, images, fs_max={"plain": 24, "formatted": 22, "table": 38, "form": 30}[kind]
            )
            if pdf is None:
                return None
            with pymupdf.open("pdf", pdf) as d:
                pix = d[0].get_pixmap(dpi=C.dpi, colorspace=pymupdf.csGRAY)
            arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
        im = Image.fromarray(np.asarray(arr).clip(0, 255).astype(np.uint8)).resize((C.side, C.side))
        d = Path(out) / split / lang
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"{doc}{mode[0]}_{v:02d}.jpg"
        im.save(f, quality=88)
        return str(f)
    except Exception:
        if os.environ.get("HWDEBUG"):
            raise
        return None


def jobs_for(work, out, langs, variants=6):
    """Every kind for every language"""
    work = Path(work)
    jobs = []
    for lang in langs:
        for doc, mode in [(d, "print") for d in range(7)] + [
            (5, "filled"),
            (6, "filled"),
            (0, "hand"),
            (1, "hand"),
            (2, "hand"),
        ]:
            if (work / f"out_{lang}_{doc}.json").exists():
                split = "train" if doc in TRAIN_DOCS else "val"
                jobs += [(str(work), lang, doc, mode, v, split, str(out)) for v in range(variants)]
    return jobs


def build(work, out, variants=6, workers=10):
    work = Path(work)
    langs = sorted({f.stem[4:].rsplit("_", 1)[0] for f in work.glob("out_*_?.json")})
    jobs = [
        j
        for j in jobs_for(work, out, langs, variants)
        if not (Path(out) / j[5] / j[1] / f"{j[2]}{j[3][0]}_{j[4]:02d}.jpg").exists()
    ]
    made = 0
    for i in range(0, len(jobs), 8 * workers):
        batch = jobs[i : i + 8 * workers]
        try:
            with ProcessPoolExecutor(workers) as ex:
                made += sum(r is not None for r in ex.map(make, batch, chunksize=2))
        except Exception:
            made += sum(make(j) is not None for j in batch)
    return made, len(jobs)
