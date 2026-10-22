"""Handwritten versions of labelled pages (pagetype labels.json): the digital text of a page of the class with its structure (headings by
size, blank lines where the page has gaps, blocks side by side as columns), written by hand on A4; the original page beside it.

python -m dspages.hand labelled --out DIR [--label formatted] [--n 6]"""
import argparse
import html
import json
import random
import re
from pathlib import Path

import cv2
import pymupdf

from . import corpus, page
from . import texts as T
from .config import PATHS
from .langs import script_ok
from .structure import to_items
from .util import flat_items

PERSONS = ["school_cursive", "round_print", "ruled_neat", "rounded_neat", "soft_upright", "compact_print", "print_script", "backhand"]
PENS = ["ballpoint", "gel", "fineliner", "fountain_fine", "ballpoint_fine"]
PAPERS = ["blank", "lined", "dotted", "margin", "squared"]


def pick(label, n, rng):
    paths = corpus.pdf_paths(corpus.connect())
    cand = [p for p in json.load(open(PATHS.labels))["pages"] if p["label"] == label and p.get("src") and p.get("kind") == "page"]
    rng.shuffle(cand)
    out, seen = [], set()
    for p in cand:
        stem = Path(p["src"]).stem
        f = paths.get(stem.split("_")[0])
        if f is None or not f.exists() or p["src"] in seen:
            continue
        m = re.search(r"_p(\d+)$", stem)
        try:
            pdf = pymupdf.open(f)
            no = min((int(m.group(1)) if m else 0) + p["page"], len(pdf) - 1)       # a labelled piece is page k of its document
            blocks = T.blocks_of(pdf[no], False, with_box=True)
        except Exception:
            continue
        text = " ".join(b[1] for b in blocks)
        lang = f.stem.split("_")[-1]
        if len(blocks) >= 5 and 500 <= len(text) <= 3200 and any(b[0] in ("h", "li") for b in blocks) and script_ok(text, lang if lang in ("ru", "uk", "bg") else "en"):
            seen.add(p["src"])
            out.append((f, no, to_items(blocks)))
        if len(out) == n:
            break
    return out


def write_pair(out, k, path, no, items, rng, label):
    plan = dict(i=k, seed=1000 + 17 * k, wseed=50 + k, persona=rng.choice(PERSONS), style=rng.choice(["casual", "accurate", "educated", "professional"]),
                pen=rng.choice(PENS), paper=rng.choice(PAPERS), fmt="a4", dense=False, xh=rng.uniform(1.8, 2.6), xh_ratio=rng.uniform(0.28, 0.4),
                fill=0.97, dy=0.2, strict=True)
    img, meta = page.make(plan, items)
    cv2.imwrite(str(out / f"{k:02d}_{label}_hw.jpg"), img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
    pymupdf.open(path)[no].get_pixmap(dpi=90).save(str(out / f"{k:02d}_{label}_orig.png"))
    return plan, meta


def sheet(out, rows, name):
    css = ("body{font:13px sans-serif;margin:12px}section{display:flex;gap:14px;margin-bottom:26px;border-bottom:1px solid #ccc;padding-bottom:12px}"
           "img{width:430px;border:1px solid #aaa}.t{max-width:380px;max-height:640px;overflow:auto}p{margin:3px 0}b{color:#a00;font-size:11px;margin-right:5px}h4{margin:2px 0 6px}")
    (out / name).write_text(f"<!doctype html><meta charset=utf-8><style>{css}</style>" + "".join(rows))


def row(k, label, path, no, items, plan, meta):
    txt = "".join(f"<p><b>{i['k']}{' gap' + str(i['gap']) if i.get('gap') else ''}</b> {html.escape(i['t'])}</p>" for i in flat_items(items) if "t" in i)
    cols = sum(i["k"] == "cols" for i in items)
    return (f"<section><div><h4>original: {html.escape(Path(path).name)} p{no}</h4><img src='{k:02d}_{label}_orig.png'></div><div><h4>handwritten (A4, {plan['paper']}, "
            f"{plan['persona']}, {plan['pen']}, xh {meta['xh_mm']} mm, {meta['lines']} lines, {cols} column bands)</h4><img src='{k:02d}_{label}_hw.jpg'></div>"
            f"<div class='t'><h4>text written</h4>{txt}</div></section>")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="formatted")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    rows = []
    for k, (path, no, items) in enumerate(pick(a.label, a.n, rng)):
        plan, meta = write_pair(out, k, path, no, items, rng, a.label)
        rows.append(row(k, a.label, path, no, items, plan, meta))
        print(k, path.name, no, flush=True)
    sheet(out, rows, f"{a.label}_pairs.html")


if __name__ == "__main__":
    main()
