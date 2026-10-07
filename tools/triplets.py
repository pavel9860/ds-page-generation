"""Triplets of restyled pages: the original PDF page | two restyles with different seeds (fonts) and margins.

python tools/triplets.py OUT_DIR [--manifest M] [--n 200] [--seed 0] [--workers 12] [--preset full] [--only 6,8,9]
Pages are random "refont" entries of the manifest; the margin of a restyle is drawn from the preset's range like the
generation does. Writes OUT_DIR/<i>_<category>_<file>_p<page>.png (half size).
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image

from dspages.config import get_preset
from dspages.layout import refont
from dspages.layout.template import sheet
from dspages.layout.sources import load_manifest

_W = {}


def _init(preset):
    P = get_preset(preset)
    _W.update(c=P.layout, fonts=refont.font_set(P.paths.font_dirs))


def _triplet(args):
    i, e, out, seed = args
    c = _W["c"]
    pw, ph = c.page_px
    with pymupdf.open(e["source_path"]) as doc:
        k = pw / doc[e["page_index"]].rect.width
        pm = doc[e["page_index"]].get_pixmap(matrix=pymupdf.Matrix(k, k), colorspace=pymupdf.csGRAY)
    panels = [Image.frombytes("L", (pm.w, pm.h), pm.samples)]
    rng = np.random.default_rng([seed, i])
    for _ in range(2):
        gray, _ = sheet(e["source_path"], e["page_index"], int(rng.integers(1 << 31)), _W["fonts"], rng.uniform(*c.margin_frac), (pw, ph))
        panels.append(Image.fromarray(gray))
    out_img = Image.new("L", (3 * pw + 40, max(p.height for p in panels)), 128)
    for j, p in enumerate(panels):
        out_img.paste(p, (j * (pw + 20), 0))
    name = f"{i:03d}_{e['category']}_{Path(e['source_path']).stem[:24]}_p{e['page_index']}.png"
    out_img.resize((out_img.width // 2, out_img.height // 2), Image.LANCZOS).save(Path(out) / name)
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--preset", default="full")
    ap.add_argument("--only", default=None, help="comma-separated triplet numbers to make (the others are left as they are)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pages = [e for e in load_manifest(a.manifest or get_preset(a.preset).paths.manifest) if e["kind"] == "refont"]
    only = None if a.only is None else {int(i) for i in a.only.split(",")}
    pick = np.random.default_rng(a.seed).permutation(len(pages))[:a.n]
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.preset,)) as pool:
        for name in pool.map(_triplet, [(i, pages[j], a.out, a.seed) for i, j in enumerate(pick) if only is None or i in only], chunksize=1):
            print(name, flush=True)


if __name__ == "__main__":
    main()
