"""Source PDF page of every labelled plain / poems / formatted image (pagetype_labelled/<label>/*.jpg): the page named by labels.json, or for
images without a label record the document named by the file (<prefix>_<hash>_...), checked by matching the rendered page against the image
(normalised correlation of 64 x 90 grey thumbnails; candidates around the named page, all pages of short documents).

python tools/pagekind_sources.py [--out UVTM/pagetype_labelled/cache/pagekind_src.jsonl] [--workers 6]
"""

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pymupdf

from dspages.hand import corpus
from dspages.hand.config import PATHS

ROOT = Path(PATHS.uvtm) / "pagetype_labelled"
LABELS = tuple(sorted(c.name for c in ROOT.iterdir() if c.is_dir() and c.name != "cache"))  # every class folder
SIZE = (64, 90)
MIN_CORR = 0.6


def thumb(img):
    g = cv2.resize(
        img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), SIZE, interpolation=cv2.INTER_AREA
    ).astype(np.float32)
    g -= g.mean()
    return g / (np.linalg.norm(g) + 1e-6)


def page_thumb(pdf, no):
    pg = pdf[no]
    s = 120 / max(pg.rect.width, pg.rect.height)
    pix = pg.get_pixmap(matrix=pymupdf.Matrix(s, s), colorspace=pymupdf.csGRAY)
    return thumb(np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w))


def match(job):
    img_path, pdf_path, guess = job
    try:
        im = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        t = thumb(im)
        pdf = pymupdf.open(pdf_path)
        n = len(pdf)
        cand = range(n) if n <= 60 else sorted({min(max(guess + d, 0), n - 1) for d in range(-4, 5)} | {0, 1, 2})
        best = max((float((t * page_thumb(pdf, k)).sum()), k) for k in cand)
        return str(img_path), str(pdf_path), best[1], best[0]
    except Exception as e:
        return str(img_path), str(pdf_path), -1, f"{type(e).__name__}: {e}"


def jobs():
    recs = {p["id"].split("/")[-1]: p for p in json.load(open(PATHS.labels))["pages"]}
    paths = corpus.pdf_paths(corpus.connect())
    out, skipped = [], []
    for lab in LABELS:
        for f in sorted((ROOT / lab).glob("*.jpg")):
            r = recs.get(f.stem)
            if r and r.get("src", "").endswith(".pdf") and r.get("kind") == "page":
                stem = Path(r["src"]).stem
                m = re.search(r"_p(\d+)$", stem)
                guess = (int(m.group(1)) if m else 0) + r.get("page", 0)
                doc = stem.split("_")[0].split(".")[0]
            else:
                m = re.match(r"^[a-z0-9]+_([0-9a-f]{12})_.*?(?:_p(\d+))?$", f.stem)
                doc, guess = (m.group(1), int(m.group(2) or 0)) if m else (None, 0)
            p = paths.get(doc)
            if p is None or not p.exists():
                skipped.append(f"{lab}/{f.name}")
                continue
            out.append((lab, (f, p, guess)))
    return out, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "cache" / "pagekind_src.jsonl"))
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    js, skipped = jobs()
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(match, [j for _, j in js], chunksize=4))
    n = 0
    with open(a.out, "w") as f:
        for (lab, _), (img, pdf, page, corr) in zip(js, res):
            ok = isinstance(corr, float) and corr >= MIN_CORR
            n += ok
            f.write(
                json.dumps(
                    dict(
                        label=lab,
                        image=img,
                        pdf=pdf,
                        page=page,
                        corr=corr if isinstance(corr, str) else round(corr, 3),
                        ok=ok,
                    )
                )
                + "\n"
            )
    print(f"{n} of {len(js)} matched (corr >= {MIN_CORR}); no source document: {len(skipped)}", skipped[:10])


if __name__ == "__main__":
    main()
