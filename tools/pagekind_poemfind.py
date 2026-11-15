"""Verse pages of the digital books (books_pdf, plan.cls text): `per` random pages of every book scored by the fast text-layer page
kind (dspages.hand.pagekind.kind); the pages most likely poems, one per book, become a candidate set of tools/pagekind_border.py
(DIR/candidates.jsonl, DIR/img/NNNN.jpg) with locally computed SigLIP 2 embeddings (DIR/embeddings.npz, at most --n pages).

python tools/pagekind_poemfind.py --out DIR [--per 4] [--n 300] [--workers 6]
"""

import argparse
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from dspages import paths as _P

sys.path.insert(0, f"{_P.CORPUS}")
import pagekind_border as B  # noqa: E402

from dspages.hand import corpus  # noqa: E402
from dspages.hand import pagekind as PK  # noqa: E402


def score(job):
    d, path, pages, seed = job
    rng = random.Random(seed)
    try:
        pdf = pymupdf.open(path)
        n = len(pdf)
        lo, hi = int(0.1 * n), max(int(0.9 * n), int(0.1 * n) + 1)
        best = (-1.0, -1)
        for no in rng.sample(range(lo, hi), min(pages, hi - lo)):
            _, p = PK.kind(pdf[no])
            best = max(best, (p["poems"], no))
        return d, str(path), best[1], best[0]
    except Exception:
        return d, str(path), -1, -1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per", type=int, default=4)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0, help="another seed: other random pages of every book")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    db = corpus.connect()
    lang = corpus.languages(db, min_conf=0.0)
    _, _, stems = B.labelled()
    used = {s.split("_")[1] for s in stems if "_" in s}
    jobs = [
        (d, corpus.resolve(p), a.per, i + 1000003 * a.seed)
        for i, (d, p) in enumerate(
            db.execute(
                "select doc,new_path from plan where cls='text' and new_path like 'books_pdf/%' and lower(new_path) like '%.pdf'"
            )
        )
        if d not in used
    ]
    with ProcessPoolExecutor(a.workers) as ex:
        res = [r for r in ex.map(score, jobs, chunksize=16) if r[2] >= 0]
    res.sort(key=lambda r: -r[3])
    print(len(res), "books scored; poem probability of the top", a.n, ":", round(res[a.n - 1][3], 3), flush=True)
    from pagetype.embedder import Embedder

    emb, X, keep = Embedder(), [], []
    for d, path, no, p in res[: a.n]:
        k = len(keep)
        B.render(path, no, out / "img" / f"{k:04d}.jpg")
        X.append(emb([Image.open(out / "img" / f"{k:04d}.jpg").convert("RGB")])[0])
        keep.append(dict(k=k, doc=d, lang=lang.get(d, "?"), pdf=path, page=no, p={"poems": round(p, 3)}, margin=None))
    with open(out / "candidates.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in keep)
    np.savez(out / "embeddings.npz", keys=np.array([str(r["k"]) for r in keep]), X=np.array(X, np.float32))
    print(len(keep), "candidates")


if __name__ == "__main__":
    main()
