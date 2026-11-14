"""Random pages of the digital poetry / verse books of the corpus (manifest topic poetry or verse, plan.cls text), for the poems class:
`pages` per book from the middle 80 % of the book, books already in pagetype_labelled left out. Writes the candidate set of
tools/pagekind_border.py (DIR/candidates.jsonl, DIR/img/NNNN.jpg) and DIR/embeddings.npz: SigLIP 2 page embeddings computed locally
(corpus_collecting pagetype.embedder, the model of the labelled embeddings), at most --max pages.

python tools/pagekind_poetry.py --out DIR [--pages 2] [--max 1000] [--docs FILE]
then: tools/pagekind_agents.py batches / collect, tools/pagekind_border.py label
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/run/media/p/D/BUISNESS/corpus_collecting")
import pagekind_border as B                                   # noqa: E402
from dspages.hand import corpus                               # noqa: E402
from dspages.hand.config import PATHS                         # noqa: E402


def books(docs=None):
    db = corpus.connect()
    lang = corpus.languages(db, min_conf=0.0)
    q = ("select p.doc,p.new_path,p.pages from old_meta m join plan p on p.doc=m.hash "
         "where (m.topic like '%poe%' or m.topic='verse') and p.cls='text' and lower(p.new_path) like '%.pdf'")
    if docs:                                             # books given by document hash (where verse was found)
        q = f"select doc,new_path,pages from plan where doc in ({','.join('?' * len(docs))})"
        return [(d, corpus.resolve(p), n or 1, lang.get(d, "?")) for d, p, n in db.execute(q, list(docs))]
    return [(d, corpus.resolve(p), n or 1, lang.get(d, "?")) for d, p, n in db.execute(q)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pages", type=int, default=2)
    ap.add_argument("--max", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--docs", help="file with document hashes, one per line: sample these books (labelled ones included)")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    _, _, stems = B.labelled()
    used = {s.split("_")[1] for s in stems if s.count("_") >= 1}
    rows = []
    docs = [l.strip() for l in open(a.docs) if l.strip()] if a.docs else None
    for d, p, n, lg in books(docs):
        if (d in used and not docs) or not p.exists():
            continue
        lo, hi = int(0.1 * n), max(int(0.9 * n), int(0.1 * n) + 1)
        for no in rng.sample(range(lo, hi), min(a.pages, hi - lo)):
            rows.append(dict(doc=d, lang=lg, pdf=str(p), page=no))
    rng.shuffle(rows)
    rows = rows[:a.max]
    from pagetype.embedder import Embedder
    emb, X, keep = Embedder(), [], []
    for r in rows:
        k = len(keep)
        try:
            B.render(r["pdf"], r["page"], out / "img" / f"{k:04d}.jpg")
        except Exception:
            continue
        X.append(emb([Image.open(out / "img" / f"{k:04d}.jpg").convert("RGB")])[0])
        keep.append(dict(r, k=k, p={}, margin=None))
    with open(out / "candidates.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in keep)
    np.savez(out / "embeddings.npz", keys=np.array([str(r["k"]) for r in keep]), X=np.array(X, np.float32))
    print(len(keep), "pages from", len({r["doc"] for r in keep}), "books; embeddings", np.array(X).shape)


if __name__ == "__main__":
    main()
