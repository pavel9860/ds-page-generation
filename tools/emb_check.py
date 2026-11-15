"""Do the embedded rows belong to the pages they are filed under? A random sample of embedded pages is rendered again from the file of the manifest
(shards like every other: tools/pages_1m.py build), embedded on the Kaggle GPUs by the same code, and every fresh vector is compared with the stored one.

  python tools/emb_check.py select [--n 3000]   # random rows of ids.npy -> info/emb_check/select.parquet (doc, page, path)
  python tools/emb_check.py compare             # (after the kernel's output is in info/emb_check/out) cosine per page, the worst ones, per-document summary"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dspages.hand.config import PATHS
from dspages.ingest.embstore import keys
from dspages.ingest.paths import EMB

LAYOUTS = Path(PATHS.layouts)
WORK = LAYOUTS / "archive" / "info" / "emb_check"


def cmd_select(a):
    ids = np.load(EMB / "ids.npy")
    man = {json.loads(l)["id"]: json.loads(l)["path"] for l in open(EMB / "manifest.jsonl")}
    pick = np.sort(np.random.default_rng(a.seed).choice(len(ids), a.n, replace=False))
    S = pd.DataFrame({"doc": ids[pick, 0], "page": ids[pick, 1]})
    S["path"] = S.doc.map(man)
    WORK.mkdir(parents=True, exist_ok=True)
    S.to_parquet(WORK / "select.parquet")
    print(len(S), "pages of", S.doc.nunique(), "documents;", int(S.path.isna().sum()), "without a path")


def cmd_compare(a):
    Eb, Ib = np.load(WORK / "out" / "embeddings.npy").astype(np.float32), np.load(WORK / "out" / "ids.npy")
    Ea, Ia = np.load(EMB / "embeddings.npy", mmap_mode="r"), np.load(EMB / "ids.npy")
    ka = keys(Ia)
    at = np.searchsorted(ka, keys(Ib))
    assert (ka[at] == keys(Ib)).all(), "sampled rows no longer in emb_out"
    old = np.asarray(Ea[at], np.float32)
    cos = (old * Eb).sum(1) / np.linalg.norm(old, axis=1) / np.linalg.norm(Eb, axis=1)
    R = pd.DataFrame({"doc": Ib[:, 0], "page": Ib[:, 1], "cos": cos})
    R.to_parquet(WORK / "compare.parquet")
    S = pd.read_parquet(WORK / "select.parquet")
    print(len(R), "of", len(S), "sampled pages compared (the rest did not render)")
    print("cosine quantiles:", R.cos.quantile([0, 0.001, 0.01, 0.05, 0.5]).round(4).to_dict())
    for t in (0.99, 0.97, 0.95, 0.9):
        print(f"below {t}:", int((R.cos < t).sum()))
    bad = R[R.cos < 0.97].merge(S[["doc", "path"]].drop_duplicates(), on="doc").sort_values("cos")
    print(bad.head(25).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("select")
    sp.add_argument("--n", type=int, default=3000)
    sp.add_argument("--seed", type=int, default=7)
    sp.set_defaults(f=cmd_select)
    sub.add_parser("compare").set_defaults(f=cmd_compare)
    a = ap.parse_args()
    a.f(a)
