"""Source pages spread over the ds_head space (16-d, tools/ds_head.py): the in-scope pages of the corpus with their vectors, and a farthest-first
ranking of the pages of a candidate set, so that every chosen page is as different in layout and look as possible from the ones before."""

import numpy as np
import pandas as pd

from ..fillpic.paths import doc_hash
from ..ingest.litefeat import farthest_first
from .config import PATHS

AUX = f"{PATHS.layouts}/archive/info/ds100k"
_cache = {}


def space():
    """(vectors, DataFrame h = document hash, page) of the in-scope pages."""
    if not _cache:
        z = np.load(f"{AUX}/targets/aux/Z.npz")
        P = pd.read_parquet(f"{AUX}/pool.parquet", columns=["path", "page"]).iloc[z["pool_i"]]
        _cache["Z"] = z["Z"].astype(np.float32)
        _cache["P"] = pd.DataFrame({"h": P.path.map(doc_hash).to_numpy(), "page": P.page.to_numpy()})
    return _cache["Z"], _cache["P"]


def ranked(docs, k=80):
    """docs: set of document hashes -> [(hash, page)] of their in-scope pages, farthest-first (at most k)."""
    Z, P = space()
    ix = np.flatnonzero(P.h.isin(docs).to_numpy())
    if not len(ix):
        return []
    o = farthest_first(Z[ix], [], min(k, len(ix)))
    return [(P.h.iat[ix[i]], int(P.page.iat[ix[i]])) for i in o]
