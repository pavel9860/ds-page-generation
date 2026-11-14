"""Merge of new embedding rows into emb_out (embeddings.npy float16 (n, 1536) + ids.npy int32 (n, 2)), rows sorted by (doc, page).

  python tools/emb_merge.py <dir> <backup> [--drop remove.parquet]   # <dir>/embeddings.npy + ids.npy (a Kaggle output) are added;
                                                                     # rows of the (doc, page) in --drop (parquet) are removed first
Backups emb_out/<file>.<backup> are made once."""
import argparse
from pathlib import Path

import numpy as np

from dspages.ingest.embstore import merge_rows  # noqa: F401  (the table and its merge live in the package)


if __name__ == "__main__":
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("backup")
    ap.add_argument("--drop")
    a = ap.parse_args()
    drop = None
    if a.drop:
        R = pd.read_parquet(a.drop)
        drop = (R.doc.astype(np.int64) * 100000 + R.page).values
    gone, added, n = merge_rows(np.load(Path(a.dir) / "embeddings.npy"), np.load(Path(a.dir) / "ids.npy"), a.backup, drop)
    print(f"removed {gone} old rows, added {added} new: {n} rows")
