"""Pages taken out of the corpus tables: embeddings (emb_out), hw_det, scan_pct (+ parquet). One backup `*.bak` of each file (state before)."""

import csv
import os
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from . import embstore
from .paths import EMB, MANIFEST_DB, PARQUET

HEX = re.compile(r"(?<![0-9a-f])[0-9a-f]{12}(?![0-9a-f])")


def candidate_pages(folder):
    """(hash, page) of the queue.csv rows (tools/ds_targets candidate pages) whose jpg is no longer in the folder: dropped by hand."""
    have = set(os.listdir(folder))
    return [(r["h"], int(r["page"])) for r in csv.DictReader(open(Path(folder) / "queue.csv")) if r["name"] not in have]


def drop_pages(pairs):
    """pairs: (name hash, page) -> removed from emb_out, hw_det, scan_pct. -> counts."""
    by = {}
    for d, p in embstore.manifest().items():
        m = HEX.search(p.rsplit("/", 1)[-1])
        if m:
            by.setdefault(m.group(0), []).append(d)
    rows = {(d, pg) for h, pg in pairs for d in by.get(h, [])}
    key = {d * 100000 + pg for d, pg in rows}
    ids = np.load(EMB / "ids.npy")
    present = {k for k in embstore.keys(ids).tolist() if k in key}
    out = {"pages": len(rows), "emb rows": len(present)}
    if present:
        embstore.merge_rows(np.zeros((0, embstore.DIM), np.float16), np.zeros((0, 2), np.int32), drop=present)
    embstore.backup((MANIFEST_DB.name,), MANIFEST_DB.parent)
    with sqlite3.connect(MANIFEST_DB, timeout=600) as m:
        out["hw_det"] = m.executemany("delete from hw_det where doc = ? and page = ?", sorted(rows)).rowcount
        out["scan_pct"] = m.executemany("delete from scan_pct where hash = ? and page = ?", sorted(set(pairs))).rowcount
    if PARQUET.exists():
        embstore.backup((PARQUET.name,), PARQUET.parent)
        P = pd.read_parquet(PARQUET)
        gone = (P.doc.astype(np.int64) * 100000 + P.page).isin(key)
        out["parquet"] = int(gone.sum())
        P[~gone].to_parquet(PARQUET)
    return out
