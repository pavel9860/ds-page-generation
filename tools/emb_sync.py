"""Embedding rows and manifest page counts against the files as they are now (pages deleted from a file, files gone).

  python tools/emb_sync.py check   # actual pages of every embedded document -> WORK/docs.parquet, counts of every kind of mismatch

Whether the stored vectors are those of the pages they are filed under: tools/emb_check.py (a random sample embedded again on the Kaggle GPUs).
Data goes to Layouts/archive/info/emb_sync (WORK). 2 workers for the page counts (the file's xref, no rendering)."""

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pymupdf

from dspages import paths as _P
from dspages.hand.config import PATHS

sys.path.insert(0, f"{_P.CORPUS}")
from pagetype.layouts import Layouts, load_manifest  # noqa: E402

LAYOUTS = Path(PATHS.layouts)
EMB = Path(PATHS.uvtm) / "emb_out"
WORK = LAYOUTS / "archive" / "info" / "emb_sync"
pymupdf.TOOLS.mupdf_display_errors(False)


def actual_pages(path):
    """Pages of the file now: -1 unreadable, 1 for an image."""
    if not path.lower().endswith(".pdf"):
        return 1
    try:
        with pymupdf.open(path) as d:
            return d.page_count
    except Exception:
        return -1


def cmd_check(_):
    WORK.mkdir(parents=True, exist_ok=True)
    ids = np.load(EMB / "ids.npy")
    df = pd.DataFrame({"doc": ids[:, 0], "page": ids[:, 1]})
    g = df.groupby("doc").page.agg(emb="size", pmax="max", pmin="min")
    lay = Layouts(LAYOUTS, load_manifest(EMB / "manifest.jsonl"))
    t0 = time.time()
    paths = [lay.path(int(d)) for d in g.index]
    print("paths", int(time.time() - t0), "s", flush=True)
    with ProcessPoolExecutor(2) as ex:
        actual = list(ex.map(actual_pages, [str(LAYOUTS / p) if p else "" for p in paths], chunksize=500))
    print("page counts", int(time.time() - t0), "s", flush=True)
    rows = [(int(d), p, a if p else None, lay.pages(int(d))) for d, p, a in zip(g.index, paths, actual)]
    D = g.reset_index().merge(pd.DataFrame(rows, columns=["doc", "path", "actual", "manifest_pages"]), on="doc")
    D.to_parquet(WORK / "docs.parquet")
    img = ~D.path.fillna("").str.lower().str.endswith(".pdf")
    has = D.path.notna() & (D.actual >= 0)
    phantom = has & (D.pmax >= D.actual)
    stale = has & D.manifest_pages.notna() & (D.manifest_pages != D.actual)
    print(f"{len(D)} embedded documents, {len(df)} rows")
    print("no file now:", int(D.path.isna().sum()), "| unreadable:", int((D.actual < 0).sum()))
    print(
        "rows of pages beyond the file's end (phantom):",
        int(df.merge(D[phantom][["doc", "actual"]], on="doc").pipe(lambda x: (x.page >= x.actual).sum())),
        "in",
        int(phantom.sum()),
        "documents (images:",
        int((phantom & img).sum()),
        ")",
    )
    print(
        "manifest page count != actual:",
        int(stale.sum()),
        "(fewer pages now:",
        int((stale & (D.actual < D.manifest_pages)).sum()),
        ", more:",
        int((stale & (D.actual > D.manifest_pages)).sum()),
        ") | manifest has no count:",
        int((has & D.manifest_pages.isna()).sum()),
    )
    gone = stale & (D.actual < D.manifest_pages)
    print(
        "pages deleted from the file (fewer than the manifest says) with phantom rows:",
        int((gone & phantom).sum()),
        "| without (embedded pages still inside):",
        int((gone & ~phantom).sum()),
        "<- their page numbers may have shifted",
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check").set_defaults(f=cmd_check)
    a = ap.parse_args()
    a.f(a)
