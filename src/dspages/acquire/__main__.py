"""Two ways to fill the open points of a ds_targets run RUN (gap targets, scans to replace) with real pages:

  lens   queries RUN [--n N] | seeds DIR.. | serve | worker | agent      Google Lens visual search (hw / filled pages), driven by a Haiku agent in the in-app browser
  cc     train | sample | descr | cands RUN | fetch | place RUN   Common Crawl pdfs: link + description (first pages' text) -> ds_head space, ranked by the open points; download, placement
  status                                                   counts of the downloads

Both end in files for dspages.ingest (acquired/pdf/<sha12>.pdf, acquired/lens/got/<class>/<sha12>.jpg).
Run long steps inside a memory-capped cgroup, so a leak kills the step and never the machine:
  systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 .venv/bin/python -m dspages.acquire cc fetch"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import cc, descr, fetch, lens, place
from .paths import LAYOUTS, WORK

AUX = LAYOUTS / "archive" / "info" / "ds100k"


def space():
    """Vectors and (path, page) of the in-scope pages (ds_targets aux)."""
    z = np.load(AUX / "targets" / "aux" / "Z.npz")
    P = pd.read_parquet(AUX / "pool.parquet", columns=["path", "page"]).iloc[z["pool_i"]]
    return z["Z"].astype(np.float32), list(zip(P.path, P.page))


def main():
    ap = argparse.ArgumentParser(prog="dspages.acquire")
    sp = ap.add_subparsers(dest="cmd", required=True)
    L = sp.add_parser("lens").add_subparsers(dest="step", required=True)
    q = L.add_parser("queries")
    q.add_argument("run", type=Path, nargs="?")
    q.add_argument("--n", type=int, default=3000, help="without RUN: query pages (half hw, half filled)")
    L.add_parser("seeds").add_argument("dirs", type=Path, nargs="+")
    for n in ("serve", "worker", "agent"):
        L.add_parser(n)
    C = sp.add_parser("cc").add_subparsers(dest="step", required=True)
    C.add_parser("train").add_argument("--r", type=float, default=None, help="radius for the held-out hit rate")
    C.add_parser("sample").add_argument("--n", type=int, default=300_000, help="index rows")
    C.add_parser("descr")
    C.add_parser("cands").add_argument("run", type=Path)
    f = C.add_parser("fetch")
    f.add_argument("--workers", type=int, default=6, help="check processes")
    f.add_argument("--limit", type=int, default=0)
    f.add_argument("--fetchers", type=int, default=768, help="live download threads")
    C.add_parser("place").add_argument("run", type=Path)
    sp.add_parser("status")
    a = ap.parse_args()
    if a.cmd == "lens":
        {
            "queries": lambda: lens.queries(a.run, a.n),
            "seeds": lambda: lens.seeds(a.dirs),
            "serve": lens.serve,
            "worker": lens.worker,
            "agent": lens.agent,
        }[a.step]()
    elif a.cmd == "cc":
        if a.step == "train":
            cc.train(*space(), a.r)
        elif a.step == "sample":
            cc.sample(a.n)
        elif a.step == "descr":
            descr.heads(WORK / "sample.parquet")
        elif a.step == "cands":
            cc.build(a.run)
        elif a.step == "fetch":
            fetch.run(WORK / "cands.parquet", a.workers, a.limit, a.fetchers)
        else:
            place.run(json.load(open(a.run / "out" / "targets.json"))["r"])
    else:
        fetch.status()


if __name__ == "__main__":
    main()
