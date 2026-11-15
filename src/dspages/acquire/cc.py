"""Common Crawl pdf candidates for the open points of a ds_targets run (out/targets.npz: the gap targets, their kernel vectors). Every url of the
local CC pdf index is sampled; the sampled pdfs get a description from the head of their capture (descr.py), are placed in the ds_head space by the
url model (link + description, urlmodel.py) and ranked by the distance to the nearest open point; per
point the PER_POINT nearest urls stay. A candidate is only a guess: place.py keeps a page only when it lands on an open point.
-> WORK/points.npz (open points: id, vector), WORK/sample.parquet, WORK/descr/ (descr.py), WORK/cands.parquet (the kept urls)."""

import glob
import json
import sqlite3

import duckdb
import numpy as np
import pandas as pd
import torch

from . import descr, urlmodel
from .config import PER_POINT  # noqa: F401
from .paths import CC_INDEX, LAYOUTS, MANIFEST_DB, WORK


def source_urls():
    """{document hash: crawl url} of the corpus."""
    with sqlite3.connect(f"file:{MANIFEST_DB}?mode=ro", uri=True) as db:
        u = dict(db.execute("select hash, url from src_urls"))
    for r in map(json.loads, open(LAYOUTS / "archive" / "info" / "untyped_sources.jsonl")):
        if r.get("url"):
            u.setdefault(r["file"][:12], r["url"])
    return u


def train(Z, paths, r=None):
    urls, texts, Y, hosts = urlmodel.corpus(source_urls, Z, paths, descr.corpus(paths))
    print(f"{len(urls)} documents with a url and a description", flush=True)
    urlmodel.train(urls, texts, Y, hosts, r=r)


def points(run):
    """The open points of a ds_targets run: ids and kernel vectors."""
    T = np.load(run / "out" / "targets.npz")
    return T["target"], T["Xt"].astype(np.float32)


def sample(n, seed=0):
    """N index rows spread evenly over the index files (url not in the corpus) -> WORK/sample.parquet."""
    files = sorted(glob.glob(str(CC_INDEX / "*" / "*.parquet")))
    per = max(n // len(files), 1)
    known = {u.split("://", 1)[-1].lower().rstrip("/").removeprefix("www.") for u in source_urls().values()}
    con = duckdb.connect()
    S = pd.concat(
        [
            con.execute(
                f"select url, warc_filename f, warc_record_offset o, warc_record_length l from read_parquet('{f}') "
                f"using sample {per} rows (reservoir, {seed})"
            ).df()
            for f in files
        ],
        ignore_index=True,
    )
    S = S[~S.url.str.split("://", n=1).str[-1].str.lower().str.rstrip("/").str.removeprefix("www.").isin(known)]
    WORK.mkdir(parents=True, exist_ok=True)
    S.to_parquet(WORK / "sample.parquet")
    print(f"{len(S)} index rows sampled from {len(files)} files", flush=True)


def build(run):
    """Descriptions of the sampled rows (descr.heads), then every row with a description is placed by link + description and ranked by its
    distance to the nearest open point; PER_POINT per point -> WORK/cands.parquet."""
    ids, V = points(run)
    np.savez(WORK / "points.npz", point=ids, V=V)
    S = pd.read_parquet(WORK / "sample.parquet")
    D = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(WORK / "descr" / "part_*.parquet")))])
    S = S.merge(D[D.text.str.len() >= 30], on="url")
    net = urlmodel.load()
    P = np.concatenate(
        [
            urlmodel.predict(net, S.url[a : a + 4096].tolist(), S.text[a : a + 4096].tolist())
            for a in range(0, len(S), 4096)
        ]
    )
    dd, jj = [], []
    for a in range(0, len(P), 4096):
        m = torch.cdist(torch.from_numpy(P[a : a + 4096]), torch.from_numpy(V)).min(1)
        dd.append(m.values.numpy())
        jj.append(m.indices.numpy())
    S = S.assign(d=np.concatenate(dd), point=ids[np.concatenate(jj)])
    C = S.sort_values("d").groupby("point").head(PER_POINT)
    C.drop(columns="text").to_parquet(WORK / "cands.parquet")
    print(
        f"candidate urls {len(C)} for {C.point.nunique()} of {len(ids)} points -> {WORK / 'cands.parquet'}", flush=True
    )
