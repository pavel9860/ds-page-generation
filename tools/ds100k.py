"""100K page set spread evenly over the page-embedding space, and what it lacks per language and type. Data: Layouts/archive/info/ds100k (OUT).

  pool     every embedded page (emb_out, 1536-d) with fill / pic (scan_pct), text / hw boxes (hw_det), language (manifest plan); kept:
           fill > FILL, pic < PIC, not warped (cleaned trace), not in an out-of-scope folder; no page classes
           -> OUT/pool.parquet, OUT/Z.npy (PCA of the unit page vectors, float32)
  select   farthest-point sampling on Z: every next page is the pool page farthest from the set, so the set is evenly spaced and every pool
           page is within the final radius of a set page -> OUT/select.parquet (order = pick order, r = distance when picked)

python tools/ds100k.py pool | select [--n 100000] | gaps [--thin 2] [--regions 20]"""
import argparse
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

import diverse_ds as DD
from dspages.hand import corpus
from dspages.hand.config import PATHS

LAYOUTS = Path(PATHS.layouts)
EMB = Path(PATHS.uvtm) / "Labels" / "embedding" / "emb_out"
OUT = LAYOUTS / "archive" / "info" / "ds100k"
FILL, PIC = 0.6, 0.3
DIM, CHUNK = 128, 100_000


OUT_FOLDERS = ("maps", "maps_blueprints", "blueprints", "pages_photo", "receipts", "images")     # clean folders of out-of-scope pages


def docs():
    """One row per embedded document: doc, path, h, folder, lang (manifest plan, else the file name), scan."""
    dt = pd.DataFrame([json.loads(l) for l in open(EMB / "manifest.jsonl")]).rename(columns={"id": "doc"})
    dt["folder"] = dt.path.map(lambda p: "/".join(p.split("/")[:2]) if p.startswith("archive/") else p.split("/")[0])
    dt["h"] = dt.path.map(corpus.doc_id)
    with sqlite3.connect(f"file:{PATHS.manifest}?mode=ro", uri=True) as db:
        plan = pd.read_sql("select doc as h, lang, cls from plan", db).drop_duplicates("h").set_index("h")
    dt = dt.join(plan, on="h")
    dt["lang"] = dt.lang.fillna(dt.path.str.extract(r"_([a-z]{2,3})(?:_p\d+)?\.\w+$")[0]).fillna("?")
    dt["scan"] = ~dt.path.str.lower().str.endswith(".pdf") | (dt.cls == "scan-like") | dt.cls.isna()
    return dt.drop(columns="cls")


def cmd_pool(_):
    from sklearn.decomposition import PCA
    ids = np.load(EMB / "ids.npy")
    P = pd.DataFrame({"row": np.arange(len(ids)), "doc": ids[:, 0], "page": ids[:, 1]}).merge(docs(), on="doc")
    s = pd.read_parquet(LAYOUTS / "archive" / "info" / "scan_pct.parquet", columns=["doc", "page", "fill", "pic"])
    with sqlite3.connect(f"file:{PATHS.manifest}?mode=ro", uri=True) as db:
        hw = pd.read_sql("select doc, page, text, hw from hw_det", db)
    P = P.merge(s, on=["doc", "page"], how="left").merge(hw, on=["doc", "page"], how="left")
    n = len(P)
    P = P.drop_duplicates(["path", "page"])                             # one file under several document ids: its pages once
    print(f"pages {n}, {n - len(P)} repeated (one file under several ids)")
    wp, wd, _ = DD.cleaned()
    warped = P.h.isin(wd) | pd.Series([k in wp for k in zip(P.h, P.page)], index=P.index)
    keep = (P.fill > FILL) & (P.pic < PIC)
    out = P.folder.isin(OUT_FOLDERS) | P.path.str.startswith(("archive/rejected_distorted", "_"))
    print(f"pages {len(P)}; fill > {FILL} and pic < {PIC}: {int(keep.sum())}; then warped out {int((keep & warped).sum())},"
          f" out-of-scope folders {int((keep & ~warped & out).sum())}")
    P = P[keep & ~warped & ~out].sort_values("row").reset_index(drop=True)
    print("pool:", len(P), "pages of", P.doc.nunique(), "documents,", P.lang.nunique(), "languages")
    E = np.load(EMB / "embeddings.npy", mmap_mode="r")
    rows = P.row.to_numpy()
    pca = PCA(DIM, random_state=0).fit(DD.unit_rows(np.asarray(E[np.sort(np.random.default_rng(0).choice(rows, 200_000, replace=False))], np.float32)))
    print(f"PCA {DIM}: explained variance {pca.explained_variance_ratio_.sum():.3f}")
    Z = np.concatenate([pca.transform(DD.unit_rows(np.asarray(E[rows[a:a + CHUNK]], np.float32))) for a in range(0, len(rows), CHUNK)])
    OUT.mkdir(parents=True, exist_ok=True)
    P.to_parquet(OUT / "pool.parquet")
    np.save(OUT / "Z.npy", Z.astype(np.float32))
    np.savez(OUT / "pca.npz", mean=pca.mean_, components=pca.components_)
    print("->", OUT)


def fps(Z, n, seed=0):
    """Farthest-point sampling: (picked indices in order, distance of each pick to the set before it)."""
    import torch
    torch.set_num_threads(6)
    X = torch.from_numpy(Z)
    first = int(np.random.default_rng(seed).integers(len(Z)))
    d = ((X - X[first]) ** 2).sum(1)
    pick, rad = [first], [float("inf")]
    t0 = time.time()
    for i in range(1, n):
        j = int(torch.argmax(d))
        pick.append(j)
        rad.append(float(d[j]) ** 0.5)
        torch.minimum(d, ((X - X[j]) ** 2).sum(1), out=d)
        if i % 10000 == 0:
            print(f"{i} picked, radius {rad[-1]:.4f}, {int(time.time() - t0)} s", flush=True)
    return np.array(pick), np.array(rad)


def cmd_select(a):
    P = pd.read_parquet(OUT / "pool.parquet")
    pick, rad = fps(np.load(OUT / "Z.npy"), a.n)
    S = P.iloc[pick].assign(order=np.arange(len(pick)), r=rad).reset_index(names="pool_i")
    S.to_parquet(OUT / "select.parquet")
    print(len(S), "pages of", S.doc.nunique(), "documents; final radius", round(float(rad[-1]), 4), "->", OUT / "select.parquet")


def nearest(Z, C, block=20_000):
    """(index of the nearest row of C, distance) for every row of Z."""
    import torch
    torch.set_num_threads(6)
    Ct = torch.from_numpy(C)
    cn = (Ct ** 2).sum(1)
    idx, dist = [], []
    for a in range(0, len(Z), block):
        z = torch.from_numpy(Z[a:a + block])
        d = (z ** 2).sum(1, keepdim=True) - 2 * z @ Ct.T + cn
        m, i = d.min(1)
        idx.append(i.numpy())
        dist.append(m.clamp(min=0).sqrt().numpy())
    return np.concatenate(idx), np.concatenate(dist)


def cmd_gaps(a):
    """Cells of the set (the pool pages nearest to each set page); a thin cell has at most --thin pool pages: the set holds a page there
    but the corpus has no alternative. Thin set pages grouped into --regions regions (k-means on Z): centroid = the expected embedding
    of what is needed -> OUT/cells.parquet, OUT/regions.parquet, OUT/regions.npz."""
    from sklearn.cluster import KMeans
    P, S, Z = pd.read_parquet(OUT / "pool.parquet"), pd.read_parquet(OUT / "select.parquet"), np.load(OUT / "Z.npy")
    cell, d = nearest(Z, Z[S.pool_i.values])
    S["cell_n"] = np.bincount(cell, minlength=len(S))
    S["cell_r"] = pd.Series(d).groupby(cell).max().reindex(range(len(S)), fill_value=0).values
    S.to_parquet(OUT / "cells.parquet")
    print("pool pages per set cell:", S.cell_n.describe().round(1).to_dict())
    T = S[S.cell_n <= a.thin]
    print(len(T), f"thin set pages (<= {a.thin} pool pages in the cell);", "languages:", T.lang.value_counts().head(12).to_dict())
    km = KMeans(a.regions, random_state=0, n_init=4).fit(Z[T.pool_i.values])
    T = T.assign(region=km.labels_, d_cen=np.linalg.norm(Z[T.pool_i.values] - km.cluster_centers_[km.labels_], axis=1))
    R = T.groupby("region").agg(pages=("doc", "size"), docs=("doc", "nunique"), radius=("d_cen", "median"),
                                langs=("lang", lambda s: ", ".join(f"{k} {v}" for k, v in s.value_counts().head(3).items())),
                                folders=("folder", lambda s: ", ".join(f"{k} {v}" for k, v in s.value_counts().head(3).items())),
                                examples=("path", lambda s: " ; ".join(s.head(3))))
    T.to_parquet(OUT / "thin.parquet")
    R.to_parquet(OUT / "regions.parquet")
    np.savez(OUT / "regions.npz", centers=km.cluster_centers_.astype(np.float32))
    print(R.to_string(max_colwidth=60))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("pool")
    s = sp.add_parser("select")
    s.add_argument("--n", type=int, default=100_000)
    g = sp.add_parser("gaps")
    g.add_argument("--thin", type=int, default=2)
    g.add_argument("--regions", type=int, default=20)
    a = ap.parse_args()
    {"pool": cmd_pool, "select": cmd_select, "gaps": cmd_gaps}[a.cmd](a)
