"""100K page set spread evenly over the page-embedding space, and what it lacks per language and type. Data: Layouts/archive/info/ds100k (OUT).

  pool     every embedded page (emb_out, 1536-d) with fill / pic (scan_pct), text / hw boxes (hw_det), language (manifest plan); kept:
           fill > FILL, pic < PIC, not warped (cleaned trace), not in an out-of-scope folder; no page classes
           -> OUT/pool.parquet, OUT/Z.npy (PCA of the unit page vectors, float32)
  select   farthest-point sampling on Z: every next page is the pool page farthest from the set, so the set is evenly spaced and every pool
           page is within the final radius of a set page -> OUT/select.parquet (order = pick order, r = distance when picked)

python tools/ds100k.py pool | append | select [--n 100000] (on Z, local) | gaps [--thin 2] [--regions 20]   (the 100K set on 1536-d: tools/kaggle_fps.py)"""
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
TESTS = Path(PATHS.uvtm) / "tests" / "ds100k"   # test runs and experiments, not persistent data
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


def filtered():
    """Every embedded page of the pool rules (fill / pic / not warped / in scope): row, doc, page, path, folder, h, lang, scan, fill, pic, text, hw."""
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
    return P[keep & ~warped & ~out].sort_values("row").reset_index(drop=True)


def cmd_append(_):
    """Pages embedded since the pool was made (not in pool.parquet by file hash + page, same rules) appended: pool.parquet rows, Z.npy rows (the
    saved PCA) and one new part pool_<k>.npy of the Kaggle pool vectors; every index of the pool stays. Run tools/kaggle_fps.py upload-part after."""
    import kaggle_fps as KF
    old = pd.read_parquet(OUT / "pool.parquet")
    P = filtered()
    new = P[~pd.Series(list(zip(P.h, P.page))).isin(set(zip(old.h, old.page))).to_numpy()].reset_index(drop=True)
    print("new pool pages:", len(new), "of", new.doc.nunique(), "documents; scans", int(new.scan.sum()))
    E = np.load(EMB / "embeddings.npy", mmap_mode="r")
    V = np.concatenate([DD.unit_rows(np.asarray(E[new.row.to_numpy()[a:a + CHUNK]], np.float32)) for a in range(0, len(new), CHUNK)])
    z = np.load(OUT / "pca.npz")
    k = len(list((KF.WORK / "data").glob("pool_*.npy")))
    np.save(KF.WORK / "data" / f"pool_{k}.npy", V.astype(np.float16))
    np.save(OUT / "Z.npy", np.concatenate([np.load(OUT / "Z.npy"), ((V - z["mean"]) @ z["components"].T).astype(np.float32)]))
    pd.concat([old, new], ignore_index=True).to_parquet(OUT / "pool.parquet")
    print("pool now", len(old) + len(new), "-> part", k)


def cmd_pool(_):
    from sklearn.decomposition import PCA
    P = filtered()
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


def fps(Z, n, seed=0, cells=1000):
    """Farthest-point sampling: (picked indices in order, distance of each pick to the set before it). Exact; the pool is split into k-means
    cells and a new pick updates only the cells it can reach (triangle inequality: |center - pick| - cell radius < the cell's largest distance)."""
    import torch
    from sklearn.cluster import MiniBatchKMeans
    torch.set_num_threads(6)
    km = MiniBatchKMeans(cells, random_state=0, batch_size=8192, n_init=1, max_iter=20).fit(Z[np.random.default_rng(0).choice(len(Z), min(len(Z), 200_000), replace=False)])
    lab = km.predict(Z)
    order = np.argsort(lab, kind="stable")
    X = torch.from_numpy(Z[order])
    C = torch.from_numpy(km.cluster_centers_.astype(np.float32))
    start = np.searchsorted(lab[order], np.arange(cells + 1))
    cell = torch.from_numpy(lab[order].astype(np.int64))
    R = torch.zeros(cells).index_reduce_(0, cell, (X - C[cell]).norm(dim=1), "amax", include_self=False)
    first = int(np.random.default_rng(seed).integers(len(Z)))
    sq = (X ** 2).sum(1)                                                # |p - x|^2 = |p|^2 - 2 p.x + |x|^2: one pass over the pages, no temporary
    x0 = torch.from_numpy(Z[first])
    d = (sq - 2 * (X @ x0) + float((x0 ** 2).sum())).clamp_(min=0)
    dmax = torch.zeros(cells).index_reduce_(0, cell, d, "amax", include_self=False)
    pick, rad = [first], [float("inf")]
    inv = np.empty(len(Z), np.int64)
    inv[order] = np.arange(len(Z))
    t0 = time.time()
    for i in range(1, n):
        c = int(torch.argmax(dmax))
        j = start[c] + int(torch.argmax(d[start[c]:start[c + 1]]))
        pick.append(int(order[j]))
        rad.append(float(d[j]) ** 0.5)
        x, xx = X[j], float(sq[j])
        hit = torch.nonzero((C - x).norm(dim=1) - R < dmax.sqrt()).flatten().tolist()
        if len(hit) > cells // 3:                                       # most cells reachable: one pass over all pages
            torch.minimum(d, (sq - 2 * (X @ x) + xx).clamp_(min=0), out=d)
            dmax = torch.zeros(cells).index_reduce_(0, cell, d, "amax", include_self=False)
        else:
            for c in hit:
                a, b = start[c], start[c + 1]
                seg = d[a:b]
                torch.minimum(seg, (sq[a:b] - 2 * (X[a:b] @ x) + xx).clamp_(min=0), out=seg)
                dmax[c] = seg.max()
        if i % 10000 == 0:
            print(f"{i} picked, radius {rad[-1]:.4f}, cells updated {len(hit)}/{cells}, {int(time.time() - t0)} s", flush=True)
    return np.array(pick), np.array(rad)


def fps_plain(Z, n, seed=0):
    """Reference for fps: every pick updates every pool page."""
    import torch
    X = torch.from_numpy(Z)
    first = int(np.random.default_rng(seed).integers(len(Z)))
    d = ((X - X[first]) ** 2).sum(1)
    pick = [first]
    for _ in range(1, n):
        j = int(torch.argmax(d))
        pick.append(j)
        torch.minimum(d, ((X - X[j]) ** 2).sum(1), out=d)
    return np.array(pick)


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
    """Cells of the set (the pool pages nearest to each set page, cover.npz of tools/kaggle_fps.py, 1536-d); a thin cell has at most --thin
    pool pages: the set holds a page there but the corpus has no alternative. Thin set pages grouped into --regions regions (k-means on the
    1536-d unit vectors): centroid = the expected embedding of what is needed -> OUT/cells.parquet, thin.parquet, regions.parquet, regions.npz."""
    from sklearn.cluster import KMeans
    S, cv = pd.read_parquet(OUT / "select.parquet"), np.load(OUT / "cover.npz")
    S["cell_n"] = np.bincount(cv["cell"], minlength=len(S))
    S["cell_r"] = pd.Series(cv["d"]).groupby(cv["cell"]).max().reindex(range(len(S)), fill_value=0).values
    S.to_parquet(OUT / "cells.parquet")
    print("pool pages per set cell:", S.cell_n.describe().round(1).to_dict())
    T = S[S.cell_n <= a.thin].copy()
    print(len(T), f"thin set pages (<= {a.thin} pool pages in the cell);", "languages:", T.lang.value_counts().head(12).to_dict())
    E = np.load(EMB / "embeddings.npy", mmap_mode="r")
    o = np.argsort(T.row.values)
    U = np.empty((len(T), E.shape[1]), np.float32)
    U[o] = DD.unit_rows(np.asarray(E[T.row.values[o]], np.float32))
    km = KMeans(a.regions, random_state=0, n_init=4).fit(U)
    T = T.assign(region=km.labels_, d_cen=np.linalg.norm(U - km.cluster_centers_[km.labels_], axis=1))
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
    sp.add_parser("append")
    s = sp.add_parser("select")
    s.add_argument("--n", type=int, default=100_000)
    g = sp.add_parser("gaps")
    g.add_argument("--thin", type=int, default=2)
    g.add_argument("--regions", type=int, default=20)
    a = ap.parse_args()
    {"pool": cmd_pool, "append": cmd_append, "select": cmd_select, "gaps": cmd_gaps}[a.cmd](a)
