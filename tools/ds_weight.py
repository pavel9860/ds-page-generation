"""Factor weights of the page space: the 1536-d unit page vectors rescaled so that each factor of the data's own fields gets as much variance
as the first principal component, the rest of the space as it is. Factors and their directions (fitted on SAMPLE in-scope pages):
  hw     hw_det boxes: none / some (0 < hw < text) / hw-dominant (hw >= text > 0)      LDA, 2 directions
  type   manifest plan type (typed documents only)                                     LDA
  lang   plan language, the NLANG largest                                              LDA
  fonts  manifest fonts: the NFONT most common font families (name before '-'), present or not   mean-difference, one each
  fill, pic   scan_pct                                                                 least-squares direction, one each
Every factor's directions are orthonormalised against the factors before it; scale s_f = sqrt(var(PC1) / variance of the subspace).
x' = x + Q diag(s - 1) Q^T x. Data: ds100k/grid (OUT).

python tools/ds_weight.py fit [--weights hw=1,type=1,...]   -> OUT/kaggle/aux_w/weight.npz (Q, s), OUT/weight.json
python tools/ds_weight.py compare                            factor shares: in-scope pool / plain grid / weighted grid -> OUT/compare.csv"""

import argparse
import json
import sqlite3

import ds100k as D
import ds_border as B
import ds_grid as G
import numpy as np
import pandas as pd
import torch

from dspages.hand.config import PATHS

SAMPLE, NLANG, NFONT = 200_000, 30, 30
FACTORS = ("hw", "type", "lang", "fonts", "fill", "pic")


def labels(P):
    with sqlite3.connect(f"file:{PATHS.manifest}?mode=ro", uri=True) as db:
        plan = pd.read_sql("select doc as h, type from plan", db).drop_duplicates("h").set_index("h").type
        fonts = pd.read_sql("select doc as h, font from fonts", db)
    L = pd.DataFrame(index=P.index)
    L["hw"] = np.select([(P.hw > 0) & (P.hw >= P.text), P.hw > 0], ["dominant", "some"], "none")
    L["type"] = P.h.map(plan).where(lambda t: t.notna() & (t != "_untyped"))
    top = P.lang.value_counts().index[:NLANG]
    L["lang"] = P.lang.where(P.lang.isin(top))
    fonts["fam"] = fonts.font.str.split("-").str[0].str.lower()
    fonts = fonts.drop_duplicates(["h", "fam"])
    fam = fonts.fam.value_counts().index[:NFONT]
    F = pd.crosstab(fonts.h[fonts.fam.isin(fam)], fonts.fam[fonts.fam.isin(fam)]).clip(upper=1)
    has = P.h.isin(fonts.h.unique()).to_numpy()
    Fm = F.reindex(P.h).fillna(0).to_numpy()
    return L, Fm, list(F.columns), has


def lda(X, y, k):
    """Directions of the largest between / within class variance (within shrunk), at most k."""
    cls = [c for c in pd.unique(y) if (y == c).sum() >= 50]
    mu = X.mean(0)
    Sw = torch.zeros(X.shape[1], X.shape[1])
    M = []
    for c in cls:
        Xc = X[torch.from_numpy(y == c)]
        mc = Xc.mean(0)
        Sw += (Xc - mc).T @ (Xc - mc)
        M.append((mc - mu) * len(Xc) ** 0.5)
    Sw = Sw / len(X)
    Sw += 0.1 * torch.trace(Sw) / Sw.shape[0] * torch.eye(Sw.shape[0])
    Mb = torch.stack(M).T
    Li = torch.linalg.cholesky(Sw)
    A = torch.linalg.solve_triangular(Li, Mb, upper=False)
    U, _, _ = torch.linalg.svd(A, full_matrices=False)
    W = torch.linalg.solve_triangular(Li.T, U[:, : min(k, len(cls) - 1)], upper=True)
    return W, len(cls)


def cmd_fit(a):
    torch.set_num_threads(6)
    w = {f: 1.0 for f in FACTORS}
    if a.weights:
        w.update({k: float(v) for k, v in (kv.split("=") for kv in a.weights.split(","))})
    P = pd.read_parquet(D.OUT / "pool.parquet", columns=["h", "lang", "fill", "pic", "hw", "text"])
    m = np.load(G.WORK / "aux" / "mask.npy")
    L, Fm, fams, has = labels(P)
    idx = np.sort(np.random.default_rng(0).choice(np.flatnonzero(m), SAMPLE, replace=False))
    X = torch.from_numpy(B.pool_X()[idx]).float()
    Xc = X - X.mean(0)
    C = Xc.T @ Xc / len(X)
    v1 = float(torch.linalg.eigvalsh(C)[-1])
    dirs = {}
    for f, k in (("hw", 2), ("type", 30), ("lang", NLANG)):
        y = L[f].to_numpy()[idx]
        ok = pd.notna(y)
        dirs[f], ncls = lda(X[torch.from_numpy(ok)], y[ok].astype(str), k)
        print(f"{f}: {int(ok.sum())} pages, {ncls} classes, {dirs[f].shape[1]} directions", flush=True)
    hs = has[idx]
    Fs = torch.from_numpy(Fm[idx][hs]).float()
    Xf = X[torch.from_numpy(hs)]
    dirs["fonts"] = torch.stack([Xf[Fs[:, j] > 0].mean(0) - Xf[Fs[:, j] == 0].mean(0) for j in range(Fs.shape[1])], 1)
    print(f"fonts: {int(hs.sum())} pages with fonts, {len(fams)} families", flush=True)
    for f in ("fill", "pic"):
        t = torch.from_numpy(P[f].to_numpy()[idx]).float()
        dirs[f] = torch.linalg.solve(C + 1e-3 * torch.eye(C.shape[0]), Xc.T @ (t - t.mean()) / len(X))[:, None]
    Qs, S, rep = [], [], {}
    for f in FACTORS:
        Wf = dirs[f]
        if Qs:
            Qp = torch.cat(Qs, 1)
            Wf = Wf - Qp @ (Qp.T @ Wf)
        Qf, _ = torch.linalg.qr(Wf)
        var = float(torch.trace(Qf.T @ C @ Qf))
        s = (w[f] * v1 / var) ** 0.5
        Qs.append(Qf)
        S += [s] * Qf.shape[1]
        rep[f] = dict(dims=Qf.shape[1], var_share=round(var / float(torch.trace(C)), 4), scale=round(s, 2), weight=w[f])
        print(f"{f}: {Qf.shape[1]} dims, variance share {var / float(torch.trace(C)):.4f}, scale {s:.2f}", flush=True)
    Q = torch.cat(Qs, 1)
    d = G.WORK / "aux_w"
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "weight.npz", Q=Q.numpy().astype(np.float32), s=np.array(S, np.float32))
    (G.OUT / "weight.json").write_text(json.dumps(dict(factors=rep, fonts=fams, v1=v1), indent=1))
    Xw = X + ((X @ Q) * (torch.tensor(S) - 1)) @ Q.T
    Cw = torch.cov(Xw.T)
    print(
        f"total variance x {float(torch.trace(Cw) / torch.trace(C)):.2f}; factor subspace share after: {float(torch.trace(Q.T @ Cw @ Q) / torch.trace(Cw)):.3f}"
    )
    (d / "dataset-metadata.json").write_text(
        json.dumps({"title": G.AUXW.split("/")[1], "id": G.AUXW, "licenses": [{"name": "CC-BY-NC-SA-4.0"}]})
    )
    print(G.KF.kaggle("datasets", "create", "-p", str(d))[-200:])


def cmd_compare(_):
    P = pd.read_parquet(D.OUT / "pool.parquet", columns=["h", "lang", "fill", "pic", "hw", "text"])
    m = np.load(G.WORK / "aux" / "mask.npy")
    L, Fm, fams, has = labels(P)
    L["fill"] = pd.cut(P.fill, [0.7, 0.8, 0.9, 1.01], right=False).astype(str)
    L["pic"] = pd.cut(P.pic, [0, 0.05, 0.1, 0.2], right=False).astype(str)
    L["fonts"] = np.where(has, "has fonts", "no fonts")
    sets = {
        "pool": np.flatnonzero(m),
        "grid": pd.read_parquet(G.OUT / "grid.parquet").nn.to_numpy(),
        "grid_w": pd.read_parquet(G.OUT / "grid_w.parquet").nn.to_numpy(),
    }
    rows = []
    for f in FACTORS:
        for k, ix in sets.items():
            for v, c in L[f].iloc[ix].value_counts(normalize=True, dropna=False).items():
                rows.append((f, str(v), k, c))
    for j, fam in enumerate(fams):
        for k, ix in sets.items():
            rows.append(("font family", fam, k, Fm[ix, j].mean()))
    T = (
        pd.DataFrame(rows, columns=["factor", "value", "set", "share"])
        .pivot_table(index=["factor", "value"], columns="set", values="share")
        .fillna(0)
    )
    T = T[["pool", "grid", "grid_w"]]
    T["w/pool"] = T.grid_w / T.pool.replace(0, np.nan)
    T.to_csv(G.OUT / "compare.csv")
    pd.set_option("display.width", 200)
    for f in ("hw", "type", "fill", "pic", "fonts"):
        print(T.loc[f].sort_values("pool", ascending=False).round(4).to_string(), "\n")
    for f in ("lang", "font family"):
        t = T.loc[f]
        print(
            f,
            "values",
            len(t),
            "| entropy-normalised spread (1 = equal): pool {:.3f} grid {:.3f} grid_w {:.3f}".format(
                *tuple(
                    (-(t[c][t[c] > 0] * np.log(t[c][t[c] > 0])).sum() / np.log(len(t)))
                    for c in ("pool", "grid", "grid_w")
                )
            ),
        )
        print(t.sort_values("pool", ascending=False).head(8).round(4).to_string(), "\n")
    print("->", G.OUT / "compare.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("fit").add_argument("--weights", default="")
    sp.add_parser("compare")
    a = ap.parse_args()
    {"fit": cmd_fit, "compare": cmd_compare}[a.cmd](a)
