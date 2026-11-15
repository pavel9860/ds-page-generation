"""Training data (gen), feature search (study) and the final models (train)."""

import json
import pickle
import re
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CONFIG
from .detector import Detector
from .features import chosen, derive
from .labels import sample
from .maps import map_groups
from .paths import LAYOUTS, MANIFEST_DB, MODEL, PARQUET, WORK, doc_paths


def sample_jobs(n, cfg=CONFIG):
    """[(path, page, seed)] of n digital pages of the embedding table (pdf, plan class not scan-like) over its top folders, half of them with a picture (pic of
    scan_pct > 0.02)."""
    D = pd.read_parquet(PARQUET, columns=["hash", "doc", "page", "pic"])
    with sqlite3.connect(f"file:{MANIFEST_DB}?mode=ro", uri=True) as db:
        scan = {h for (h,) in db.execute("select doc from plan where cls = 'scan-like'")}
    D = D[~D.hash.isin(scan) & D["pic"].notna()].assign(path=lambda d: d.doc.map(doc_paths()))
    D = D[D.path.notna() & D.path.str.lower().str.endswith(".pdf")]
    D = D.assign(top=D.path.str.replace(str(LAYOUTS) + "/", "", regex=False).str.split("/").str[0])
    rng = np.random.default_rng(cfg.train.seed)
    per = max(n // D.top.nunique(), 20)
    P = (
        pd.concat(
            [
                pd.concat(
                    [
                        g[g["pic"] > 0.02].sample(min(per // 2, int((g["pic"] > 0.02).sum())), random_state=1),
                        g.sample(min(per // 2, len(g)), random_state=2),
                    ]
                )
                for _, g in D.groupby("top")
            ]
        )
        .drop_duplicates(["path", "page"])
        .sample(frac=1, random_state=1)
        .reset_index(drop=True)
    )
    return [(r.path, int(r.page), int(rng.integers(1 << 30))) for r in P.itertuples()]


def train_pages(name="gen.pkl"):
    """(path, page) of the pages of WORK/<name>: the training data."""
    return {(r["path"], r["page"]) for r in pickle.load(open(WORK / name, "rb"))}


def gen(n=None, limit=0, cfg=CONFIG):
    jobs = sample_jobs(n or cfg.train.n_pages, cfg)[: limit or None]
    t0 = time.time()
    with ProcessPoolExecutor(cfg.train.workers) as ex:
        res = [r for r in ex.map(sample, jobs, chunksize=2) if r]
    WORK.mkdir(parents=True, exist_ok=True)
    pickle.dump(res, open(WORK / ("gen_test.pkl" if limit else "gen.pkl"), "wb"))
    print(f"{len(res)} of {len(jobs)} pages, {int(time.time() - t0)} s")


def load(name="gen.pkl"):
    """The pages of WORK/<name>; pages listed in WORK/text_pages.txt (`<doc hash>_<lang>_<page>`, page `*` = every page of the document) are text pages: their picture
    label is 0 in the tiles with ink, the empty tiles of a picture keep theirs."""
    R = pickle.load(open(WORK / name, "rb"))
    pages, docs = set(), set()
    for line in (ln.strip() for ln in open(WORK / "text_pages.txt")):
        if line:
            h, p = re.search(r"[0-9a-f]{12}", line).group(), line.rsplit("_", 1)[1]
            (docs.add(h) if p == "*" else pages.add((h, int(p))))
    n = 0
    for r in R:
        h = re.search(r"[0-9a-f]{12}", Path(r["path"]).name).group()
        if h in docs or (h, r["page"]) in pages:
            r["picA"], r["picB"], n = np.where(r["inkA"], 0, r["picA"]), np.where(r["inkB"], 0, r["picB"]), n + 1
    print(f"{n} text pages of {len(R)}")
    return R


def split(R, n_eval):
    """Train pages and held-out pages (other documents)."""
    docs = sorted({r["path"] for r in R})
    test_docs = set(docs[:: max(len(docs) // n_eval, 1)][:n_eval])
    return [r for r in R if r["path"] not in test_docs], [r for r in R if r["path"] in test_docs]


def matrices(R, v, cfg=CONFIG):
    """(X, ink, pic) of the pages for the window v ('A' or 'B'), all columns of derive."""
    c = cfg.train.chunk_pages
    X = np.concatenate(
        [
            derive(np.stack([r[f"cells{v}"] for r in R[i : i + c]]), np.stack([r["scal"] for r in R[i : i + c]]), cfg)
            for i in range(0, len(R), c)
        ]
    )
    return X, np.concatenate([r[f"ink{v}"] for r in R]), np.concatenate([r[f"pic{v}"] for r in R])


def fit_models(X, ink, pic, trees, depth, lr, cols, cfg=CONFIG):
    from xgboost import XGBClassifier, XGBRegressor

    t = cfg.train
    kw = dict(
        n_estimators=trees,
        max_depth=depth,
        learning_rate=lr,
        subsample=t.subsample,
        colsample_bytree=t.colsample,
        n_jobs=t.workers,
        tree_method="hist",
        random_state=t.seed,
    )
    return Detector(
        XGBClassifier(**kw).fit(X[:, cols], ink),
        XGBRegressor(objective="reg:logistic", **kw).fit(X[:, cols], pic),
        cols,
        cfg,
    )


def summary(e):
    """Mean abs error, median, p90 (percentage points), share within 5 points and bias of an error array in shares."""
    e = 100 * np.asarray(e)
    return dict(
        mean_abs=np.abs(e).mean(),
        median=np.median(np.abs(e)),
        p90=np.quantile(np.abs(e), 0.9),
        within5=100 * (np.abs(e) <= 5).mean(),
        bias=e.mean(),
    )


def tile_metrics(ink, pic, pi, pp):
    tp, fp, fn = (pi & ink).sum(), (pi & ~ink).sum(), (~pi & ink).sum()
    return dict(
        ink_acc=100 * (pi == ink).mean(),
        ink_prec=100 * tp / max(tp + fp, 1),
        ink_rec=100 * tp / max(tp + fn, 1),
        ink_f1=100 * 2 * tp / max(2 * tp + fp + fn, 1),
        pic_tile_mae=100 * np.abs(pp - pic).mean(),
        pic_tile_r2=1 - ((pp - pic) ** 2).sum() / max(((pic - pic.mean()) ** 2).sum(), 1e-9),
    )


def _pair(R, cfg):
    Xa, ia, pa = matrices(R, "A", cfg)
    Xb, ib, pb = matrices(R, "B", cfg)
    return np.concatenate([Xa, Xb]), np.concatenate([ia, ib]), np.concatenate([pa, pb])


def study(test=False, cfg=CONFIG):
    """Forward search: the base features, then every group (maps, then derived) is added on its own on top of what was kept and stays if the held-out scan-window tiles
    gain (ink error or picture MAE keep_gain times lower, the other not more than keep_loss times worse)."""
    t, f = cfg.train, cfg.features
    tr, te = split(load("gen_test.pkl" if test else "gen.pkl"), t.n_eval_docs)
    print(f"{len(tr)} training pages, {len(te)} held-out pages", flush=True)
    X, ink, pic = _pair(tr, cfg)
    rows = np.random.default_rng(t.seed).choice(len(X), min(t.study_rows, len(X)), replace=False)
    Xt, it, pt = matrices(te, "B", cfg)
    groups = map_groups(cfg)
    maps, derived = set(groups["base"]), set(f.base_derived)
    cand = [("maps", g) for g in groups if g != "base"] + [("derived", d) for d in f.candidate_derived]

    def score(mp, dv):
        cols = chosen(mp, dv, cfg)
        d = fit_models(X[rows], ink[rows], pic[rows], t.study_trees, t.study_depth, t.study_lr, cols, cfg)
        return dict(columns=int(cols.sum()), **tile_metrics(it, pt, *d.tiles(Xt)))

    best = score(maps, derived)
    log = [dict(step="base", kept=True, **best)]
    print("base", {k: round(float(v), 3) for k, v in best.items()}, flush=True)
    for kind, name in cand:
        mp, dv = set(maps), set(derived)
        (mp.update(groups[name]) if kind == "maps" else dv.add(name))
        s = score(mp, dv)
        err, mae = (100 - s["ink_acc"]) / (100 - best["ink_acc"]), s["pic_tile_mae"] / best["pic_tile_mae"]
        keep = min(err, mae) < t.keep_gain and max(err, mae) < t.keep_loss
        print(f"{kind} {name}: kept {keep}", {k: round(float(v), 3) for k, v in s.items()}, flush=True)
        log.append(dict(step=f"{kind} {name}", kept=bool(keep), **s))
        if keep:
            maps, derived, best = mp, dv, s
    json.dump(
        dict(maps=sorted(maps), derived=sorted(derived), log=log),
        open(WORK / "features.json", "w"),
        indent=1,
        default=float,
    )
    print("kept maps", sorted(maps), "derived", sorted(derived))


def train(cfg=CONFIG):
    """The final models with the features of WORK/features.json -> MODEL; metrics on held-out documents, for both windows."""
    t = cfg.train
    feat = json.load(open(WORK / "features.json"))
    cols = chosen(set(feat["maps"]), set(feat["derived"]), cfg)
    tr, te = split(load(), t.n_eval_docs)
    print(
        f"{len(tr)} training pages, {len(te)} held-out pages (other documents), {int(cols.sum())} features", flush=True
    )
    X, ink, pic = _pair(tr, cfg)
    t0 = time.time()
    d = fit_models(X, ink, pic, t.train_trees, t.train_depth, t.train_lr, cols, cfg)
    print(f"trained in {int(time.time() - t0)} s", flush=True)
    d.save(MODEL)
    n = len(te)
    for v, name in (("A", "window of the clean page"), ("B", "window of the scan's own ink (the detector in use)")):
        X, ink, pic = matrices(te, v, cfg)
        pi, pp = d.tiles(X)
        print(f"\n{name}: tiles", {k: round(float(x), 2) for k, x in tile_metrics(ink, pic, pi, pp).items()})
        fill_p, pic_p = pi.reshape(n, -1).mean(1), pp.reshape(n, -1).mean(1)
        rows = [
            ("fill vs label", fill_p - ink.reshape(n, -1).mean(1)),
            ("pic vs label", pic_p - pic.reshape(n, -1).mean(1)),
        ]
        for what, e in rows:
            print(f"  page {what} ({len(e)} pages):", {k: round(float(x), 2) for k, x in summary(e).items()})
    t0 = time.time()
    for _ in range(20):
        d.tiles(X[: cfg.geometry.grid**2])
    print(f"\nmodel time per page {1000 * (time.time() - t0) / 20:.0f} ms (ink + picture, one window)")
