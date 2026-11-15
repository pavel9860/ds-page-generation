"""Pages without a value, why, and a recheck of the stored values from the source files."""

import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .config import CONFIG
from .detector import Detector, fit, jpeg_roundtrip
from .maps import ink
from .paths import EMB, MODEL, PARQUET, PCT, doc_paths
from .render import render, render_other
from .tiles import window
from .util import time_limit

_DET = None


def _init():
    global _DET
    _DET = Detector.load(MODEL)


def _row(r):
    return float(r.fill), float(r.pic_mean), *r.window


def _fill_page(job):
    """(doc, page, path) -> (fill, pic, x, y, w, h) of the page from its source file (poppler / PIL), in the shard format; None if it cannot be read."""
    _, number, path = job
    try:
        with time_limit(CONFIG.kaggle.local_render_seconds):
            g = render_other(path, number)
            return None if g is None else _row(_DET.predict([jpeg_roundtrip(fit(g), CONFIG.geometry.jpeg_quality)])[0])
    except BaseException:
        return None


def _recheck_page(job):
    """The same from the MuPDF render, as the shards were made."""
    _, number, path = job
    try:
        with time_limit(CONFIG.kaggle.local_render_seconds):
            return _row(_DET.predict([jpeg_roundtrip(render(path, number), CONFIG.geometry.jpeg_quality)])[0])
    except BaseException:
        return None


def _why(job):
    """Why a page has no value: 'file missing', 'page beyond an image file', 'render failed', 'blank' (rendered, no ink) or 'ink'."""
    _, number, path = job
    try:
        with time_limit(CONFIG.kaggle.local_render_seconds):
            from pathlib import Path

            if not Path(path).exists():
                return "file missing"
            g = render_other(path, number)
            if g is None:
                return "page beyond an image file"
            return "blank" if window(ink(fit(g)).mask, CONFIG.geometry.infer_margin) is None else "ink"
    except BaseException:
        return "render failed"


def _pool(fn, jobs, workers):
    with ProcessPoolExecutor(workers, initializer=_init) as ex:
        return list(ex.map(fn, jobs, chunksize=4))


def fill_missing(cfg=CONFIG):
    """Embedded pages without a value in PARQUET, from their source files -> PCT/out/local/scan_pct_local.npz (earlier results stay; the merge takes a value over a NaN)."""
    D = pd.read_parquet(PARQUET, columns=["doc", "page", "fill"])
    ids = np.load(EMB / "ids.npy").astype(np.int64)
    have = set((D.doc.astype(np.int64) * 100000 + D.page)[D.fill.notna()])
    todo = [(int(d), int(p)) for d, p in ids if int(d) * 100000 + int(p) not in have]
    man = doc_paths({d for d, _ in todo})
    print(len(todo), "pages without a value", flush=True)
    t0 = time.time()
    res = _pool(_fill_page, [(d, p, man[d]) for d, p in todo], cfg.train.workers)
    ok = [i for i, r in enumerate(res) if r]
    new = dict(
        ids=np.array([todo[i] for i in ok], np.int32).reshape(-1, 2),
        fill=np.array([res[i][0] for i in ok], np.float32),
        pic=np.array([res[i][1] for i in ok], np.float32),
        win=np.array([res[i][2:] for i in ok], np.float32).reshape(-1, 4),
    )
    out = PCT / "out" / "local"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "scan_pct_local.npz").exists():
        old = dict(np.load(out / "scan_pct_local.npz"))
        keep = ~np.isin(
            old["ids"][:, 0].astype(np.int64) * 100000 + old["ids"][:, 1],
            new["ids"][:, 0].astype(np.int64) * 100000 + new["ids"][:, 1],
        )
        new = {k: np.concatenate([old[k][keep], new[k]]) for k in new}
    np.savez(out / "scan_pct_local.npz", **new)
    print(
        f"{len(ok)} pages got a value, {len(todo) - len(ok)} not (file or page unreadable), {int(time.time() - t0)} s"
    )


def no_value_reasons(cfg=CONFIG):
    """Why the pages of PARQUET without a value have none -> PCT/no_value_reasons.csv, counted by reason."""
    D = pd.read_parquet(PARQUET, columns=["doc", "page", "fill"])
    D = D[D.fill.isna()]
    man = doc_paths(set(D.doc))
    D = D.assign(why=_pool(_why, [(int(d), int(p), man[int(d)]) for d, p in zip(D.doc, D.page)], cfg.train.workers))
    D.to_csv(PCT / "no_value_reasons.csv", index=False)
    print(len(D), "pages without a value:", D.why.value_counts().to_dict())


def recheck(n=3000, seed=1, cfg=CONFIG):
    """n random pages of PARQUET with a value, done again from the source files like the shards; stored and new values compared -> PCT/recheck.csv."""
    D = pd.read_parquet(PARQUET)
    D = D[D.fill.notna()].sample(n, random_state=seed).reset_index(drop=True)
    man = doc_paths(set(D.doc))
    res = _pool(_recheck_page, [(int(d), int(p), man[int(d)]) for d, p in zip(D.doc, D.page)], cfg.train.workers)
    ok = np.array([r is not None for r in res])
    new = np.array([r if r else (np.nan,) * 6 for r in res], float)
    cols = ("fill", "pic", "wx", "wy", "ww", "wh")
    for i, c in enumerate(cols):
        D["new_" + c] = new[:, i]
    D.to_csv(PCT / "recheck.csv", index=False)
    print(f"{len(D)} pages, {int(ok.sum())} done again, {int((~ok).sum())} failed")
    d = D[ok]
    for c in cols:
        e = (d["new_" + c] - d[c]).abs()
        print(
            f"  {c}: mean abs diff {e.mean():.4f}, median {e.median():.4f}, p99 {e.quantile(0.99):.4f}, max {e.max():.4f}, within 0.01 on {100 * (e <= 0.01).mean():.1f} % of pages, "
            f"within 0.05 on {100 * (e <= 0.05).mean():.1f} %"
        )
