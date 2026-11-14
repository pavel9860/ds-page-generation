"""N more pages for the 768 x 1024 set (tools/pages_1m.py): pages that are not in the 1M, the most diverse ones by a cheap layout proxy.

  python tools/pages_extra.py features                # lite layout vectors of every unused page of the documents with a budget (+ 2 seed pages)
  python tools/pages_extra.py select [--n 500000]     # -> info/pages_extra/select.parquet
  python tools/pages_1m.py build --work pages_extra --first-shard 84 --ds-base 9 --workers 2

A document gives at most max(10 + 10 % of its pages, 50) pages in all (the 1M counts). Per document: the most different pages first
(greedy farthest-first on the lite features, seeded with pages of the document already in the 1M). Per language (water filling over the
languages for the total): the candidates are trimmed like the 1M was: k-means clusters of the feature space, a common cap per cluster (big
clusters lose pages), and a trimmed cluster loses its densest points first. No embedding exists for these pages: the features are a cheap
proxy (text blocks, 9.5 ms a page; a page without text: the picture proxy of a thumbnail of tools/more_pages.py)."""
import argparse
import json
import re
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import more_pages as M
import pages_1m as P1

LAYOUTS, EMB = P1.LAYOUTS, P1.EMB
WORK = LAYOUTS / "archive" / "info" / "pages_extra"
KEY = M.KEY
HEX = re.compile(r"(?<![0-9a-f])[0-9a-f]{12}(?![0-9a-f])")
SEEDS, FEAT_CHUNK = 2, 400


LITE = 20 + M.GRID[0] * M.GRID[1]                  # block-level layout features of a text page
LDIM = LITE + M.NP + 2                             # + the picture proxy of tools/more_pages.py (scans / pages without text) + two flags


def lite_feats(pg):
    """Cheap layout vector of a PDF page from its text blocks only (get_text("blocks"): 3-5 ms, no spans, no drawings): sizes and alignment
    of the blocks, characters per line, digits / capitals, a density grid of the text; -> (vector, number of characters)."""
    W, H = pg.rect.width, pg.rect.height
    bl = pg.get_text("blocks")
    f = np.zeros(LITE)
    txt = [b for b in bl if b[6] == 0 and b[4].strip()]
    img = [b for b in bl if b[6] == 1]
    f[2] = len(img)
    f[3] = min(sum(max(0, b[2] - b[0]) * max(0, b[3] - b[1]) for b in img) / (W * H), 1)
    f[18] = np.log1p(W / H * 10)
    chars = 0
    if txt:
        a = np.array([[b[0] / W, b[1] / H, b[2] / W, b[3] / H] for b in txt])
        n = np.array([len(b[4].strip()) for b in txt], float)
        nl = np.array([b[4].count("\n") + 1 for b in txt], float)
        chars = int(n.sum())
        w, h = a[:, 2] - a[:, 0], a[:, 3] - a[:, 1]
        x0 = np.round(a[:, 0] * 20).astype(int)
        s = "".join(b[4] for b in txt)
        f[0], f[1] = np.log1p(chars), np.log1p(len(txt))
        f[4], f[5], f[6], f[7] = min((w * h).sum(), 1), w.mean(), w.std(), h.mean()
        f[8], f[9] = np.log1p(n.mean()), np.bincount(x0).max() / len(x0)
        f[10], f[11] = len(set(x0)) / 20, np.log1p((nl).mean())
        f[12], f[13] = sum(c.isdigit() for c in s) / max(len(s), 1), sum(c.isupper() for c in s) / max(len(s), 1)
        f[14], f[15] = a[:, 1].min(), a[:, 3].max()
        f[16], f[17] = (np.abs((a[:, 0] + a[:, 2]) / 2 - .5) < .05).mean(), (a[:, 2] > .9).mean()
        f[19] = np.log1p((n / nl).mean())
        gx0, gx1 = np.clip(a[:, 0] * M.GRID[1], 0, M.GRID[1] - 1).astype(int), np.clip(a[:, 2] * M.GRID[1], 0, M.GRID[1] - 1).astype(int)
        gy0, gy1 = np.clip(a[:, 1] * M.GRID[0], 0, M.GRID[0] - 1).astype(int), np.clip(a[:, 3] * M.GRID[0], 0, M.GRID[0] - 1).astype(int)
        cells = np.zeros((M.GRID[0], M.GRID[1]))
        for i in range(len(a)):
            cells[gy0[i]:gy1[i] + 1, gx0[i]:gx1[i] + 1] += n[i] / ((gy1[i] - gy0[i] + 1) * (gx1[i] - gx0[i] + 1))
        f[20:] = (cells / max(cells.sum(), 1)).ravel()
    return f, chars


def lite_job(args):
    """(doc, path, scan, pages) -> (doc, {page: vector}, error): lite_feats of the listed pages (a page without text: the picture proxy of a
    thumbnail; an empty page is not a candidate)."""
    import signal
    doc_id, path, scan, pages = args
    out, err = {}, ""
    signal.signal(signal.SIGALRM, M._alarm)
    signal.alarm(M.DOC_SECONDS)
    try:
        with M.pymupdf.open(path) as doc:
            for p in pages:
                if p >= len(doc):
                    continue
                pg = doc[p]
                v = np.zeros(LDIM, np.float32)
                try:
                    f, chars = lite_feats(pg)
                    v[:LITE], v[-2] = f, chars >= M.TEXT_MIN_CHARS
                except Exception as e:
                    chars, err = 0, f"text {type(e).__name__}"
                if scan or chars < M.TEXT_MIN_CHARS:
                    try:
                        q, ink = M.proxy_feats(M.scan_thumb(doc, pg))
                        v[LITE:LITE + M.NP], v[-1] = q, 1
                        if chars < M.TEXT_MIN_CHARS and ink < M.EMPTY_INK:
                            continue
                    except Exception as e:
                        err = f"proxy {type(e).__name__}"
                        if chars < M.TEXT_MIN_CHARS:
                            continue
                out[p] = v
    except BaseException as e:
        err = f"{type(e).__name__}: {e}"[:200]
    finally:
        signal.alarm(0)
    return doc_id, out, err


def budget(n):
    """Pages a document may give in all (the 1M and this set together): max(10 + 10 % of its pages, 50)."""
    return np.maximum(10 + np.ceil(0.10 * np.nan_to_num(np.asarray(n, float))), 50).astype(int)


def thin_l2(V, q):
    """Indices of q of the n vectors V kept by removing the densest points (P1.thin on Euclidean distance): again and again the point whose
    nearest remaining neighbour is closest goes, until q are left."""
    n = len(V)
    sq = (V * V).sum(1)
    S = -(sq[:, None] + sq[None, :] - 2 * V @ V.T)
    np.fill_diagonal(S, -np.inf)
    alive = np.ones(n, bool)
    am = S.argmax(1)
    nn = S[np.arange(n), am]
    for _ in range(n - q):
        i = int(np.argmax(np.where(alive, nn, -np.inf)))
        alive[i] = False
        S[i, :], S[:, i] = -np.inf, -np.inf
        for j in np.where(alive & (am == i))[0]:
            am[j] = S[j].argmax()
            nn[j] = S[j, am[j]]
    return np.where(alive)[0]


def doc_table(topup=False):
    """Documents with an extra budget: doc, path, scan, actual, used (pages in the 1M), extra, the pages of the 1M, the free pages."""
    I = LAYOUTS / "archive" / "info"
    D = pd.read_parquet(I / "emb_sync" / "docs.parquet", columns=["doc", "path", "actual"])
    D = D[D.path.notna() & (D.actual > 0) & D.path.str.lower().str.endswith(".pdf")].copy()
    if topup:                                                     # the pages already in the embedding table
        ids = np.load(EMB / "ids.npy")
        sel = pd.DataFrame({"doc": ids[:, 0], "page": ids[:, 1]})
        I = LAYOUTS / "archive" / "info"                         # pages that are being embedded again (re-embedded) count as taken
        R = pd.read_parquet(I / "pages_extra" / "missing_diag.parquet")
        M = pd.concat([pd.read_parquet(I / d / "missing_render.parquet") for d in ("pages_extra", "pages_1m")])
        sel = pd.concat([sel, M[M.doc.isin(R.doc[R.reason != "no page rendered"])][["doc", "page"]]])
    else:
        sel = pd.read_parquet(P1.WORK / "select.parquet", columns=["doc", "page"])
    used = sel.groupby("doc").page.apply(sorted)
    D["used"] = D.doc.map(used.map(len)).fillna(0).astype(int)
    D["extra"] = (budget(D.actual.values) - D.used).clip(lower=0)
    D = D[(D.extra > 0) & (D.actual > D.used)].reset_index(drop=True)
    db = sqlite3.connect(f"file:{LAYOUTS / 'manifest.sqlite'}?mode=ro", uri=True)
    cls = dict(db.execute("select doc, cls from plan"))
    h = [next(iter(HEX.findall(p.rsplit("/", 1)[-1])), None) for p in D.path]
    D["scan"] = [cls.get(x) in (None, "scan-like") for x in h]
    D["h"] = h
    return D, used.to_dict()


def pick(free, k):
    return [free[i] for i in np.unique(np.linspace(0, len(free) - 1, min(k, len(free))).round().astype(int))]


def cmd_features(a):
    D, used = doc_table()
    jobs = []
    for r in D.itertuples():
        u = set(used.get(r.doc, ()))
        free = [p for p in range(int(r.actual)) if p not in u]
        seeds = pick(sorted(u), SEEDS) if u else []
        jobs.append((r.doc, str(LAYOUTS / r.path), bool(r.scan), seeds + (pick(free, a.window) if a.window else free)))
    print(len(jobs), "documents,", sum(len(j[3]) for j in jobs), "pages to look at", flush=True)
    fd = WORK / "lite_feats"
    fd.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(2) as ex:
        for c in range(0, len(jobs), FEAT_CHUNK):
            f = fd / f"{c // FEAT_CHUNK:04d}.npz"
            if f.exists():
                continue
            res = list(ex.map(lite_job, jobs[c:c + FEAT_CHUNK], chunksize=4))
            key = [d * KEY + p for d, o, _ in res for p in o]
            V = np.stack([v for _, o, _ in res for v in o.values()]) if key else np.zeros((0, LDIM), np.float32)
            np.savez(f"{f}.tmp.npz", key=np.array(key, np.int64), V=V, err=json.dumps({int(d): e for d, _, e in res if e}))
            f.with_name(f.name + ".tmp.npz").rename(f)
            print(f"features {c + len(res)}/{len(jobs)} docs, {int(time.time() - t0)} s", flush=True)


def lost_languages():
    """{language: pages} of the selected pages that do not exist (page number past the end of the file): what a top-up replaces."""
    I = LAYOUTS / "archive" / "info"
    R = pd.read_parquet(I / "pages_extra" / "missing_diag.parquet")
    M = pd.concat([pd.read_parquet(I / d / "missing_render.parquet") for d in ("pages_extra", "pages_1m")])
    M = M[M.doc.isin(R.doc[R.reason == "no page rendered"])]
    db = sqlite3.connect(f"file:{LAYOUTS / 'manifest.sqlite'}?mode=ro", uri=True)
    lang = dict(db.execute("select doc, lang from plan"))
    return pd.Series([lang.get(next(iter(HEX.findall(p.rsplit("/", 1)[-1])), None), "unknown") for p in M.path]).value_counts().to_dict()


def cmd_select(a):
    from sklearn.cluster import MiniBatchKMeans
    t0 = time.time()
    D, used = doc_table(a.topup)
    keys, Vs = [], []
    for f in sorted((WORK / "lite_feats").glob("*.npz")):
        z = np.load(f)
        keys.append(z["key"]), Vs.append(z["V"])
    key, V = np.concatenate(keys), np.concatenate(Vs)
    bad = ~np.isfinite(V).all(1)
    print(int(bad.sum()), "pages with a NaN / inf feature (set to 0)", flush=True)
    V = np.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)
    for lo, hi, flag in ((0, LITE, LDIM - 2), (LITE, LITE + M.NP, LDIM - 1)):                    # z-score of each block over the pages that have it
        m = V[:, flag] > 0
        mu, sd = V[m, lo:hi].mean(0), V[m, lo:hi].std(0) + 1e-6
        V[:, lo:hi] = np.where(m[:, None], np.clip((V[:, lo:hi] - mu) / sd, -5, 5), 0)
    V[:, -2:] *= 3
    at = {int(k): i for i, k in enumerate(key)}
    print(len(key), "pages with features", flush=True)
    rows = []
    for r in D.itertuples():
        u = set(used.get(r.doc, ()))
        have = [p for p in range(int(r.actual)) if p not in u and r.doc * KEY + p in at]
        if not have:
            continue
        seeds = [V[at[r.doc * KEY + p]] for p in u if r.doc * KEY + p in at]
        M_ = V[[at[r.doc * KEY + p] for p in have]]
        rows += [(r.doc, have[i], rank) for rank, i in enumerate(M.farthest_first(M_, seeds, int(r.extra)))]
    S = pd.DataFrame(rows, columns=["doc", "page", "rank"])
    S["i"] = [at[int(d) * KEY + int(p)] for d, p in zip(S.doc, S.page)]
    print(len(S), "candidate pages of", S.doc.nunique(), "documents (most different first)", int(time.time() - t0), "s", flush=True)
    man = {json.loads(l)["id"]: json.loads(l)["path"] for l in open(EMB / "manifest.jsonl")}
    db = sqlite3.connect(f"file:{LAYOUTS / 'manifest.sqlite'}?mode=ro", uri=True)
    lang = dict(db.execute("select doc, lang from plan"))
    S["path"] = S.doc.map(man)
    S = S[S.path.notna()].copy()
    S["lang"] = [lang.get(next(iter(HEX.findall(p.rsplit("/", 1)[-1])), None), "unknown") for p in S.path]
    ok = {p: (LAYOUTS / p).is_file() for p in S.path.unique()}
    S = S[S.path.map(ok)].reset_index(drop=True)
    lc = S.lang.value_counts()
    if a.topup:                                                   # like for like: the languages of the lost pages first, the rest by water filling
        lost = lost_languages()
        q = {l: min(lost.get(l, 0), int(lc[l])) for l in lc.index}
        if sum(q.values()) > a.n:                                 # fewer pages wanted than were lost: the same mix, scaled down
            q = dict(zip(q, P1.water(list(q.values()), a.n)))
        rest = a.n - sum(q.values())
        if rest > 0:
            left = np.array([lc[l] - q[l] for l in lc.index])
            for l, e in zip(lc.index, P1.water(left, rest)):
                q[l] += int(e)
    else:
        q = dict(zip(lc.index, P1.water(lc.values, a.n)))
    print("language cap:", max(q.values()), "| languages kept whole:", sum(q[l] == lc[l] for l in lc.index), "of", len(lc), flush=True)
    keep = []
    for l, g in S.groupby("lang"):
        if q[l] >= len(g):
            keep.append(g.index.values)
            continue
        X = V[g.i.values]
        k = int(np.clip(len(g) // 250, 20, 400))
        lab = MiniBatchKMeans(k, random_state=0, batch_size=4096, n_init=1, max_iter=50).fit_predict(X)
        cl = pd.Series(lab).value_counts()
        for c, qq in zip(cl.index, P1.water(cl.values, q[l])):
            if qq:
                idx = np.where(lab == c)[0]
                keep.append(g.index.values[idx if qq >= len(idx) else idx[thin_l2(X[idx], qq)]])
        print(" ", l, len(g), "->", q[l], flush=True)
    K = S.loc[np.concatenate(keep)].drop(columns=["i"])
    rng = np.random.default_rng(0)
    docs = K.doc.unique()
    o = dict(zip(docs[rng.permutation(len(docs))], range(len(docs))))
    K = K.assign(o=K.doc.map(o)).sort_values(["o", "page"]).drop(columns="o").reset_index(drop=True)
    WORK.mkdir(parents=True, exist_ok=True)
    K.to_parquet(WORK / ("select_topup.parquet" if a.topup else "select.parquet"))
    print(len(K), "pages of", K.doc.nunique(), "documents; pages per document", K.groupby("doc").size().describe().round(1).to_dict())
    print(K.lang.value_counts().head(8).to_dict(), "->", WORK / "select.parquet")


KERNEL = "pavel9860/emb-extra-768x1024"
DATASETS = [(f"pavel9860/pages-768x1024-{i}", n) for i, n in zip(range(9, 14), (10, 10, 10, 10, 2))]        # datasets of shards 84..125


def cmd_run(a):
    """Embeds the 5 datasets of the extra shards on both T4 GPUs (kaggle_embed.run) -> WORK/out_extra."""
    import kaggle_embed
    kaggle_embed.run(KERNEL, DATASETS, WORK, WORK / "out_extra", a.wait_h)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    fp = sub.add_parser("features")
    fp.add_argument("--window", type=int, default=0, help="0: every unused page; else an evenly spaced window of that many pages")
    fp.set_defaults(f=cmd_features)
    rp = sub.add_parser("run")
    rp.add_argument("--wait-h", type=float, default=4, help="hours to wait for the datasets to be on Kaggle")
    rp.set_defaults(f=cmd_run)
    sp = sub.add_parser("select")
    sp.add_argument("--n", type=int, default=500_000)
    sp.add_argument("--topup", action="store_true", help="pages that are in no embedding yet, replacing the selected pages that do not exist")
    sp.set_defaults(f=cmd_select)
    a = ap.parse_args()
    a.f(a)
