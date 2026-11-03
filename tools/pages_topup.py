"""N new pages for the 768 x 1024 set, by the rules of tools/pages_extra.py, keyed by file against the embedding as it is now (emb_out after
tools/layouts_revise.py: one id per file, no repeated page): a document gives at most max(10 + 10 % of its pages, 50) pages in all;
inside a document the most different free pages first (farthest-first on the lite layout features, seeded with its embedded pages);
languages by water filling (a small language gives all it has); inside a language the k-means clusters of the features trimmed to a
common cap, the densest points first. Page counts from layouts_revise's scan. A document never embedded gets a new id (above the
largest one). Data: Layouts/archive/info/pages_topup (WORK).

  python tools/pages_topup.py prior                   # pages of the earlier selections (pages_1m, pages_extra, more_pages, form_pages, ...) that are
                                                      # not embedded and still exist (file now: same path, else the manifest's file of
                                                      # its hash; copies dropped as duplicates and pages past the end out) -> WORK/prior.parquet;
                                                      # they go first and count against the budgets
  python tools/pages_topup.py features [--window 3]   # lite features of up to window x budget free pages per document (+ 2 seeds) -> WORK/lite_feats
  python tools/pages_topup.py select [--total 1500000]  # prior + new pages up to the total -> WORK/select.parquet (doc, page, path, lang, new_doc,
                                                      # src), shard order (documents shuffled)
then: python tools/pages_1m.py build --work pages_topup --first-shard 400 --ds-base 18

6 workers."""
import argparse
import json
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import more_pages as M
import pages_1m as P1
import pages_extra as X

LAYOUTS = P1.LAYOUTS
WORK = LAYOUTS / "archive" / "info" / "pages_topup"
KEY, SEEDS, CHUNK, WORKERS = M.KEY, X.SEEDS, 400, 6
SELECTIONS = ("pages_1m", "pages_extra", "more_pages", "form_pages", "pages_fixsh", "pages_top180", "emb_check")


def lost_pages():
    """Files that lost pages after the manifest counted them (layouts_revise scan): their old page numbers no longer hold."""
    d = pd.read_parquet(LAYOUTS / "archive" / "info" / "layouts_revise" / "diff.parquet")
    return set(d.now[d.pages_now < d.pages].dropna())


def emb_dir():
    from dspages.hand.config import PATHS
    return P1.F.Path(PATHS.uvtm) / "Labels" / "embedding" / "emb_out"


def cmd_prior(_):
    import re
    info = LAYOUTS / "archive" / "info"
    hexn = re.compile(r"^([0-9a-f]{12})(?![0-9a-f])")
    npages = dict(pd.read_parquet(info / "layouts_revise" / "scan.parquet", columns=["path", "pages"]).itertuples(False, None))
    dup = set(pd.read_parquet(info / "layouts_revise" / "dups.parquet").src)
    with sqlite3.connect(f"file:{LAYOUTS / 'manifest.sqlite'}?mode=ro", uri=True) as db:
        plan = dict(db.execute("select doc, new_path from plan"))
    man = {r["id"]: r["path"] for r in map(json.loads, open(emb_dir() / "manifest.jsonl"))}
    ids = np.load(emb_dir() / "ids.npy")
    emb = set(zip([man[d] for d in ids[:, 0]], ids[:, 1].tolist()))
    A = pd.concat([pd.read_parquet(info / f / "select.parquet", columns=["path", "page"]).assign(src=f) for f in SELECTIONS]
                  + [pd.read_parquet(info / "pages_extra" / "select_topup.parquet", columns=["path", "page"]).assign(src="pages_extra")])

    def now(p):
        if p in npages:
            return p
        if p in dup:
            return None
        m = hexn.match(p.rsplit("/", 1)[-1])
        q = plan.get(m.group(1)) if m else None
        return q if q in npages else None

    mp = {p: now(p) for p in A.path.unique()}
    A["path"] = A.path.map(mp)
    A = A[A.path.notna() & ~A.path.isin(lost_pages()) & (A.page < A.path.map(npages))]
    A = A[[k not in emb for k in zip(A.path, A.page)]].drop_duplicates(["path", "page"])
    WORK.mkdir(parents=True, exist_ok=True)
    A.reset_index(drop=True).to_parquet(WORK / "prior.parquet")
    print(len(A), "pages of", A.path.nunique(), "files selected before, not embedded, still there:", A.src.value_counts().to_dict())


def docs(all_docs=False):
    """Documents of the manifest with their file now: doc (embedding id; new ones above the largest), path, lang, actual pages,
    used (embedded + prior pages), extra (budget left), new_doc; only PDFs with budget left unless all_docs."""
    emb = emb_dir()
    scan = pd.read_parquet(LAYOUTS / "archive" / "info" / "layouts_revise" / "scan.parquet", columns=["path", "pages"])
    with sqlite3.connect(f"file:{LAYOUTS / 'manifest.sqlite'}?mode=ro", uri=True) as db:
        D = pd.read_sql("select doc as h, new_path as path, lang, cls from plan", db).merge(scan, on="path")
    D = D[D.pages > 0].rename(columns={"pages": "actual"})
    man = pd.DataFrame([json.loads(l) for l in open(emb / "manifest.jsonl")]).rename(columns={"id": "doc"})
    D = D.merge(man, on="path", how="left")
    D["new_doc"] = D.doc.isna()
    D.loc[D.new_doc, "doc"] = man.doc.max() + 1 + np.arange(int(D.new_doc.sum()))
    D["doc"] = D.doc.astype(int)
    ids = np.load(emb / "ids.npy")
    prior = pd.read_parquet(WORK / "prior.parquet").merge(D[["path", "doc"]], on="path")[["doc", "page"]]
    used = pd.concat([pd.DataFrame({"doc": ids[:, 0], "page": ids[:, 1]}), prior]).groupby("doc").page.apply(sorted)
    D["used"] = D.doc.map(used.map(len)).fillna(0).astype(int)
    D["extra"] = (np.minimum(X.budget(D.actual.values), D.actual) - D.used).clip(lower=0)
    D["scan"] = D.cls.isin([None, "scan-like"]) | D.cls.isna()
    if not all_docs:
        D = D[D.path.str.lower().str.endswith(".pdf") & (D.extra > 0)]
    return D.reset_index(drop=True), used.to_dict()


def old_features(D):
    """Features of tools/pages_extra.py (keys: old ids of emb_out/manifest.jsonl.before_revise) that still hold: the file is there and
    unchanged, the page exists -> WORK/lite_feats/old.npz under the ids of D; {doc: set of pages}."""
    f = WORK / "lite_feats" / "old.npz"
    if not f.exists():
        old = {r["id"]: r["path"] for r in map(json.loads, open(emb_dir() / "manifest.jsonl.before_revise"))}
        now = dict(zip(D.path, D.doc))
        n = dict(zip(D.doc, D.actual))
        bad = lost_pages()
        keys, Vs = [], []
        for g in sorted((LAYOUTS / "archive" / "info" / "pages_extra" / "lite_feats").glob("*.npz")):
            z = np.load(g)
            d = [now.get(p) if p not in bad else None for p in map(old.get, (z["key"] // KEY).tolist())]
            pg = z["key"] % KEY
            ok = np.array([x is not None and q < n[x] for x, q in zip(d, pg)], bool)
            keys.append(np.array([x * KEY for x, o in zip(d, ok) if o], np.int64) + pg[ok]), Vs.append(z["V"][ok])
        key, V = np.concatenate(keys), np.concatenate(Vs)
        key, i = np.unique(key, return_index=True)
        f.parent.mkdir(parents=True, exist_ok=True)
        np.savez(f, key=key, V=V[i], err="{}")
    key = np.load(f)["key"]
    have = pd.Series(key % KEY).groupby(key // KEY).apply(set)
    print(len(key), "pages with old features", flush=True)
    return have.to_dict()


def cmd_features(a):
    D, used = docs()
    have = old_features(docs(all_docs=True)[0])
    jobs = []
    for r in D.itertuples():
        u, h = set(used.get(r.doc, ())), have.get(r.doc, set())
        free = [p for p in range(int(r.actual)) if p not in u and p not in h]
        k = max(0, a.window * int(r.extra) - len(h - u))
        pages = (X.pick(sorted(u - h), SEEDS) if u - h and len(u & h) < SEEDS else []) + (X.pick(free, k) if k else [])
        if pages:
            jobs.append((r.doc, str(LAYOUTS / r.path), bool(r.scan), pages))
    print(len(jobs), "documents,", sum(len(j[3]) for j in jobs), "pages to look at", flush=True)
    fd = WORK / "lite_feats"
    fd.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(WORKERS) as ex:
        for c in range(0, len(jobs), CHUNK):
            f = fd / f"{c // CHUNK:04d}.npz"
            if f.exists():
                continue
            res = list(ex.map(X.lite_job, jobs[c:c + CHUNK], chunksize=4))
            key = [d * KEY + p for d, o, _ in res for p in o]
            V = np.stack([v for _, o, _ in res for v in o.values()]) if key else np.zeros((0, X.LDIM), np.float32)
            np.savez(f"{f}.tmp.npz", key=np.array(key, np.int64), V=V, err=json.dumps({int(d): e for d, _, e in res if e}))
            f.with_name(f.name + ".tmp.npz").rename(f)
            if c // CHUNK % 10 == 0:
                print(f"features {c + len(res)}/{len(jobs)} docs, {int(time.time() - t0)} s", flush=True)


def cmd_select(a):
    from sklearn.cluster import MiniBatchKMeans
    D, used = docs()
    keys, Vs = [], []
    for f in sorted((WORK / "lite_feats").glob("*.npz")):
        z = np.load(f)
        keys.append(z["key"]), Vs.append(z["V"])
    key, V = np.concatenate(keys), np.nan_to_num(np.concatenate(Vs), nan=0.0, posinf=0.0, neginf=0.0)
    for lo, hi, flag in ((0, X.LITE, X.LDIM - 2), (X.LITE, X.LITE + M.NP, X.LDIM - 1)):     # z-score of each block over the pages that have it
        m = V[:, flag] > 0
        mu, sd = V[m, lo:hi].mean(0), V[m, lo:hi].std(0) + 1e-6
        V[:, lo:hi] = np.where(m[:, None], np.clip((V[:, lo:hi] - mu) / sd, -5, 5), 0)
    V[:, -2:] *= 3
    at = {int(k): i for i, k in enumerate(key)}
    rows = []
    for r in D.itertuples():
        u = set(used.get(r.doc, ()))
        have = [p for p in range(int(r.actual)) if p not in u and r.doc * KEY + p in at]
        if have:
            seeds = [V[at[r.doc * KEY + p]] for p in u if r.doc * KEY + p in at]
            rows += [(r.doc, have[i], k) for k, i in enumerate(M.farthest_first(V[[at[r.doc * KEY + p] for p in have]], seeds, int(r.extra)))]
    n_emb = len(np.load(emb_dir() / "ids.npy"))
    prior = pd.read_parquet(WORK / "prior.parquet").merge(docs(all_docs=True)[0][["doc", "path", "lang", "new_doc"]], on="path")
    n = a.total - n_emb - len(prior)
    print(f"embedded {n_emb} + prior {len(prior)} -> {n} new pages to the total {a.total}", flush=True)
    S = pd.DataFrame(rows, columns=["doc", "page", "rank"]).merge(D[["doc", "path", "lang", "new_doc"]], on="doc")
    S["lang"] = S.lang.fillna("unknown")
    S["i"] = [at[int(d) * KEY + int(p)] for d, p in zip(S.doc, S.page)]
    print(len(S), "candidate pages of", S.doc.nunique(), "documents", flush=True)
    lc = S.lang.value_counts()
    q = dict(zip(lc.index, P1.water(lc.values, n)))
    print("language cap:", max(q.values()), "| languages kept whole:", sum(q[l] == lc[l] for l in lc.index), "of", len(lc), flush=True)
    keep = []
    for l, g in S.groupby("lang"):
        if q[l] >= len(g):
            keep.append(g.index.values)
            continue
        Xl = V[g.i.values]
        lab = MiniBatchKMeans(int(np.clip(len(g) // 250, 20, 400)), random_state=0, batch_size=4096, n_init=1, max_iter=50).fit_predict(Xl)
        cl = pd.Series(lab).value_counts()
        for c, qq in zip(cl.index, P1.water(cl.values, q[l])):
            if qq:
                idx = np.where(lab == c)[0]
                keep.append(g.index.values[idx if qq >= len(idx) else idx[X.thin_l2(Xl[idx], qq)]])
    K = pd.concat([prior.assign(lang=prior.lang.fillna("unknown")), S.loc[np.concatenate(keep)].drop(columns=["i", "rank"]).assign(src="new")])
    docs_ = K.doc.unique()
    o = dict(zip(docs_[np.random.default_rng(0).permutation(len(docs_))], range(len(docs_))))
    K = K.assign(o=K.doc.map(o)).sort_values(["o", "page"]).drop(columns="o").reset_index(drop=True)
    assert not K.duplicated(["path", "page"]).any()
    K.to_parquet(WORK / "select.parquet")
    print(len(K), "pages of", K.doc.nunique(), "documents (new documents", int(K.drop_duplicates("doc").new_doc.sum()), ")", K.src.value_counts().to_dict())
    print(K.lang.value_counts().head(10).to_dict(), "->", WORK / "select.parquet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("features")
    f.add_argument("--window", type=int, default=3)
    s = sp.add_parser("select")
    s.add_argument("--total", type=int, default=1_500_000)
    sp.add_parser("prior")
    a = ap.parse_args()
    {"prior": cmd_prior, "features": cmd_features, "select": cmd_select}[a.cmd](a)
