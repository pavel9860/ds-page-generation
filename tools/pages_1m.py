"""1M diverse pages of the corpus as 768 x 1024 JPEG shards (tools/form_pages.py: no padding, q90) uploaded to Kaggle as they are built.

  python tools/pages_1m.py select   # embedded pages -> WORK/select.parquet: N pages by coverage of the embedding space, languages capped
  python tools/pages_1m.py build    # shards WORK/npz/pages_NNNN.npz; every dataset (SHARDS_PER_DATASET shards, ~5 GB) uploaded when complete

Selection: pages whose file exists and that are inside it (not page 1 of a one-image file); a cap per language (water filling: a language
with fewer pages than the cap keeps all of them, the rest share what is left); inside a language the k-means clusters of the embedding space are trimmed to a common cap (water filling: big clusters lose pages, small ones
keep all) and a trimmed cluster loses its densest points first (`thin`: the point closest to another goes, until the quota is left).
The documents are shuffled before the shards are cut, so every shard is a sample of the whole set. 6 workers; build resumes."""
import argparse
import json
import re
import sqlite3
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import pandas as pd

import form_pages as F

sys.path.insert(0, "/run/media/p/D/BUISNESS/corpus_collecting")

LAYOUTS, EMB = F.LAYOUTS, F.EMB
WORK = LAYOUTS / "archive" / "info" / "pages_1m"
NPZ = WORK / "npz"
TOTAL = 1_000_000
CLUSTERS = 1000
DATASET = "pavel9860/pages-768x1024"
HEX = re.compile(r"(?<![0-9a-f])[0-9a-f]{12}(?![0-9a-f])")


def water(supply, total):
    """Quota per key: min(supply, cap) with the cap found so that the quotas add up to `total` (all of it when it is not more than that)."""
    s = np.asarray(supply, np.int64)
    if s.sum() <= total:
        return s.copy()
    lo, hi = 0, int(s.max())
    while lo < hi:
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if np.minimum(s, mid).sum() <= total else (lo, mid - 1)
    q = np.minimum(s, lo)
    rest = int(total - q.sum())
    room = np.where(s > q)[0]
    q[room[:rest]] += 1
    return q


def thin(V, q):
    """Indices of q of the n unit vectors V kept by removing the densest points: again and again the point whose nearest remaining
    neighbour is closest goes (of a pair of near copies one is left), until q are left. Sparse points and outliers stay."""
    n = len(V)
    S = V @ V.T
    np.fill_diagonal(S, -2)
    alive = np.ones(n, bool)
    am = S.argmax(1)                                           # nearest neighbour of every point and its similarity
    nn = S[np.arange(n), am]
    for _ in range(n - q):
        i = int(np.argmax(np.where(alive, nn, -3)))
        alive[i] = False
        S[i, :], S[:, i] = -3, -3
        for j in np.where(alive & (am == i))[0]:               # the points that had i as their neighbour look again
            am[j] = S[j].argmax()
            nn[j] = S[j, am[j]]
    return np.where(alive)[0]


def cmd_select(_):
    from sklearn.cluster import MiniBatchKMeans
    ids = np.load(EMB / "ids.npy")
    E = np.load(EMB / "embeddings.npy", mmap_mode="r")
    man = {json.loads(l)["id"]: json.loads(l)["path"] for l in open(EMB / "manifest.jsonl")}
    D = pd.read_parquet(LAYOUTS / "archive/info/emb_sync/docs.parquet", columns=["doc", "actual"]).set_index("doc").actual
    db = sqlite3.connect(f"file:{LAYOUTS / 'manifest.sqlite'}?mode=ro", uri=True)
    lang = dict(db.execute("select doc, lang from plan"))
    S = pd.DataFrame({"row": np.arange(len(ids)), "doc": ids[:, 0], "page": ids[:, 1]})
    S["path"] = S.doc.map(man)
    S["actual"] = S.doc.map(D)
    S["h"] = [next(iter(HEX.findall(p.rsplit("/", 1)[-1])), None) if isinstance(p, str) else None for p in S.path]
    S["lang"] = S.h.map(lang).fillna("unknown")
    print(len(S), "embedded pages", flush=True)
    S = S[S.path.notna() & (S.page < S.actual)]
    ex = {p: (LAYOUTS / p).is_file() for p in S.path.unique()}
    S = S[S.path.map(ex)].reset_index(drop=True)
    print(len(S), "pages of existing files inside them,", S.lang.nunique(), "languages", flush=True)
    X = np.asarray(E[S.row.values], np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    km = MiniBatchKMeans(CLUSTERS, random_state=0, batch_size=8192, n_init=1, max_iter=60).fit(X)
    S["c"] = km.labels_
    cen = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
    S["d"] = 1 - (X * cen[km.labels_]).sum(1)
    lc = S.lang.value_counts()
    lq = dict(zip(lc.index, water(lc.values, TOTAL)))
    print("language cap:", int(max(lq.values())), "| languages kept whole:", sum(lq[l] == lc[l] for l in lc.index), "of", len(lc), flush=True)
    keep = []
    for l, g in S.groupby("lang"):
        if lq[l] >= len(g):
            keep.append(g.index.values)
            continue
        cl = g.groupby("c").size()
        cq = water(cl.values, lq[l])
        for (c, n), q in zip(cl.items(), cq):
            if q:
                gg = g[g.c == c]
                keep.append(gg.index.values if q >= len(gg) else gg.index.values[thin(X[gg.index.values], q)])
    K = S.loc[np.concatenate(keep)].drop(columns=["row", "actual", "h"])
    docs = K.doc.unique()
    order = dict(zip(docs[np.random.default_rng(0).permutation(len(docs))], range(len(docs))))
    K = K.assign(o=K.doc.map(order)).sort_values(["o", "page"]).drop(columns="o").reset_index(drop=True)
    WORK.mkdir(parents=True, exist_ok=True)
    K.to_parquet(WORK / "select.parquet")
    print(len(K), "pages,", K.doc.nunique(), "documents;", "scans/pdf by extension:", K.path.str.lower().str.endswith(".pdf").mean().round(3), "pdf share")
    print(K.lang.value_counts().head(8).to_dict())


class Shard:
    """The pages of one shard as they come in: JPEG bytes appended to a .bin file (memory stays small), (doc, page) and lengths in lists."""

    def __init__(self, idx, part):
        self.idx, self.n = idx, len(part)
        self.jobs = [(str(LAYOUTS / g.path.iloc[0]), list(zip(g.doc, g.page))) for _, g in part.groupby("doc", sort=False)]
        self.path = NPZ / f"pages_{idx:04d}.npz"
        self.bin = open(NPZ / f"pages_{idx:04d}.bin", "wb")
        self.ids, self.len, self.failed = [], [], []
        self.left, self.lock, self.done = len(self.jobs), threading.Lock(), threading.Event()

    def add(self, res):
        for d, p, b in res:
            self.bin.write(b)
            self.ids.append((d, p))
            self.len.append(len(b))

    def finish(self, job, fut):
        """Done callback of one document's job (runs in the pool's thread); a job that died with the pool is retried in `write`."""
        try:
            res = fut.result()
        except BaseException:
            res = None
        with self.lock:
            if res is None:
                self.failed.append(job)
            else:
                self.add(res)
            self.left -= 1
            if not self.left:
                self.done.set()

    def write(self):
        """Wait for all the jobs, retry the ones a crashed pool lost (F.run_jobs splits them until the file that kills MuPDF is found),
        then the .bin file + ids + offsets -> the npz (the jpeg array is a memmap: nothing is held in memory); -> (pages, seconds)."""
        self.done.wait()
        if self.failed:
            self.add(F.run_jobs(self.failed))
        self.bin.close()
        off = np.cumsum([0] + self.len).astype(np.int64)
        tmp = self.path.with_name(self.path.name + ".tmp.npz")
        binf = self.path.with_suffix(".bin")
        np.savez(tmp, ids=np.asarray(self.ids, np.int32).reshape(-1, 2), offsets=off,
                 jpeg=np.memmap(binf, np.uint8, "r") if off[-1] else np.zeros(0, np.uint8))
        tmp.rename(self.path)
        binf.unlink()
        return len(self.ids), self.n


def make_pool():
    return ProcessPoolExecutor(F.WORKERS)


def safe_upload(k, i, files):
    """F.upload that tries again (a failed attempt must not lose the dataset; the next start repeats whatever is not listed)."""
    for attempt in range(4):
        try:
            return F.upload(k, i, files, DATASET, WORK)
        except Exception as e:
            print(f"upload of dataset {i} failed ({attempt + 1}/4): {e!r}"[:300], flush=True)
            time.sleep(120)
    return f"dataset {i}: upload FAILED (shards stay on disk)"


def cmd_build(a):
    """The 6 workers are never idle: the jobs of the next shard are queued while the previous one is written, and the writes and the
    uploads run in threads. A shard that is not on disk is built again after a crash; datasets already on Kaggle are skipped."""
    from dspages.kaggle_api import api
    global WORK, NPZ
    if a.work:
        WORK = LAYOUTS / "archive" / "info" / a.work
        NPZ = WORK / "npz"
    F.WORKERS = a.workers
    first, base = a.first_shard, a.ds_base
    S = pd.read_parquet(WORK / "select.parquet")
    NPZ.mkdir(parents=True, exist_ok=True)
    for f in list(NPZ.glob("*.tmp.npz")) + list(NPZ.glob("*.bin")):        # a shard that was being written when the process died
        f.unlink()
    k = api()
    n_sh = -(-len(S) // F.SHARD_PAGES)
    per = F.SHARDS_PER_DATASET
    files = lambda i: [NPZ / f"pages_{first + j:04d}.npz" for j in range(i * per, min((i + 1) * per, n_sh))]
    uploader, writer, t0 = ThreadPoolExecutor(1), ThreadPoolExecutor(1), time.time()
    submitted, futs, lock = set(), [], threading.Lock()

    def maybe_upload():
        with lock:
            for i in range(-(-n_sh // per)):
                if i not in submitted and all(f.exists() for f in files(i)):
                    submitted.add(i)
                    futs.append(uploader.submit(safe_upload, k, base + i, files(i)))

    def write_shard(sh):
        n, want = sh.write()
        print(f"{sh.path.name}: {n}/{want} pages, {round(sh.path.stat().st_size / 1e9, 2)} GB, {int(time.time() - t0)} s", flush=True)
        maybe_upload()

    maybe_upload()                                                         # datasets complete from an earlier run
    todo = [s for s in range(n_sh) if not (NPZ / f"pages_{first + s:04d}.npz").exists()]
    pool, inflight, wfuts = make_pool(), [], []

    def feed(s):
        nonlocal pool
        sh = Shard(first + s, S.iloc[s * F.SHARD_PAGES:(s + 1) * F.SHARD_PAGES])
        for job in sh.jobs:
            try:
                fut = pool.submit(F.render_job, job)
            except BrokenProcessPool:                                      # a file killed MuPDF: new pool, the lost jobs are retried in write()
                pool = make_pool()
                fut = pool.submit(F.render_job, job)
            fut.add_done_callback(lambda f, sh=sh, job=job: sh.finish(job, f))
        inflight.append(sh)

    for s in todo[:2]:
        feed(s)
    queue = todo[2:]
    while inflight:
        sh = inflight.pop(0)
        sh.done.wait()
        if queue:
            feed(queue.pop(0))
        wfuts.append(writer.submit(write_shard, sh))
    for w in wfuts:
        w.result()
    maybe_upload()
    for fut in futs:
        print(fut.result(), flush=True)
    print("done", int(time.time() - t0), "s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("select").set_defaults(f=cmd_select)
    bp = sub.add_parser("build")
    bp.add_argument("--work", help="folder under Layouts/archive/info with select.parquet (default pages_1m)")
    bp.add_argument("--first-shard", type=int, default=0, help="number of the first shard file (pages_NNNN.npz)")
    bp.add_argument("--ds-base", type=int, default=0, help="number of the first dataset (pages-768x1024-<n>)")
    bp.add_argument("--workers", type=int, default=6)
    bp.set_defaults(f=cmd_build)
    a = ap.parse_args()
    try:
        a.f(a)
    except BaseException:
        import os
        import traceback
        traceback.print_exc()
        os._exit(1)                         # not a lingering process held up by the upload thread: the service restarts at once
