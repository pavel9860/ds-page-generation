"""Which pages of a document enter the set, and the shards of them. A document gives at most max(10 + 10 % of its pages, 50) pages (tools/pages_extra.py
budget); more pages than that: the most different first (farthest-first on the lite layout features). Shards: dspages.fillpic.shards format
(ids int32 (n, 2), offsets, jpeg), documents shuffled with a fixed seed so that every shard samples the whole run."""
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pymupdf

from . import render
from .litefeat import pick as pick_pages
from .paths import LAYOUTS, WORK
from .registry import db

SHARD_PAGES = 10_000
pymupdf.TOOLS.mupdf_display_errors(False)


def pick(args):
    """(doc, relative path, pages) -> (doc, [page numbers]); an image is page 0."""
    doc_id, rel, n = args
    if not rel.lower().endswith(".pdf"):
        return doc_id, [0]
    try:
        with pymupdf.open(LAYOUTS / rel) as doc:
            return doc_id, pick_pages(doc)
    except Exception:
        return doc_id, []


def select(workers=6):
    """Picks the pages of every registered document without any -> table pages. -> (documents, pages)."""
    c = db()
    todo = c.execute("select doc, path, pages from docs where picked is null").fetchall()
    with ProcessPoolExecutor(workers) as ex:
        for d, ps in ex.map(pick, todo, chunksize=16):
            c.executemany("insert or ignore into pages (doc, page) values (?,?)", [(d, p) for p in ps])
            c.execute("update docs set picked = ?, state = ? where doc = ?", (len(ps), "picked" if ps else "unreadable", d))
    c.commit()
    return len(todo), c.execute("select count(*) from pages").fetchone()[0]


def build(run, pages_per_shard=SHARD_PAGES, workers=render.WORKERS):
    """Shards WORK/npz/<run>/pages_<run>_NN.npz of the picked pages not in a shard yet (resumable); -> {shard: pages}."""
    c = db()
    out = WORK / "npz" / run
    out.mkdir(parents=True, exist_ok=True)
    rows = c.execute("select d.doc, d.path, p.page from docs d join pages p using (doc) where p.shard is null and d.state = 'picked' order by d.doc, p.page").fetchall()
    docs = {}
    for d, path, p in rows:
        docs.setdefault((d, path), []).append(p)
    order = np.random.default_rng(0).permutation(len(docs))
    keys = list(docs)
    first = len(list(out.glob("pages_*.npz")))
    shards, cur, n = [], [], 0
    for i in order:
        cur.append(keys[i])
        n += len(docs[keys[i]])
        if n >= pages_per_shard:
            shards.append(cur)
            cur, n = [], 0
    if cur:
        shards.append(cur)
    res = {}
    for s, part in enumerate(shards, first):
        t = time.time()
        got = render.run_jobs([(str(LAYOUTS / path), [(d, p) for p in docs[(d, path)]]) for d, path in part], workers)
        got.sort(key=lambda x: (x[0], x[1]))
        name = f"pages_{run}_{s:02d}"
        off = np.cumsum([0] + [len(b) for _, _, b in got]).astype(np.int64)
        tmp = out / f"{name}.tmp.npz"
        np.savez(tmp, ids=np.array([(d, p) for d, p, _ in got], np.int32).reshape(-1, 2), offsets=off,
                 jpeg=np.frombuffer(b"".join(b for _, _, b in got), np.uint8) if got else np.zeros(0, np.uint8))
        tmp.rename(out / f"{name}.npz")
        c.executemany("update pages set shard = ? where doc = ? and page = ?", [(name, d, p) for d, p, _ in got])
        c.commit()
        res[name] = len(got)
        print(f"{name}: {len(got)} of {sum(len(docs[k]) for k in part)} pages, {time.time() - t:.0f} s", flush=True)
    return res
