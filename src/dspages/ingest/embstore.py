"""The embedding table emb_out (embeddings.npy float16 (n, 1536), ids.npy int32 (n, 2) [doc, page] sorted by (doc, page), manifest.jsonl {id, path})."""
import json
import os
import shutil

import numpy as np

from .paths import EMB

DIM, BLOCK = 1536, 50000


def keys(ids):
    return ids[:, 0].astype(np.int64) * 100000 + ids[:, 1]


def backup(names=("embeddings.npy", "ids.npy", "manifest.jsonl"), root=None):
    """<file>.bak of each file: ONE version, the state before the latest change (an older .bak is replaced)."""
    root = root or EMB
    for f in names:
        shutil.copy2(root / f, root / f"{f}.bak")


def manifest():
    with open(EMB / "manifest.jsonl") as fh:
        return {r["id"]: r["path"] for r in map(json.loads, fh)}


def max_doc():
    return max(int(np.load(EMB / "ids.npy")[:, 0].max()), max(manifest()))


def add_manifest(rows):
    """rows: [(doc, path)] appended (a doc already there must have the same path)."""
    have = manifest()
    new = [(d, p) for d, p in rows if d not in have]
    assert all(have[d] == p for d, p in rows if d in have)
    with open(EMB / "manifest.jsonl", "a") as fh:
        fh.writelines(json.dumps({"id": d, "path": p}) + "\n" for d, p in new)
    return len(new)


def merge_rows(E_new, I_new, drop=None):
    """Rows of (E_new, I_new) are added, rows whose key is in `drop` (doc * 100000 + page) are removed first; a new row for a page that is kept is
    not added twice. Backup <file>.bak. Returns (removed, added, total)."""
    Ea, Ia = np.load(EMB / "embeddings.npy", mmap_mode="r"), np.load(EMB / "ids.npy")
    ka, kb = keys(Ia), keys(I_new)
    assert E_new.shape[1] == DIM and np.isfinite(E_new.astype(np.float32)).all() and len(set(kb.tolist())) == len(kb)
    keep = np.flatnonzero(~np.isin(ka, np.fromiter(drop, np.int64))) if drop is not None else np.arange(len(ka))
    add = np.flatnonzero(~np.isin(kb, ka[keep]))
    backup(("embeddings.npy", "ids.npy"))
    key = np.concatenate([ka[keep], kb[add]])
    order = np.argsort(key, kind="stable")
    n = len(key)
    src = np.concatenate([np.zeros(len(keep), np.int8), np.ones(len(add), np.int8)])
    loc = np.concatenate([keep, add])
    out = np.lib.format.open_memmap(EMB / "embeddings.new.npy", mode="w+", dtype=np.float16, shape=(n, DIM))
    for s in range(0, n, BLOCK):
        o = order[s:s + BLOCK]
        blk = np.empty((len(o), DIM), np.float16)
        for t, E in enumerate((Ea, E_new)):
            m = src[o] == t
            if m.any():
                li = loc[o][m]
                srt = np.argsort(li)
                res = np.empty((len(li), DIM), np.float16)
                res[srt] = np.asarray(E[li[srt]])
                blk[m] = res
        out[s:s + len(o)] = blk
    out.flush()
    del out
    np.save(EMB / "ids.new.npy", np.concatenate([Ia[keep], I_new[add]])[order].astype(np.int32))
    E2, I2 = np.load(EMB / "embeddings.new.npy", mmap_mode="r"), np.load(EMB / "ids.new.npy")
    assert E2.shape == (n, DIM) and I2.shape == (n, 2) and (np.diff(keys(I2)) > 0).all()
    os.replace(EMB / "embeddings.new.npy", EMB / "embeddings.npy")
    os.replace(EMB / "ids.new.npy", EMB / "ids.npy")
    return len(ka) - len(keep), len(add), n
