"""Placement of the kept downloads: the pages of a file are picked like the corpus top-ups (tools/pages_extra.py: at most
max(10 + 10 % of its pages, 50), the most different first: farthest-first on the lite layout features), rendered like the shards
(768 x 1024), embedded like the pool (SigLIP 2, corpus_collecting pagetype/embed_kaggle.py), put through the ds_metric heads; a page fills
the open point it lands on (distance <= r = half the spacing of the run, targets.json r; nearest first, a point once). A page whose
1536-d vector has cosine > curate.DUP_SIM with one of its 16 nearest corpus pages is a corpus duplicate and fills nothing (a file of
only such pages: status `known`). Every file is kept. -> table `placed` (file, page, point, d) of the acquire db."""

import json
import sys

import numpy as np
import pymupdf
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ingest.litefeat import pick
from .fetch import db
from .paths import CC_ROOT, STORE, TOOLS, WORK

W, H = 768, 1024
METRIC = STORE.parent / "archive" / "info" / "ds100k" / "metric" / "out"


def heads():
    """1536-d page vectors -> the 24-d space of ds_metric (3 heads of 8, each scaled to mean squared random-pair distance 1)."""
    sd = torch.load(METRIC / "head.pt", map_location="cpu")
    f = [nn.Sequential(nn.Linear(1536, 512), nn.GELU(), nn.Linear(512, 8)) for _ in range(3)]
    for k in range(3):
        f[k].load_state_dict({n.split(".", 2)[2]: v for n, v in sd.items() if n.startswith(f"f.{k}.")})
    msd = json.load(open(METRIC / "metric.json"))["pair_msd"]
    return lambda V: (
        torch.cat([F.normalize(f[h](torch.from_numpy(V).float()), dim=1) / msd[h] ** 0.5 for h in range(3)], 1)
        .detach()
        .numpy()
    )


def corpus():
    """(24-d vectors, 1536-d vectors) of the in-scope corpus pages (ds_targets aux Z, the pool parts in its order)."""
    import glob

    sys.path.insert(0, str(TOOLS))
    import ds_targets as DT
    import kaggle_fps as KF

    z = np.load(DT.OUT / "aux" / "Z.npz")
    parts = [np.load(x, mmap_mode="r") for x in sorted(glob.glob(str(KF.WORK / "data" / "pool_*.npy")))]
    starts = np.cumsum([0] + [len(x) for x in parts])
    k = np.searchsorted(starts, z["pool_i"], side="right") - 1
    return torch.from_numpy(z["Z"].astype(np.float32)), lambda I: np.stack(
        [parts[k[i]][z["pool_i"][i] - starts[k[i]]] for i in I]
    ).astype(np.float32)


def render(path):
    """(page, JPEG bytes) of the picked pages, stretched to W x H like the shards."""
    import cv2

    out = []
    with pymupdf.open(path) as doc:
        for p in (doc[i] for i in pick(doc)):
            pm = p.get_pixmap(
                matrix=pymupdf.Matrix(W / p.rect.width, H / p.rect.height), colorspace=pymupdf.csRGB, alpha=False
            )
            rgb = np.frombuffer(pm.samples, np.uint8).reshape(pm.h, pm.w, 3)
            out.append(
                (
                    p.number,
                    cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])[
                        1
                    ].tobytes(),
                )
            )
    return out


def batches(files, pages):
    """Files grouped so that a group renders about PAGES pages (a file's pick is at most 50 pages): bounded memory per step."""
    out, n = [], 0
    for f in files:
        out.append(f)
        n += 50
        if n >= pages:
            yield out
            out, n = [], 0
    if out:
        yield out


def run(r, pages=400):
    """r: half the spacing of the run (targets.json r)."""
    sys.path.insert(0, str(CC_ROOT / "pagetype"))
    import embed_kaggle as K

    torch.set_num_threads(8)
    c = db()
    c.execute("create table if not exists placed (file text, page int, point int, d real)")
    P = np.load(WORK / "points.npz")
    filled = {p for (p,) in c.execute("select point from placed")}
    seen = {f for (f,) in c.execute("select distinct file from placed")} | {
        f for (f,) in c.execute("select file from got where status = 'unplaced'")
    }
    todo = [f for (f,) in c.execute("select file from got where file is not null") if f not in seen]
    from ..curate import DUP_SIM

    model, head = K.load_model("cpu"), heads()
    Zc, raw = corpus()
    done = 0
    for files in batches(todo, pages):
        done += len(files)
        rendered = [(f, i, j) for f in files for i, j in render(STORE / f)]
        with torch.inference_mode():
            V = np.concatenate(
                [
                    K.embed_pages(
                        model,
                        [torch.frombuffer(bytearray(j), dtype=torch.uint8) for _, _, j in rendered[b : b + 8]],
                        "cpu",
                    )[0]
                    for b in range(0, len(rendered), 8)
                ]
            )
        open_ = np.array([p not in filled for p in P["point"]])
        ids, Vp = P["point"][open_], torch.from_numpy(P["V"][open_])
        Zn = torch.from_numpy(head(V))
        nb = torch.cat(
            [torch.cdist(Zn[a : a + 256], Zc).topk(16, largest=False).indices for a in range(0, len(Zn), 256)]
        ).numpy()
        Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
        dup = np.array(
            [(F.normalize(torch.from_numpy(raw(nb[k])), dim=1).numpy() @ Vn[k]).max() > DUP_SIM for k in range(len(V))]
        )
        d, i = torch.cdist(Zn, Vp).min(1)
        d[torch.from_numpy(dup)] = float("inf")
        used = set()
        for k in np.argsort(d.numpy()):
            f, pg, _ = rendered[k]
            p = int(ids[i[k]])
            if d[k] > r:
                break
            if p in filled:
                continue
            filled.add(p)
            used.add(f)
            c.execute("insert into placed values (?,?,?,?)", (f, pg, p, float(d[k])))
        known = {f for f in files if all(dup[k] for k, (g, _, _) in enumerate(rendered) if g == f)}
        for f in set(files) - used:
            c.execute("update got set status = ? where file = ?", ("known" if f in known else "unplaced", f))
        c.commit()
        print(f"{done}/{len(todo)} files: placed {len(used)} of {len(files)}; points filled {len(filled)}", flush=True)
