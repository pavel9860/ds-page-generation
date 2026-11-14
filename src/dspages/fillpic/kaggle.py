"""Kaggle side: the detector as a dataset, one CPU script kernel per group of shard datasets, and the results merged into the manifest."""
import json
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CONFIG
from .paths import EMB, MANIFEST_DB, MODEL, PARQUET, PCT, WORK, doc_hash
from ..kaggle_api import api  # noqa: F401  (used by callers as <module>.api)

SHIPPED = ("__init__.py", "config.py", "maps.py", "tiles.py", "features.py", "detector.py", "shards.py")
DETECTOR_DIR = WORK / "kaggle_detector"
KERNEL = Path(__file__).with_name("kernel.py")




def export(cfg=CONFIG):
    """DETECTOR_DIR: the model, the modules of the detector (flat: kernel.py assembles package `fillpic` from them) and the dataset metadata."""
    k = cfg.kaggle
    shutil.rmtree(DETECTOR_DIR, ignore_errors=True)
    DETECTOR_DIR.mkdir(parents=True)
    for f in ("ink.json", "pic.json", "cols.npy"):
        shutil.copy(MODEL / f, DETECTOR_DIR / f)
    for f in SHIPPED:
        shutil.copy(Path(__file__).with_name(f), DETECTOR_DIR / f)
    (DETECTOR_DIR / "dataset-metadata.json").write_text(json.dumps({"title": k.detector_dataset, "id": f"{k.user}/{k.detector_dataset}", "licenses": [{"name": "CC-BY-NC-SA-4.0"}]}))
    print(sorted(p.name for p in DETECTOR_DIR.iterdir()), "in", DETECTOR_DIR)


def upload_detector(cfg=CONFIG):
    k, ref = api(), f"{cfg.kaggle.user}/{cfg.kaggle.detector_dataset}"
    try:
        k.dataset_list_files(ref)
        print(k.dataset_create_version(str(DETECTOR_DIR), "detector", dir_mode="skip"))
    except Exception:
        print(k.dataset_create_new(str(DETECTOR_DIR), public=False, dir_mode="skip"))


def push(kernel, datasets, limit=0, only="", internet=False, cfg=CONFIG):
    k, c = api(), cfg.kaggle
    folder = PCT / "kernel" / kernel
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    script = KERNEL.read_text()
    if limit or only:
        script = f'import os\nos.environ["LIMIT"] = "{limit}"\nos.environ["ONLY"] = "{only}"\n' + script
    (folder / KERNEL.name).write_text(script)
    sources = [f"{c.user}/{c.shard_dataset.format(d)}" for d in datasets] + [f"{c.user}/{c.detector_dataset}"]
    (folder / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{c.user}/{kernel}", "title": kernel, "code_file": KERNEL.name, "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": False,
        "enable_tpu": False, "enable_internet": internet, "keywords": [], "dataset_sources": sources, "kernel_sources": [], "competition_sources": [], "model_sources": []}))
    print(k.kernels_push(str(folder)), flush=True)


def wait(kernels, hours=10, cfg=CONFIG):
    c = cfg.kaggle
    k, t0 = api(), time.time()
    while True:
        st = {n: str(k.kernels_status(f"{c.user}/{n}").status).lower() for n in kernels}
        print(f"[{int((time.time() - t0) / 60)} min]", st, flush=True)
        if all(("complete" in s or "error" in s or "cancel" in s) for s in st.values()):
            break
        if time.time() - t0 > hours * 3600:
            raise SystemExit("time limit")
        time.sleep(c.poll_seconds)
    if not all("complete" in s for s in st.values()):
        raise SystemExit(1)


def fetch(kernels, cfg=CONFIG):
    for n in kernels:
        out = PCT / "out" / n
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        subprocess.run([sys.executable, "-m", "kaggle", "kernels", "output", f"{cfg.kaggle.user}/{n}", "-p", str(out)], check=False, timeout=3600)
        print(n, len(list(out.glob("scan_pct_*.npz"))), "shard files", flush=True)


def rerun(prefix, n, cfg=CONFIG):
    """n kernels <prefix>1.. over the shards of the shard datasets that no PCT/out/* has."""
    k, c = api(), cfg.kaggle
    have = {f.stem[len("scan_pct_"):] for d in (PCT / "out").iterdir() for f in d.glob("scan_pct_*.npz")}
    missing = {}
    for d in c.shard_datasets:
        try:
            names = [Path(f.name).stem for f in k.dataset_list_files(f"{c.user}/{c.shard_dataset.format(d)}").files]
        except Exception:
            continue
        if [x for x in names if x not in have]:
            missing[d] = [x for x in names if x not in have]
    flat = sorted((x, d) for d, ns in missing.items() for x in ns)
    print(len(flat), "shards missing:", {d: len(v) for d, v in missing.items()}, flush=True)
    for i in range(n):
        part = flat[i::n]
        if part:
            push(f"{prefix}{i + 1}", sorted({d for _, d in part}), only=",".join(x for x, _ in part), cfg=cfg)


def merge(cfg=CONFIG):
    """All PCT/out/*/scan_pct_*.npz -> PARQUET and the manifest table scan_pct. A page in several shards takes the value (not NaN) of the highest shard name; only pages of the
    embedding table are kept."""
    best = {}
    for f in (f for d in (PCT / "out").iterdir() for f in d.glob("scan_pct_*.npz")):
        if f.name not in best or len(np.load(f)["ids"]) > len(np.load(best[f.name])["ids"]):
            best[f.name] = f
    parts = []
    for name in sorted(best):
        z = np.load(best[name])
        parts.append(pd.DataFrame({"doc": z["ids"][:, 0], "page": z["ids"][:, 1], "fill": z["fill"], "pic": z["pic"],
                                   "wx": z["win"][:, 0], "wy": z["win"][:, 1], "ww": z["win"][:, 2], "wh": z["win"][:, 3]}))
    D = pd.concat(parts, ignore_index=True)
    D = D.iloc[np.argsort(D.fill.notna().to_numpy(), kind="stable")].drop_duplicates(["doc", "page"], keep="last")
    ids = np.load(EMB / "ids.npy").astype(np.int64)
    D = D[pd.Series(D.doc.astype(np.int64) * 100000 + D.page).isin(ids[:, 0] * 100000 + ids[:, 1]).to_numpy()]
    hashes = {}
    for line in open(EMB / "manifest.jsonl"):
        r = json.loads(line)
        hashes[r["id"]] = doc_hash(r["path"])
    D.insert(0, "hash", D.doc.map(hashes))
    D.to_parquet(PARQUET)
    E = D[D.hash.notna()].drop_duplicates(["hash", "page"])
    with sqlite3.connect(MANIFEST_DB, timeout=600) as db:
        db.execute("create table if not exists scan_pct (hash text, page integer, fill real, pic real, wx real, wy real, ww real, wh real)")
        db.execute("delete from scan_pct")
        db.executemany("insert into scan_pct values (?,?,?,?,?,?,?,?)", E[["hash", "page", "fill", "pic", "wx", "wy", "ww", "wh"]].itertuples(index=False, name=None))
        db.execute("create index if not exists scan_pct_hash on scan_pct (hash, page)")
    print(len(D), "pages of", len(ids), f"in {PARQUET.name},", len(E), "in manifest.sqlite scan_pct;", int(D.fill.isna().sum()), "without a value\n", D[["fill", "pic"]].describe().round(3).to_string())
