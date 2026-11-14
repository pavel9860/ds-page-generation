"""Kaggle side of an ingest run: the shards as a private dataset, one GPU script kernel (kernel.py), the outputs downloaded to WORK/out/<run>.
  push RUN [LIMIT]   (LIMIT pages per shard: a smoke test) dataset pavel9860/ingest-<run> from WORK/npz/<run> (create or new version), kernel pavel9860/ingest-<run>-kernel
  wait RUN   polls the kernel every 5 minutes
  fetch RUN  kernel output -> WORK/out/<run> (embeddings.npy, ids.npy, hw/*.npz, scan_pct_*.npz); checks every shard has its three results"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from ..fillpic.config import CONFIG
from .paths import CC_ROOT, HW_WEIGHTS, WORK
from ..kaggle_api import api  # noqa: F401  (used by callers as <module>.api)

USER = CONFIG.kaggle.user
DETECTOR = f"{USER}/{CONFIG.kaggle.detector_dataset}"
WEIGHTS = f"{USER}/hwdet-yolo26s-weights"
EMBED = CC_ROOT / "pagetype" / "embed_kaggle.py"
FILLPIC = Path(__file__).parents[1] / "fillpic" / "kernel.py"




def dataset(run):
    return f"{USER}/ingest-{run}"


def kernel(run):
    return f"{USER}/ingest-{run}-kernel"


def push(run, limit=0):
    k, ref = api(), dataset(run)
    src = WORK / "npz" / run
    assert list(src.glob("pages_*.npz")), f"no shards in {src}"
    meta = WORK / "dataset" / run
    shutil.rmtree(meta, ignore_errors=True)
    meta.mkdir(parents=True)
    for f in src.glob("pages_*.npz"):
        (meta / f.name).hardlink_to(f)
    (meta / "dataset-metadata.json").write_text(json.dumps({"title": ref.split("/")[1], "id": ref, "licenses": [{"name": "CC-BY-NC-SA-4.0"}]}))
    n = len(list(src.glob("pages_*.npz")))
    exists = any(x.ref == ref for x in k.dataset_list(mine=True, search=ref.split("/")[1]))
    if not exists:
        print(k.dataset_create_new(str(meta), public=False, dir_mode="skip"))
    while True:                                  # a new dataset answers 403 until it has been processed
        try:
            if len(k.dataset_list_files(ref).files) >= n:
                break
        except Exception:
            pass
        time.sleep(30)
    d = WORK / "kernel" / run
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    env = f'import os\nos.environ["LIMIT"] = "{limit}"\n' if limit else ""
    (d / "run.py").write_text(env + f"EMBED_SRC = {EMBED.read_text()!r}\n" + Path(__file__).with_name("kernel.py").read_text())
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": kernel(run), "title": kernel(run).split("/")[1], "code_file": "run.py", "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "keywords": ["gpu"], "dataset_sources": [ref, WEIGHTS, DETECTOR],
        "kernel_sources": [], "competition_sources": [], "model_sources": [], "machine_shape": "NvidiaTeslaT4"}))
    print(k.kernels_push(str(d)))
    push_pct(run, k, ref)


def pct_kernel(run):
    return f"{USER}/ingest-{run}-pct"


def push_pct(run, k, ref):
    """fill / pic / window: a CPU kernel (no GPU session) over the same shards + the detector dataset."""
    d = WORK / "kernel" / f"{run}-pct"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    (d / "run.py").write_text(FILLPIC.read_text())
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": pct_kernel(run), "title": pct_kernel(run).split("/")[1], "code_file": "run.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False, "keywords": [], "dataset_sources": [ref, DETECTOR],
        "kernel_sources": [], "competition_sources": [], "model_sources": []}))
    print(k.kernels_push(str(d)))


def status(run):
    k, out = api(), []
    for x in (kernel(run), pct_kernel(run)):
        try:
            out.append(str(k.kernels_status(x).status).lower())
        except Exception:                                  # a kernel of this run that was never pushed
            pass
    return out


def wait(run, hours=10):
    t0 = time.time()
    while True:
        st = " ".join(status(run))
        print(f"[{int((time.time() - t0) / 60)} min] {st}", flush=True)
        if all(any(x in s for x in ("complete", "error", "cancel")) for s in status(run)):
            return st.count("complete") == len(status(run))
        if time.time() - t0 > hours * 3600:
            raise SystemExit("time limit")
        time.sleep(CONFIG.kaggle.poll_seconds)


def fetch(run):
    out = WORK / "out" / run
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    for x in (kernel(run), pct_kernel(run)):
        subprocess.run([sys.executable, "-m", "kaggle", "kernels", "output", x, "-p", str(out)], check=False, timeout=3600)
    shards = sorted(p.stem for p in (WORK / "npz" / run).glob("pages_*.npz"))
    miss = [s for s in shards if not (out / "hw" / f"{s}.npz").exists() or not (out / f"scan_pct_{s}.npz").exists()]
    ok = (out / "embeddings.npy").exists() and (out / "ids.npy").exists() and not miss
    print("output", out, "| complete" if ok else f"| INCOMPLETE: missing {miss} / embeddings")
    return ok
