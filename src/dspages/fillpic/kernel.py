"""Fill / pic / window of every page of the shards pages_*.npz of Kaggle datasets, on the CPUs of one script kernel. Standalone: python kernel.py [DATA_DIR] [OUT_DIR]

DATA_DIR (default /kaggle/input) holds the shard datasets and the detector dataset (modules of fillpic, ink.json, pic.json, cols.npy; DETECTOR_DIR overrides where it is
searched; the modules sit flat in it and are assembled into package fillpic at start). Output (OUT_DIR, default /kaggle/working), one file per shard, resumable: scan_pct_<shard>.npz with ids (n, 2), fill (n,), pic (n,), win (n, 4) [x, y, w, h as
shares of the canvas]; NaN rows: the JPEG did not decode. A page without ink is fill 0, pic 0 with the whole canvas, 0 0 1 1, as its window. One worker process per CPU
takes whole shards; no shard is started after HARD_STOP_H hours; a worker that dies raises here instead of hanging the run."""
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/kaggle/input")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/kaggle/working")
ROOT = os.environ.get("DETECTOR_DIR") or str(Path(next(glob.iglob(str(DATA / "**" / "detector.py"), recursive=True))).parent)
PKG = Path("/tmp/pkg/fillpic")
PKG.mkdir(parents=True, exist_ok=True)
for f in Path(ROOT).glob("*.py"):
    (PKG / f.name).write_bytes(f.read_bytes())
sys.path.insert(0, str(PKG.parent))
from fillpic import CONFIG, Detector, shards  # noqa: E402   (sets one BLAS thread before numpy is imported)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

LIMIT = int(os.environ.get("LIMIT", 0))
ONLY = [x for x in os.environ.get("ONLY", "").split(",") if x]
HARD_STOP_H = float(os.environ.get("HARD_STOP_H", CONFIG.kaggle.hard_stop_hours))
WORKERS = int(os.environ.get("WORKERS", os.cpu_count() or 4))
T0 = time.time()
DET = None


def init():
    global DET
    cv2.setNumThreads(1)
    DET = Detector.load(ROOT)


def do_shard(path):
    out = OUT / shards.result_name(path)
    if out.exists() or time.time() - T0 > HARD_STOP_H * 3600:
        return path, 0, 0.0, 0.0
    t = time.time()
    ids, off, jpeg = shards.read(path)
    t_read = time.time() - t
    n = min(LIMIT, len(ids)) if LIMIT else len(ids)
    fill, pic, win = np.full(n, np.nan, np.float32), np.full(n, np.nan, np.float32), np.full((n, 4), np.nan, np.float32)
    batch = CONFIG.kaggle.batch
    for a in range(0, n, batch):
        idx, pages = [], []
        for i in range(a, min(a + batch, n)):
            g = shards.decode(jpeg, off, i)
            if g is not None:
                idx.append(i)
                pages.append(g)
        for i, r in zip(idx, DET.predict(pages)):
            win[i], fill[i], pic[i] = r.window, r.fill, r.pic_mean
    shards.write_result(out, ids[:n], fill, pic, win)
    return path, n, time.time() - t, t_read


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [s for s in sorted(glob.glob(str(DATA / "**" / "pages_*.npz"), recursive=True)) if not ONLY or Path(s).stem in ONLY]
    print(len(todo), "shards,", WORKERS, "workers, detector in", ROOT, flush=True)
    done = pages = 0
    with ProcessPoolExecutor(WORKERS, initializer=init) as pool:
        for path, n, sec, rd in pool.map(do_shard, todo):
            done += 1
            pages += n
            print(f"[{int((time.time() - T0) / 60)} min] {Path(path).name}: {n} pages, {sec:.0f} s (read {rd:.0f} s) ({done}/{len(todo)} shards, {pages} pages)", flush=True)
    print("not done (time limit):", [Path(s).name for s in todo if not (OUT / shards.result_name(s)).exists()], flush=True)
