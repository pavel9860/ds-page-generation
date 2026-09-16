"""Shared multiprocess driver for materializing manifest entries to disk,
used by export_pages.py (100k .npz) and export_visual_sample.py (1k jpg)."""
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

from src.dataset.materialize import materialize_page

_OUT_DIR = None
_SAVE_FN = None


def _init(out_dir, save_fn):
    cv2.setNumThreads(1)
    global _OUT_DIR, _SAVE_FN
    _OUT_DIR = Path(out_dir)
    _SAVE_FN = save_fn


def _one(args):
    idx, entry = args
    page = materialize_page(entry)
    if page is None:
        return None
    _SAVE_FN(_OUT_DIR, idx, page)
    return idx


def run_export(jobs: list, out_dir: str, workers: int, save_fn, chunksize: int = 8,
               log_every: int = 2000):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    done = skipped = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(out_dir, save_fn)) as pool:
        for result in pool.map(_one, jobs, chunksize=chunksize):
            done += 1
            skipped += result is None
            if log_every and done % log_every == 0:
                el = time.time() - t0
                print(f"{done}/{len(jobs)} elapsed={el:.0f}s rate={done / el:.1f}/s "
                     f"eta={(len(jobs) - done) / (done / el) / 60:.1f}min", flush=True)
    print(f"done: n={len(jobs)} skipped={skipped} wall={time.time() - t0:.0f}s -> {out_dir}", flush=True)
