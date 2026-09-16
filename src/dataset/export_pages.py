"""Materialize a manifest (src/dataset/build_manifest.py) into one flat
1024x1024 uint8 grayscale .npz per page (key "page"), multiprocessed.

Run:
    .venv/bin/python -m src.dataset.export_pages \\
        --manifest /run/media/me/D/ML_DS/UVTM/Layouts/manifest_100k.jsonl \\
        --out_dir /run/media/me/D/ML_DS/UVTM/Layouts/pages_100k_npz
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from src.dataset.materialize import materialize_page

_OUT_DIR = None


def _init(out_dir):
    cv2.setNumThreads(1)
    global _OUT_DIR
    _OUT_DIR = Path(out_dir)


def _one(args):
    idx, entry = args
    page = materialize_page(entry)
    np.savez_compressed(_OUT_DIR / f"{idx:06d}.npz", page=page)
    return idx


def main(manifest_path: str, out_dir: str, workers: int):
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f]
    jobs = [(idx, entry) for idx, entry in enumerate(entries)
           if not (out_dir_p / f"{idx:06d}.npz").exists()]
    print(f"n={len(entries)} already_done={len(entries) - len(jobs)} remaining={len(jobs)}", flush=True)

    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(out_dir,)) as pool:
        for _ in pool.map(_one, jobs, chunksize=8):
            done += 1
            if done % 2000 == 0:
                el = time.time() - t0
                print(f"{done}/{len(jobs)} elapsed={el:.0f}s rate={done / el:.1f}/s "
                     f"eta={(len(jobs) - done) / (done / el) / 60:.1f}min", flush=True)
    print(f"done: n={len(entries)} wall={time.time() - t0:.0f}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    main(args.manifest, args.out_dir, args.workers)
