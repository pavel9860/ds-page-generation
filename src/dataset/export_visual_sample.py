"""Randomly sample entries from a finished manifest and render each as a
plain .jpg for eyeballing only -- not part of the data pipeline (the 100k
set stays .npz, see export_pages.py); this is a disposable spot-check.

Run:
    .venv/bin/python -m src.dataset.export_visual_sample \\
        --manifest /run/media/me/D/ML_DS/UVTM/Layouts/manifest_100k.jsonl \\
        --out_dir /run/media/me/D/ML_DS/UVTM/Layouts/manifest_100k_vis_1k --n 1000 --seed 0
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
    cv2.imwrite(str(_OUT_DIR / f"{idx:05d}.jpg"), page)
    return idx


def main(manifest_path: str, out_dir: str, n: int, seed: int, workers: int):
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f]

    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(len(entries), size=min(n, len(entries)), replace=False)
    sampled = [entries[i] for i in sample_idx]
    print(f"sampling {len(sampled)}/{len(entries)}", flush=True)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(out_dir,)) as pool:
        for _ in pool.map(_one, enumerate(sampled), chunksize=4):
            pass
    print(f"done: n={len(sampled)} wall={time.time() - t0:.0f}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    main(args.manifest, args.out_dir, args.n, args.seed, args.workers)
