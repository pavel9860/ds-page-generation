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
from pathlib import Path

import cv2
import numpy as np

from src.dataset.export_common import run_export


def _save_jpg(out_dir: Path, idx: int, page: np.ndarray):
    cv2.imwrite(str(out_dir / f"{idx:05d}.jpg"), page)


def main(manifest_path: str, out_dir: str, n: int, seed: int, workers: int):
    with open(manifest_path, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f]

    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(len(entries), size=min(n, len(entries)), replace=False)
    jobs = list(enumerate(entries[i] for i in sample_idx))
    print(f"sampling {len(jobs)}/{len(entries)}", flush=True)
    run_export(jobs, out_dir, workers, _save_jpg, chunksize=4, log_every=0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    main(args.manifest, args.out_dir, args.n, args.seed, args.workers)
