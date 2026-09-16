"""Materialize a manifest (src/dataset/build_manifest.py) into one flat
1024x1024 uint8 grayscale .npz per page (key "page"), multiprocessed.

Run:
    .venv/bin/python -m src.dataset.export_pages \\
        --manifest /run/media/me/D/ML_DS/UVTM/Layouts/manifest_100k.jsonl \\
        --out_dir /run/media/me/D/ML_DS/UVTM/Layouts/pages_100k_npz
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.dataset.export_common import run_export


def _save_npz(out_dir: Path, idx: int, page: np.ndarray):
    np.savez_compressed(out_dir / f"{idx:06d}.npz", page=page)


def main(manifest_path: str, out_dir: str, workers: int):
    out_dir_p = Path(out_dir)
    with open(manifest_path, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f]
    jobs = [(idx, entry) for idx, entry in enumerate(entries)
           if not (out_dir_p / f"{idx:06d}.npz").exists()]
    print(f"n={len(entries)} already_done={len(entries) - len(jobs)} remaining={len(jobs)}", flush=True)
    run_export(jobs, out_dir, workers, _save_npz)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    main(args.manifest, args.out_dir, args.workers)
