"""Materialize a manifest (src/dataset/build_manifest.py) into one
flat+3D+UV+warped .npz per page, split into train/val/test dirs, plus a
flat|warped|dewarped triptych PNG per page for spot-checking.
Pure geometric rendering: shading/blur/photometric effects/2D-and-3D
creases all off -- only the page's own bend/fold surface (make_surface's
BEND_PROB/FOLD_PROB) and the camera projection. Off-page background is
pure white (cfg.TEXT_BG_GRAY).

Split is by source_path (grouped, not per-page) so pages from the same
document never straddle splits, ~96/2/2 by page count.

Run:
    .venv/bin/python -m src.dataset.export_3d \\
        --manifest /run/media/me/D/ML_DS/UVTM/Layouts/manifest_100k.jsonl \\
        --out_dir /run/media/me/D/ML_DS/UVTM/Layouts/pages_100k_3d \\
        --vis_dir /run/media/me/D/ML_DS/UVTM/Layouts/pages_100k_3d_vis
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.dataset.build_manifest import _seed
from src.dataset.materialize import materialize_page
from src.synth import config as cfg
from src.synth.text_render import PAGE_PX, generate_text, rectify_backward
from scripts.export_uv_from_images import _make_triptych

SPLIT_FRACS = (("train", 0.96), ("val", 0.02), ("test", 0.02))


def _tex_from_matrix(gray_u8: np.ndarray) -> np.ndarray:
    """Raw pixel value as texture, 0..1 -- no INK/PAPER reflectance
    compression (that's the print-stage "paper-like" toning this export
    turns off), so a white source pixel warps to true white, not
    PAPER=0.93 gray."""
    return gray_u8.astype(np.float32) / 255.0


def _init():
    cv2.setNumThreads(1)


def _assign_splits(entries: list) -> list:
    """One deterministic split decision per unique source_path (hashed),
    not per page, so a document's pages never straddle train/val/test."""
    sources = sorted({e["source_path"] for e in entries})
    thresholds = []
    acc = 0.0
    for name, frac in SPLIT_FRACS:
        acc += frac
        thresholds.append((acc, name))
    split_of = {}
    for src in sources:
        r = (_seed(src) % 1_000_000) / 1_000_000
        split_of[src] = next(name for t, name in thresholds if r <= t)
    return [split_of[e["source_path"]] for e in entries]


def _one(args):
    idx, entry, split, out_dir, vis_dir = args
    flat = materialize_page(entry)
    tex = _tex_from_matrix(flat)
    seed = entry.get("text_seed", _seed(entry["source_path"]) + entry.get("page_index", 0))

    rgb, _rgb_lowres, uv_map, _flat_page, cam, zf, map3d = generate_text(
        seed, corpus_paths=None, font_files=None, flat_tex=tex,
        shade_enabled=False, blur_scale=0, add_creases=False, photometric_enabled=False)

    np.savez_compressed(
        Path(out_dir) / split / f"{idx:06d}.npz",
        original=flat,
        warped=rgb,
        uv=uv_map.astype(np.float16),
        map3d=map3d.astype(np.float16),
    )

    if vis_dir is not None:
        dewarped = rectify_backward(zf, cam, rgb, cfg.TEXT_CANVAS, page_px=PAGE_PX,
                                    bg_value=cfg.TEXT_BG_GRAY)
        cv2.imwrite(str(Path(vis_dir) / split / f"{idx:06d}.png"), _make_triptych(flat, rgb, dewarped))
    return split


def main(manifest_path: str, out_dir: str, workers: int, vis_dir: str = None):
    from concurrent.futures import ProcessPoolExecutor
    import time

    out_dir_p = Path(out_dir)
    vis_dir_p = Path(vis_dir) if vis_dir else None
    with open(manifest_path, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f]
    splits = _assign_splits(entries)
    for name, _ in SPLIT_FRACS:
        (out_dir_p / name).mkdir(parents=True, exist_ok=True)
        if vis_dir_p is not None:
            (vis_dir_p / name).mkdir(parents=True, exist_ok=True)

    jobs = [(idx, entry, split, out_dir, vis_dir) for idx, (entry, split) in enumerate(zip(entries, splits))
           if not (out_dir_p / split / f"{idx:06d}.npz").exists()]
    counts = {name: splits.count(name) for name, _ in SPLIT_FRACS}
    print(f"n={len(entries)} split_counts={counts} remaining={len(jobs)}", flush=True)

    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as pool:
        for _ in pool.map(_one, jobs, chunksize=8):
            done += 1
            if done % 2000 == 0:
                el = time.time() - t0
                print(f"{done}/{len(jobs)} elapsed={el:.0f}s rate={done / el:.1f}/s "
                     f"eta={(len(jobs) - done) / (done / el) / 60:.1f}min", flush=True)
    print(f"done: n={len(jobs)} wall={time.time() - t0:.0f}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--vis_dir", default=None, help="if given, also save a flat|warped|dewarped PNG per page")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    main(args.manifest, args.out_dir, args.workers, args.vis_dir)
