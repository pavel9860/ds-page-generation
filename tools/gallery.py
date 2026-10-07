"""Contact sheets of a run: per sample flat | warped | dewarped (the warped image mapped back through the UV map) | UV map | 3D map.

python tools/gallery.py RUN_DIR [--n 200] [--per_sheet 10] [--row_h 300] [--workers 12]
Writes RUN_DIR/report/gallery_<k>.jpg, per_sheet samples each.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from dspages import io
from viz import back_map


def uv_image(grid2d):
    rgb = np.full((*grid2d.shape[1:], 3), 128, np.uint8)
    rgb[..., :2] = np.moveaxis(127.5 * (grid2d + 1), 0, -1).astype(np.uint8)
    return rgb


def xyz_image(xyz):
    """The 3D map as colour: X, Y, Z each scaled over the page to 0..255."""
    lo, hi = np.nanmin(xyz, (0, 1)), np.nanmax(xyz, (0, 1))
    return (255 * (xyz - lo) / np.maximum(hi - lo, 1e-9)).astype(np.uint8)


def gray3(g):
    return np.repeat(np.nan_to_num(g, nan=255).astype(np.uint8)[..., None], 3, -1)


def row(path, h):
    a = io.load(path)[0]
    g2 = a["grid2d"].astype(np.float32)
    parts = [gray3(a["flat"]), gray3(a["warped"]), gray3(back_map(a["warped"], g2, a["flat"].shape)), uv_image(g2),
             xyz_image(np.moveaxis(a["grid3d"], 0, -1).astype(np.float32))]
    return np.hstack([cv2.resize(p, (round(h * p.shape[1] / p.shape[0]), h), interpolation=cv2.INTER_AREA) for p in parts])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--per_sheet", type=int, default=10)
    ap.add_argument("--row_h", type=int, default=300)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    run = Path(a.run)
    paths = sorted((run / "samples").glob("*.npz"))[:a.n]
    with ProcessPoolExecutor(a.workers) as pool:
        rows = list(pool.map(row, paths, [a.row_h] * len(paths), chunksize=2))
    (run / "report").mkdir(exist_ok=True)
    for k in range(0, len(rows), a.per_sheet):
        part = rows[k:k + a.per_sheet]
        w = max(r.shape[1] for r in part)
        sheet = np.vstack([np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in part])
        cv2.imwrite(str(run / "report" / f"gallery_{k // a.per_sheet:02d}.jpg"), sheet[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(run / "report")


if __name__ == "__main__":
    main()
