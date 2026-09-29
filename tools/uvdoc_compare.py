"""Our run against UVDoc: formats and sizes, and the same UV round trip on both (warped -> flat through the UV map
-> warped again, compared with the warped image inside the page: intensity RMSE and phase-correlation shift in px).

python tools/uvdoc_compare.py RUN_DIR UVDOC_DIR [--n 100] [--workers 6]
UVDoc's .mat files are MATLAB v7.3 (HDF5): h5py is needed.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from dspages import io
from uv_check import TILE, WINDOW
from viz import back_map


def round_trip(warped, uv, page_hw):
    """-> (RMSE 0..255, RMS tile shift px) of the warped image against itself mapped to the flat and back."""
    ph, pw = page_hw
    back = np.nan_to_num(back_map(warped, uv, page_hw), nan=0.0)
    full = cv2.resize(np.nan_to_num(uv, nan=-1.0), warped.shape[1::-1], interpolation=cv2.INTER_LINEAR)
    ok = cv2.resize(np.isfinite(uv[..., 0]).astype(np.uint8), warped.shape[1::-1], interpolation=cv2.INTER_NEAREST)
    ok = cv2.erode(ok, np.ones((5, 5), np.uint8)) > 0
    fw = cv2.remap(back, full[..., 0] * (pw - 1), full[..., 1] * (ph - 1), cv2.INTER_LINEAR)
    w = warped.astype(np.float32)
    rmse = float(np.sqrt(np.mean((fw[ok] - w[ok]) ** 2)))
    s = []
    for y in range(0, w.shape[0] - TILE + 1, TILE):
        for x in range(0, w.shape[1] - TILE + 1, TILE):
            if ok[y:y + TILE, x:x + TILE].all() and w[y:y + TILE, x:x + TILE].std() > 10:
                (dx, dy), r = cv2.phaseCorrelate(w[y:y + TILE, x:x + TILE], fw[y:y + TILE, x:x + TILE], WINDOW)
                if r > 0.1:
                    s.append(np.hypot(dx, dy))
    return rmse, float(np.sqrt(np.mean(np.square(s)))) if s else None


def ours(path):
    a, _ = io.load(path)
    page = io.load(path.parents[1] / "layouts" / path.name)[0]["page"]
    return round_trip(a["warped"], a["uv"].astype(np.float32), page.shape)


def uvdoc(args):
    import h5py
    root, sid = args
    m = json.loads((root / "metadata_sample" / f"{sid}.json").read_text())
    w = cv2.imread(str(root / "warped_textures" / f"{sid}.png"), cv2.IMREAD_GRAYSCALE)
    with h5py.File(root / "uvmap" / f"{m['geom_name']}.mat") as f:
        uv = np.array(f["uv"]).transpose(2, 1, 0).astype(np.float32)
    uv = cv2.resize(uv, (uv.shape[1] * 300 // uv.shape[0], 300), interpolation=cv2.INTER_NEAREST)
    return round_trip(w, uv, w.shape)


def summary(rows):
    r = np.array([x[0] for x in rows])
    s = np.array([x[1] for x in rows if x[1] is not None])
    return dict(n=len(rows), rmse_median=float(np.median(r)), rmse_p90=float(np.percentile(r, 90)),
                shift_px_median=float(np.median(s)), shift_px_p90=float(np.percentile(s, 90)), shift_px_max=float(s.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("uvdoc")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    run, root = Path(a.run), Path(a.uvdoc)
    ids = sorted(p.stem for p in (root / "warped_textures").glob("*.png"))
    ids = [ids[i] for i in np.random.default_rng(0).choice(len(ids), a.n, replace=False)]
    with ProcessPoolExecutor(a.workers) as pool:
        res = dict(ours=summary(list(pool.map(ours, sorted((run / "samples").glob("*.npz"))[:a.n]))),
                   uvdoc=summary(list(pool.map(uvdoc, [(root, i) for i in ids]))))
    out = run.with_name(run.name + "_viz") / "uvdoc_compare.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
