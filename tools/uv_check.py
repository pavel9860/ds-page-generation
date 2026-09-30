"""UV map check of a run: the warped image mapped back through its UV map against the flat page.

python tools/uv_check.py RUN_DIR [--workers 6]
Per sample: seen share of the page, intensity RMSE (0..255) raw and after a least-squares gain/offset (the warped
image carries lighting and camera effects), and the local misalignment in px: phase correlation shifts of
TILE x TILE px tiles with ink. Writes RUN_DIR_viz/uv_check.json and prints the summary.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from dspages import io
from viz import back_map

TILE = 128
WINDOW = cv2.createHanningWindow((TILE, TILE), cv2.CV_32F)


def check(path):
    a, _ = io.load(path)
    flat = a["flat"].astype(np.float32)
    page_hw = flat.shape
    back = back_map(a["warped"], a["uv"].astype(np.float32), page_hw)
    ok = np.isfinite(back)
    f, b = flat[ok], back[ok]
    A = np.c_[b, np.ones_like(b)]
    fit = A @ np.linalg.lstsq(A, f, rcond=None)[0]
    shifts = []
    for y in range(0, page_hw[0] - TILE + 1, TILE):
        for x in range(0, page_hw[1] - TILE + 1, TILE):
            t_ok = ok[y:y + TILE, x:x + TILE]
            tf = flat[y:y + TILE, x:x + TILE]
            if not t_ok.all() or tf.std() < 10:
                continue
            (dx, dy), resp = cv2.phaseCorrelate(tf, back[y:y + TILE, x:x + TILE], WINDOW)
            if resp > 0.1:
                shifts.append(np.hypot(dx, dy))
    s = np.array(shifts)
    return dict(sample=path.stem, seen=float(ok.mean()), rmse=float(np.sqrt(np.mean((f - b) ** 2))),
                rmse_gain=float(np.sqrt(np.mean((f - fit) ** 2))), tiles=len(s),
                shift_rms_px=float(np.sqrt(np.mean(s ** 2))) if len(s) else None,
                shift_max_px=float(s.max()) if len(s) else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    run = Path(a.run)
    with ProcessPoolExecutor(a.workers) as pool:
        rows = list(pool.map(check, sorted((run / "samples").glob("*.npz")), chunksize=2))
    out = run.with_name(run.name + "_viz") / "uv_check.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))

    def q(k):
        v = np.array([r[k] for r in rows if r[k] is not None])
        return f"{k}: median {np.median(v):.2f}  p90 {np.percentile(v, 90):.2f}  max {v.max():.2f}"
    print(len(rows), "samples")
    for k in ("seen", "rmse", "rmse_gain", "shift_rms_px", "shift_max_px"):
        print(q(k))
    print(out)


if __name__ == "__main__":
    main()
