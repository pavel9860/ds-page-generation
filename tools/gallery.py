"""Contact sheet of a run: per sample flat | warped | UV map | height map.

python tools/gallery.py RUN_DIR [--n 24] [--width 1600]
Writes RUN_DIR/report/gallery.jpg.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from dspages import io


def uv_image(uv):
    ok = np.isfinite(uv[..., 0])
    rgb = np.zeros((*uv.shape[:2], 3), np.uint8)
    rgb[ok, 0] = (255 * uv[ok, 0]).astype(np.uint8)
    rgb[ok, 1] = (255 * uv[ok, 1]).astype(np.uint8)
    rgb[ok, 2] = 128
    return rgb


def tile(arrays, h):
    flat, warped, uv, map3d = arrays["flat"], arrays["warped"], arrays["uv"].astype(np.float32), arrays["map3d"]
    parts = [np.repeat(flat[..., None], 3, -1), np.repeat(warped[..., None], 3, -1), uv_image(uv),
             io.height_preview(map3d[..., 2].astype(np.float32))]
    return np.hstack([cv2.resize(p, (round(h * p.shape[1] / p.shape[0]), h), interpolation=cv2.INTER_AREA)
                      for p in parts])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--row_h", type=int, default=300)
    a = ap.parse_args()
    run = Path(a.run)
    paths = sorted((run / "samples").glob("*.npz"))[:a.n]
    rows = [tile(io.load(p)[0], a.row_h) for p in paths]
    w = max(r.shape[1] for r in rows)
    sheet = np.vstack([np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows])
    (run / "report").mkdir(exist_ok=True)
    cv2.imwrite(str(run / "report" / "gallery.jpg"), sheet[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(run / "report" / "gallery.jpg")


if __name__ == "__main__":
    main()
