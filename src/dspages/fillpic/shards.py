"""Shards pages_*.npz: ids int32 (n, 2) [doc, page], offsets int64 (n + 1), jpeg uint8 (page i = jpeg[offsets[i]:offsets[i + 1]]); and the result files per shard."""

from pathlib import Path

import cv2
import numpy as np


def read(path):
    z = np.load(path)
    return z["ids"], z["offsets"], z["jpeg"]


def decode(jpeg, offsets, i):
    return cv2.imdecode(jpeg[offsets[i] : offsets[i + 1]], cv2.IMREAD_GRAYSCALE)


def result_name(shard):
    return f"scan_pct_{Path(shard).stem}.npz"


def write_result(path, ids, fill, pic, win):
    """ids (n, 2), fill / pic (n,), win (n, 4) shares of the canvas; NaN rows for pages that did not decode."""
    path = Path(path)
    np.savez(f"{path}.tmp.npz", ids=ids, fill=fill, pic=pic, win=win)
    Path(f"{path}.tmp.npz").rename(path)
