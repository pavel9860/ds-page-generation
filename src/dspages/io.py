"""Sample files: one .npz per sample and part (arrays + meta as JSON), a .jpg preview beside it."""
import json
from pathlib import Path

import cv2
import numpy as np


def _json(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def save(path: Path, arrays: dict, meta: dict, preview=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, meta=np.array(json.dumps(meta, default=_json)), **arrays)
    if preview is not None:
        cv2.imwrite(str(path.with_suffix(".jpg")), preview[..., ::-1] if preview.ndim == 3 else preview,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])


def load(path: Path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files if k != "meta"}, json.loads(str(f["meta"]))


def height_preview(Z):
    z = (Z - Z.min()) / max(float(np.ptp(Z)), 1e-6)
    return cv2.applyColorMap((255 * z).astype(np.uint8), cv2.COLORMAP_VIRIDIS)[..., ::-1]
