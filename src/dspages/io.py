"""Sample files: one .npz per sample and part (arrays, + meta as JSON for the parts), a .jpg preview where asked."""
import json
import os
import zipfile
from pathlib import Path

import cv2
import numpy as np


def _json(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def save(path: Path, arrays: dict, meta: dict = None, preview=None, preview_path: Path = None):
    """meta (None: none stored) as a JSON member, written whole or not at all (a temp file renamed); preview is written to
    preview_path (default: beside the .npz)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    items = arrays if meta is None else dict(meta=np.array(json.dumps(meta, default=_json)), **arrays)
    tmp = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:    # np.load-able .npz; level 1:
        for k, v in items.items():                                                  # 2x faster, +5%
            with z.open(k + ".npy", "w", force_zip64=True) as f:
                np.lib.format.write_array(f, np.asarray(v))
    os.replace(tmp, path)
    if preview is not None:
        preview_path = Path(preview_path or path.with_suffix(".jpg"))
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(preview_path), preview[..., ::-1] if preview.ndim == 3 else preview,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])


def load(path: Path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files if k != "meta"}, json.loads(str(f["meta"])) if "meta" in f.files else {}


def height_preview(Z):
    z = (Z - Z.min()) / max(float(np.ptp(Z)), 1e-6)
    return cv2.applyColorMap((255 * z).astype(np.uint8), cv2.COLORMAP_VIRIDIS)[..., ::-1]
