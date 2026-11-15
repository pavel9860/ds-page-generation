"""Pictures on a page: YOLO11n trained on DocLayNet, run as ONNX on the CPU, and filled areas (artwork, full-page
illustrations, dark blocks the detector does not know)."""

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from ..config import PictureCfg
from .config import STRIDE  # noqa: F401


@lru_cache(maxsize=2)
def _session(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"{path}: run python -m dspages.prep.layout_model")
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.log_severity_level = 3
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def pictures(img, c: PictureCfg):
    """Picture boxes (x0, y0, x1, y1) in page px of a uint8 RGB page. Input as in training: longest side imgsz,
    each side padded to a multiple of the stride (a square pad changes the predictions on portrait pages). Boxes
    over large_area of the page are kept only at large_conf."""
    s = _session(str(Path(c.model).expanduser()))
    h, w = img.shape[:2]
    k = c.imgsz / max(h, w)
    nh, nw = round(h * k), round(w * k)
    H, W = (-(-v // STRIDE) * STRIDE for v in (nh, nw))
    x = np.full((H, W, 3), 114, np.uint8)
    y0, x0 = (H - nh) // 2, (W - nw) // 2
    x[y0 : y0 + nh, x0 : x0 + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = s.run(None, {s.get_inputs()[0].name: x.transpose(2, 0, 1)[None].astype(np.float32) / 255})[0][0]
    cls = out[4:]
    score = cls[c.picture_class]
    keep = np.flatnonzero((score >= c.conf) & (cls.argmax(0) == c.picture_class))
    if not keep.size:
        return np.zeros((0, 4))
    cx, cy, bw, bh = out[:4, keep]
    boxes = np.stack([cx - bw / 2 - x0, cy - bh / 2 - y0, bw, bh], -1) / k
    idx = cv2.dnn.NMSBoxes(boxes.tolist(), score[keep].tolist(), c.conf, c.iou)
    idx = np.ravel(idx)
    b, sc = boxes[idx], score[keep][idx]
    b = b[(sc >= c.large_conf) | (b[:, 2] * b[:, 3] <= c.large_area * h * w)]
    return np.clip(np.stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]], -1), 0, [w, h, w, h])


def picture_mask(boxes, shape, s):
    """Picture boxes (x0, y0, x1, y1) in page px rasterized on a mask of shape at scale s."""
    m = np.zeros(shape, bool)
    for x0, y0, x1, y1 in np.asarray(boxes, float).reshape(-1, 4) * s:
        m[int(y0) : int(np.ceil(y1)), int(x0) : int(np.ceil(x1))] = True
    return m


def picture_area(img, shape, c: PictureCfg):
    """Mask of shape (a reduced copy of the RGB page img) over its pictures: the detector's boxes and the filled
    patches."""
    h, w = shape
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    ink, paper = np.percentile(g, [1, 90])
    dark = (g < ink + c.fill_paper * (paper - ink)).astype(np.float32)
    cells = cv2.resize(dark, (max(w // c.fill_patch, 1), max(h // c.fill_patch, 1)), interpolation=cv2.INTER_AREA)
    filled = cv2.resize((cells > c.fill_need).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return picture_mask(pictures(img, c), shape, h / img.shape[0]) | filled
