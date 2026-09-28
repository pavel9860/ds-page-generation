"""Pictures on a page: YOLO11n trained on DocLayNet, run as ONNX on the CPU."""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from ..config import PictureCfg

STRIDE = 32


@lru_cache(maxsize=2)
def _session(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"{path}: run python -m dspages.prep.layout_model")
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.log_severity_level = 3
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def pictures(gray, c: PictureCfg):
    """Picture boxes (x0, y0, x1, y1) in page px of a uint8 gray page. Input as in training: longest side imgsz,
    each side padded to a multiple of the stride (a square pad changes the predictions on portrait pages)."""
    s = _session(str(Path(c.model).expanduser()))
    h, w = gray.shape
    k = c.imgsz / max(h, w)
    nh, nw = round(h * k), round(w * k)
    H, W = (-(-v // STRIDE) * STRIDE for v in (nh, nw))
    x = np.full((H, W), 114, np.uint8)
    y0, x0 = (H - nh) // 2, (W - nw) // 2
    x[y0:y0 + nh, x0:x0 + nw] = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    out = s.run(None, {s.get_inputs()[0].name: np.repeat(x[None, None], 3, 1).astype(np.float32) / 255})[0][0]
    cls = out[4:]
    score = cls[c.picture_class]
    keep = np.flatnonzero((score >= c.conf) & (cls.argmax(0) == c.picture_class))
    if not keep.size:
        return np.zeros((0, 4))
    cx, cy, bw, bh = out[:4, keep]
    boxes = np.stack([cx - bw / 2 - x0, cy - bh / 2 - y0, bw, bh], -1) / k
    idx = cv2.dnn.NMSBoxes(boxes.tolist(), score[keep].tolist(), c.conf, c.iou)
    b = boxes[np.ravel(idx)]
    return np.clip(np.stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]], -1), 0, [w, h, w, h])


def picture_mask(gray, shape, s, c: PictureCfg):
    """Picture boxes of the page rasterized on a mask of shape at scale s."""
    m = np.zeros(shape, bool)
    for x0, y0, x1, y1 in pictures(gray, c) * s:
        m[int(y0):int(np.ceil(y1)), int(x0):int(np.ceil(x1))] = True
    return m
