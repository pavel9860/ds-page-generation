"""Pages that are pictures without any text or handwriting"""

import json
from pathlib import Path

import cv2
import numpy as np
import pymupdf

from .config import Config


def load(path, page, long_side):
    if str(path).lower().endswith(".pdf"):
        with pymupdf.open(path) as doc:
            pg = doc[page]
            k = long_side / max(pg.rect.width, pg.rect.height)
            px = pg.get_pixmap(matrix=pymupdf.Matrix(k, k), alpha=False)
        return np.frombuffer(px.samples, np.uint8).reshape(px.h, px.w, px.n)[:, :, :3].copy()
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    k = long_side / max(img.shape[:2])
    return cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)


def build(cfg: Config):
    """Keeps the photo items in which the detector (cfg.picture_model) finds no print and no handwriting."""
    from ultralytics import YOLO

    cfg.pictures.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(cfg.picture_model), task="detect")
    for line in cfg.picture_items.read_text().splitlines():
        r = json.loads(line)
        out = cfg.pictures / f"{Path(r['path']).stem[:60]}_p{r.get('page', 0)}.png"
        if r["class"] != "photo" or out.exists():
            continue
        try:
            img = load(r["path"], int(r.get("page", 0)), cfg.long_side)
        except Exception:
            continue
        if min(img.shape[:2]) >= 400 and not len(
            model.predict(img, imgsz=cfg.long_side, conf=cfg.picture_conf, verbose=False, device="cpu")[0].boxes
        ):
            cv2.imwrite(str(out), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return len(list(cfg.pictures.glob("*.png")))
