"""The detector: grey pages of the canvas -> window, ink tiles, picture share per tile, fill and pic."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import CONFIG
from .features import derive
from .maps import ink, page_maps
from .tiles import tile_cells, window

MODEL_FILES = ("ink.json", "pic.json", "cols.npy")


@dataclass(frozen=True)
class Result:
    window: tuple
    ink: np.ndarray
    pic: np.ndarray

    @property
    def fill(self):
        return float(self.ink.mean())

    @property
    def pic_mean(self):
        return float(self.pic.mean())


def blank(cfg=CONFIG):
    g = cfg.geometry.grid
    return Result((0.0, 0.0, 1.0, 1.0), np.zeros((g, g), bool), np.zeros((g, g), np.float32))


def fit(gray, cfg=CONFIG):
    """A grey page stretched to the canvas (no padding)."""
    h, w = cfg.geometry.canvas
    if gray.shape == (h, w):
        return gray
    return cv2.resize(
        gray, (w, h), interpolation=cv2.INTER_AREA if gray.shape[0] * gray.shape[1] > h * w else cv2.INTER_CUBIC
    )


def jpeg_roundtrip(gray, quality):
    return cv2.imdecode(cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, quality])[1], cv2.IMREAD_GRAYSCALE)


class Detector:
    def __init__(self, ink_model, pic_model, cols, cfg=CONFIG):
        self.ink_model, self.pic_model, self.cols, self.cfg = ink_model, pic_model, np.asarray(cols), cfg

    @classmethod
    def load(cls, folder, cfg=CONFIG, n_jobs=1):
        import xgboost as xgb

        folder = Path(folder)
        ink_model, pic_model = xgb.XGBClassifier(n_jobs=n_jobs), xgb.XGBRegressor(n_jobs=n_jobs)
        ink_model.load_model(str(folder / "ink.json"))
        pic_model.load_model(str(folder / "pic.json"))
        return cls(ink_model, pic_model, np.load(folder / "cols.npy"), cfg)

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.ink_model.save_model(str(folder / "ink.json"))
        self.pic_model.save_model(str(folder / "pic.json"))
        np.save(folder / "cols.npy", self.cols)

    def cells(self, gray, win=None):
        """(window in mask px, cells (1, G, G, M), scal (1, 2), mask shape) of a canvas page, or None without ink; `win` given: that window."""
        ink_ = ink(gray, self.cfg)
        win = win or window(ink_.mask, self.cfg.geometry.infer_margin, self.cfg)
        if win is None:
            return None
        maps, paper, noise = page_maps(gray, ink_, self.cfg)
        return (
            win,
            tile_cells(maps, win, gray.shape, self.cfg)[None],
            np.array([[paper, noise]], np.float32),
            ink_.mask.shape,
        )

    def tiles(self, X):
        """(ink bool, picture share) of feature rows (all columns of derive)."""
        X = X[:, self.cols]
        return self.ink_model.predict(X).astype(bool), np.clip(self.pic_model.predict(X), 0, 1)

    def predict(self, pages):
        """[Result] of grey canvas pages, the models predicting all pages together. A page without ink is blank (fill 0, pic 0, the whole canvas as window)."""
        g = self.cfg.geometry.grid
        got = [self.cells(p) for p in pages]
        ok = [i for i, r in enumerate(got) if r]
        out = [blank(self.cfg)] * len(pages)
        if ok:
            ink_, pic = self.tiles(np.concatenate([derive(got[i][1], got[i][2], self.cfg) for i in ok]))
            ink_, pic = ink_.reshape(len(ok), g, g), pic.reshape(len(ok), g, g).astype(np.float32)
            for k, i in enumerate(ok):
                (x, y, w, h), (mh, mw) = got[i][0], got[i][3]
                out[i] = Result((x / mw, y / mh, w / mw, h / mh), ink_[k], pic[k])
        return out
