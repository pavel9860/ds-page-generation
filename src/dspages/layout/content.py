"""Content detection on a gray page and the choice of a content-filled window of any aspect ratio."""
import cv2
import numpy as np

SCAN_MAX_SIDE = 512
LINE_KSIZE = 15
LINE_THRESH = 36
PATCH_INK_EPS = 0.01


def content_mask(gray, max_side=SCAN_MAX_SIDE):
    """Ink (dark or light strokes on a local background) on a copy downscaled to max_side, and that scale."""
    s = min(1.0, max_side / max(gray.shape))
    small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (LINE_KSIZE, LINE_KSIZE))
    hat = cv2.max(cv2.morphologyEx(small, cv2.MORPH_BLACKHAT, k), cv2.morphologyEx(small, cv2.MORPH_TOPHAT, k))
    return hat > LINE_THRESH, s


def bbox(mask):
    """(y0, y1, x0, x1) inclusive of the true cells, or None."""
    rows, cols = mask.any(1), mask.any(0)
    if not rows.any():
        return None
    return (int(rows.argmax()), len(rows) - 1 - int(rows[::-1].argmax()),
            int(cols.argmax()), len(cols) - 1 - int(cols[::-1].argmax()))


def window_fill(mask, win, grid):
    """Fill of every window position sliding along axis 0 of mask (window spans axis 1 fully): the fraction
    of its grid x grid patches holding ink. win = window length along axis 0."""
    ii = cv2.integral(mask.astype(np.uint8)).astype(np.int64)
    n = mask.shape[0] - win + 1
    rows = np.arange(n)[:, None] + np.round(np.linspace(0, win, grid + 1)).astype(int)[None, :]
    cols = np.round(np.linspace(0, mask.shape[1], grid + 1)).astype(int)
    c = ii[rows[:, :, None], cols[None, None, :]]
    s = c[:, 1:, 1:] - c[:, :-1, 1:] - c[:, 1:, :-1] + c[:, :-1, :-1]
    area = np.diff(rows, axis=1)[:, :, None] * np.diff(cols)[None, None, :]
    return (s > PATCH_INK_EPS * area).mean(axis=(1, 2))


def select_window(mask, aspect, rng, min_fill, grid):
    """Largest window of aspect w/h inside mask's content bbox, placed along its free axis at a random
    position with fill >= min_fill (the best one if none reaches it). -> (y0, x0, h, w) in mask cells, fill."""
    b = bbox(mask)
    if b is None:
        return None, 0.0
    y0, y1, x0, x1 = b
    bh, bw = y1 - y0 + 1, x1 - x0 + 1
    sub = mask[y0:y1 + 1, x0:x1 + 1]
    tall = bw / bh < aspect
    win = max(1, min(bh, round(bw / aspect))) if tall else max(1, min(bw, round(bh * aspect)))
    fill = window_fill(sub if tall else sub.T, win, grid)
    ok = np.flatnonzero(fill >= min_fill)
    o = int(rng.choice(ok)) if ok.size else int(fill.argmax())
    if tall:
        return (y0 + o, x0, win, bw), float(fill[o])
    return (y0, x0 + o, bh, win), float(fill[o])
