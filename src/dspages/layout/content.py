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


def _window_share(mask, win):
    """Share of mask inside every window position sliding along axis 0 (window spans axis 1)."""
    rows = np.concatenate([[0], np.cumsum(mask.sum(1))])
    return (rows[win:] - rows[:-win]) / (win * mask.shape[1])


def window_limits(bw, bh, aspect, max_margin):
    """Widths [lo, hi] of a window of aspect w/h around a content box bw x bh: lo contains all of it; hi keeps the
    margins <= max_margin of the window on the axis where the content is relatively narrower (the other axis is
    cropped when lo > hi)."""
    return max(bw, bh * aspect), min(bw, bh * aspect) / (1 - 2 * max_margin)


def place_window(rng, box, aspect, max_margin, min_w, mask, s, images, min_fill, grid, max_image):
    """Window (x, y, w, h) in page px of aspect w/h for a content box (x0, y0, bw, bh) in page px: width drawn in
    [max(lo, min_w), hi] when the content fits (all of it inside, margins <= max_margin), else hi, cropping the
    content along its relatively longer axis at a position with fill >= min_fill and picture share
    <= max_image (masks at scale s; images = picture boxes). The window may reach past the page: those parts are
    padded. -> window, fill, picture share."""
    x0, y0, bw, bh = box
    lo, hi = window_limits(bw, bh, aspect, max_margin)
    w = rng.uniform(max(lo, min_w), hi) if max(lo, min_w) <= hi else max(hi, min_w)
    h = w / aspect
    pos, fill, img = [], 1.0, 0.0
    for c0, cn, d, axis in ((x0, bw, w, 1), (y0, bh, h, 0)):
        if d >= cn:
            m = min(max_margin * d, d - cn)
            a = rng.uniform(max(0.0, d - cn - m), m) if d - cn <= 2 * m else (d - cn) / 2
            pos.append(c0 - a)
            continue
        m0, m1 = (round(v * s) for v in (y0, y0 + bh)) if axis == 0 else (round(v * s) for v in (x0, x0 + bw))
        other = (round(x0 * s), round((x0 + bw) * s)) if axis == 0 else (round(y0 * s), round((y0 + bh) * s))
        sub = mask[m0:m1, other[0]:other[1]] if axis == 0 else mask[other[0]:other[1], m0:m1].T
        isub = (images[m0:m1, other[0]:other[1]] if axis == 0 else images[other[0]:other[1], m0:m1].T)
        win = max(1, min(sub.shape[0], round(d * s)))
        f = window_fill(sub, win, grid)
        im = _window_share(isub, win)
        fits = im <= max_image
        ok = np.flatnonzero((f >= min_fill) & fits)
        pool = np.flatnonzero(fits) if fits.any() else np.arange(len(f))
        o = int(rng.choice(ok)) if ok.size else int(pool[f[pool].argmax()])
        pos.append(c0 + o / s)
        fill, img = float(f[o]), float(im[o])
    return (pos[0], pos[1], w, h), fill, img
