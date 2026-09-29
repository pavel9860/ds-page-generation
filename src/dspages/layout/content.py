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


def window_limits(bw, bh, aspect, max_margin):
    """Widths [lo, hi] of a window of aspect w/h around a content box bw x bh: lo contains all of it; hi keeps the
    margins <= max_margin of the window on the axis where the content is relatively narrower (the other axis is
    cropped when lo > hi)."""
    return max(bw, bh * aspect), min(bw, bh * aspect) / (1 - 2 * max_margin)


WIDTHS = 6
POSITIONS = 9


def window_stats(ink, pics, s, wins, grid):
    """Fill (share of the window's grid x grid patches holding ink) and picture share of windows (k, 4) of
    (x, y, w, h) in page px; ink and pics are integral images of masks at scale s. Parts past the page count as
    empty. -> (fill (k,), share (k,))."""
    wins = np.asarray(wins, float).reshape(-1, 4) * s
    x, y, w, h = wins.T
    H, W = ink.shape[0] - 1, ink.shape[1] - 1
    t = np.linspace(0, 1, grid + 1)
    ys = np.round(y[:, None] + h[:, None] * t).astype(int)
    xs = np.round(x[:, None] + w[:, None] * t).astype(int)
    area = np.maximum(np.diff(ys)[:, :, None] * np.diff(xs)[:, None, :], 1)
    yc, xc = np.clip(ys, 0, H), np.clip(xs, 0, W)
    c = ink[yc[:, :, None], xc[:, None, :]]
    cells = c[:, 1:, 1:] - c[:, :-1, 1:] - c[:, 1:, :-1] + c[:, :-1, :-1]
    fill = (cells > PATCH_INK_EPS * area).mean(axis=(1, 2))
    y0, y1, x0, x1 = yc[:, 0], yc[:, -1], xc[:, 0], xc[:, -1]
    p = pics[y1, x1] - pics[y0, x1] - pics[y1, x0] + pics[y0, x0]
    return fill, p / np.maximum(w * h, 1.0)


def _axis(c0, cn, d, max_margin):
    """Window starts along one axis: around the content with margins <= max_margin when it fits, else crops."""
    if d >= cn:
        m = min(max_margin * d, d - cn)
        a = max(0.0, d - cn - m)
        return c0 - np.linspace(a, m, POSITIONS) if a <= m else np.array([c0 - (d - cn) / 2])
    return c0 + np.linspace(0, cn - d, POSITIONS)


def place_window(rng, box, aspect, max_margin, min_w, mask, s, pics, min_fill, grid, max_image):
    """Window (x, y, w, h) in page px of aspect w/h for a content box (x0, y0, bw, bh) in page px, drawn among the
    candidates (WIDTHS widths in [max(lo, min_w), hi], POSITIONS starts per cropped or margin axis) with fill >=
    min_fill and picture share <= max_image (masks at scale s). The window may reach past the page: those parts
    are padded and count as empty. -> window, fill, picture share, whether it meets both limits (else the
    candidate of highest fill among the lowest picture share)."""
    x0, y0, bw, bh = box
    lo, hi = window_limits(bw, bh, aspect, max_margin)
    a = max(lo, min_w)
    widths = np.linspace(a, hi, WIDTHS) if a <= hi else np.array([max(hi, min_w)])
    ink = cv2.integral(mask.astype(np.uint8)).astype(np.int64)
    pic = cv2.integral(pics.astype(np.uint8)).astype(np.int64)
    wins = np.array([(x, y, w, w / aspect) for w in widths for x in _axis(x0, bw, w, max_margin)
                     for y in _axis(y0, bh, w / aspect, max_margin)])
    fill, share = window_stats(ink, pic, s, wins, grid)
    ok = np.flatnonzero((fill >= min_fill) & (share <= max_image))
    k = int(ok[rng.integers(len(ok))]) if ok.size else int(np.lexsort((-fill, np.maximum(share - max_image, 0)))[0])
    return tuple(float(v) for v in wins[k]), float(fill[k]), float(share[k]), bool(ok.size)
