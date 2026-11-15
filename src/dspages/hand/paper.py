"""Writing paper: blank, lined, squared, dotted, legal, notebook with margin / holes / spiral. Every number is drawn per page.

paper(kind, rng) -> Paper(img float (H, W, 3) in [0, 1], pitch mm (baseline pitch the writer follows or None), first mm (y of
the first baseline), left / right / bottom mm (usable text window), kind)
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .config import FORMATS, LEGAL, RULE, TONES  # noqa: F401

W, H = 1024, 1440
K = W / 210.0  # px per mm
SIZE = (210.0, H / K)  # sheet [mm]; set by set_format
_R = [np.random.default_rng(0)]  # per page randomness of the printed lines
DRAW = True  # False: only the geometry (planning)

KINDS = ("blank", "lined", "squared", "dotted", "legal", "seyes", "margin", "spiral")


def set_format(fmt):
    global K, SIZE
    K = W / FORMATS[fmt]
    SIZE = (FORMATS[fmt], H / K)


@dataclass
class Paper:
    img: np.ndarray
    pitch: float
    first: float
    left: float
    right: float
    bottom: float
    kind: str
    meta: dict
    size: tuple = (210.0, 295.3)
    k: float = 4.876
    fmt: str = "a4"


def base(rng, tone=None):
    if not DRAW:
        return np.zeros((1, 1, 3), np.float32)
    t = np.array(tone if tone is not None else TONES[rng.integers(len(TONES))]) + rng.normal(0, 0.008, 3)
    fine = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), rng.uniform(0.8, 1.6))
    cloud = cv2.resize(
        rng.normal(0, 1, (H // 60 + 2, W // 60 + 2)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC
    )
    fiber = cv2.GaussianBlur(
        rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), sigmaX=rng.uniform(0.4, 1.0), sigmaY=rng.uniform(2.0, 6.0)
    )
    tex = fine * rng.uniform(0.002, 0.012) + cloud * rng.uniform(0.002, 0.012) + fiber * rng.uniform(0, 0.01)
    gx = np.linspace(-1, 1, W)[None]
    gy = np.linspace(-1, 1, H)[:, None]
    shade = 1 - rng.uniform(0, 0.05) * (np.cos(rng.uniform(0, 6.3)) * gx + np.sin(rng.uniform(0, 6.3)) * gy + 1) / 2
    return np.clip((t[None, None] + tex[..., None]) * shade[..., None], 0, 1).astype(np.float32)


def hline(img, y, x0, x1, col, th, a=0.9):
    """Anti-aliased faint line with a slight wobble of its tone, y in px."""
    if not DRAW:
        return
    h = img.shape[0]
    x1 = min(x1, img.shape[1])
    y0 = int(np.floor(y - th / 2))
    n = int(x1 - x0)
    if n <= 0:
        return
    wob = (
        1
        + 0.12
        * cv2.resize(
            _R[0].normal(0, 1, (1, max(n // 40, 2))).astype(np.float32), (n, 1), interpolation=cv2.INTER_CUBIC
        )[0]
    )
    for r in range(int(np.ceil(th)) + 1):
        yy = y0 + r
        if 0 <= yy < h:
            cover = np.clip(min(yy + 1, y + th / 2) - max(yy, y - th / 2), 0, 1)
            m = np.clip(cover * a * wob, 0, 1)[:, None]
            img[yy, int(x0) : int(x1)] = img[yy, int(x0) : int(x1)] * (1 - m) + np.asarray(col) * m


def vline(img, x, y0, y1, col, th, a=0.9):
    if not DRAW:
        return
    t = img.transpose(1, 0, 2)
    t2 = np.ascontiguousarray(t)
    hline(t2, x, y0, y1, col, th, a)
    img[:] = t2.transpose(1, 0, 2)


def jitter_col(rng, col):
    return np.clip(np.array(col) * rng.uniform(0.88, 1.08) + rng.normal(0, 0.02, 3), 0, 1)


def holes(img, rng, n, x_mm, r_mm, y0, y1):
    if not DRAW:
        return
    ys = np.linspace(y0, y1, n)
    for y in ys:
        cx, cy, r = int(x_mm * K + rng.normal(0, 1)), int(y + rng.normal(0, 1.5)), int(r_mm * K * 0.85)
        cv2.circle(img, (cx, cy), r + 2, (0.86, 0.86, 0.84), -1, cv2.LINE_AA)  # torn rim, lighter than the sheet shadow
        cv2.circle(img, (cx, cy), r, (0.17, 0.17, 0.19), -1, cv2.LINE_AA)  # the table through the hole


def paper(kind, rng, fmt="a4", dense=False):
    set_format(fmt)
    F = FORMATS[fmt] / 210.0  # margins and holes follow the sheet
    notebook = fmt != "a4"
    meta = {}
    tone = LEGAL[rng.integers(len(LEGAL))] if kind == "legal" else None
    img = base(np.random.default_rng(int(rng.integers(2**31))), tone)
    _R[0] = np.random.default_rng(int(rng.integers(2**31)))
    left, right, bottom, first, pitch = 14.0, 14.0, 14.0, 20.0, None
    rc_name = list(RULE)[rng.integers(len(RULE))]
    col = jitter_col(rng, RULE["gray"] if kind == "legal" and rng.random() < 0.3 else RULE[rc_name])
    th = rng.uniform(0.8, 1.7)
    a = rng.uniform(0.55, 0.95)
    off = rng.uniform(0, 3)  # the print is not aligned with the sheet edge
    if kind == "blank":
        left, right, bottom, first = (
            rng.uniform(5, 28) * F,
            rng.uniform(5, 25) * F,
            rng.uniform(6, 25) * F,
            rng.uniform(9, 35) * F,
        )
        pitch = None
    elif kind in ("lined", "margin", "spiral", "legal"):
        pitches = (5.5, 6.0, 6.5, 7.0, 7.1, 8.0, 8.7, 9.0, 9.5, 10.0)
        w = np.array(
            (0.06, 0.08, 0.08, 0.12, 0.12, 0.22, 0.12, 0.08, 0.06, 0.06)
            if not notebook
            else (0.10, 0.14, 0.12, 0.20, 0.14, 0.20, 0.06, 0.02, 0.01, 0.01)
        )
        if dense:  # dense writing is on narrow rulings
            w = w * np.array((1, 1, 1, 1, 1, 0, 0, 0, 0, 0))
        pitch = float(rng.choice(pitches, p=w / w.sum()))
        top = (rng.uniform(14, 32) if kind != "legal" else rng.uniform(22, 36)) * F
        first = top + pitch * rng.integers(0, 2) + off
        y = top + off
        while y * K < H - 8:
            hline(img, y * K, 0, W, col, th, a)
            y += pitch
        left, right = rng.uniform(8, 18) * F, rng.uniform(6, 15) * F
        bottom = rng.uniform(8, 20) * F
        if kind in ("margin", "legal") or rng.random() < 0.25:
            mx = rng.uniform(20, 34) * F
            mc = (0.86, 0.45, 0.45) if rng.random() < 0.8 else RULE["blue"]
            vline(img, mx * K, 0, H, mc, th, a)
            if rng.random() < 0.4:
                vline(img, (mx + 0.8) * K, 0, H, mc, th * 0.8, a * 0.8)
            left = mx + rng.uniform(1.5, 4)
            if kind == "legal":
                right = rng.uniform(6, 14)
        if kind == "legal":
            hline(img, (top - pitch) * K, 0, W, (0.75, 0.40, 0.40), th * 1.3, 0.8)
        if rng.random() < 0.3 and kind == "lined":
            hline(img, (top - pitch * rng.uniform(1.5, 2.5)) * K, 0, W, col, th, a)  # header rule
        meta["margin_line"] = kind in ("margin", "legal")
    elif kind in ("squared", "seyes"):
        step = (
            float(rng.choice((4.0, 5.0, 5.0, 5.0, 6.0, 7.0) if not dense else (4.0, 5.0, 5.0)))
            if kind == "squared"
            else 8.0
        )
        sub = step if kind == "squared" else 2.0
        th2 = th * rng.uniform(0.6, 0.9)
        bold5 = meta["bold5"] = bool(rng.random() < 0.3)
        x = off
        i = 0
        while x * K < W:
            heavy = kind == "squared" and bold5 and i % 5 == 0
            vline(img, x * K, 0, H, col, th2 * (1.5 if heavy else 1), a * (0.9 if kind == "squared" else 0.8))
            x += sub
            i += 1
        y = off * 0.7
        i = 0
        while y * K < H:
            heavy = kind == "squared" and bold5 and i % 5 == 0
            big = kind == "seyes" and i % 4 == 0
            hline(
                img,
                y * K,
                0,
                W,
                col,
                th2 * (1.5 if heavy else 1) * (1.2 if big else 1),
                a * (1 if big or kind == "squared" else 0.6),
            )
            y += sub
            i += 1
        pitch = step if kind == "squared" else 8.0
        if kind == "seyes":
            mx = rng.uniform(20, 30) * F
            vline(img, mx * K, 0, H, (0.86, 0.45, 0.45), th, a)
            left = mx + 2
        first = (
            round(rng.uniform(16, 30) * F / pitch) * pitch + off * 0.7
        )  # on a thick line of Seyès, on a grid line otherwise
        left = max(left, rng.uniform(8, 18) * F)
        meta["step"] = step
        if kind == "squared":
            pitch = (
                step if dense else step * (int(rng.choice((1, 2))) if step >= 7 else 2)
            )  # one text line per grid row, or every second row
    elif kind == "dotted":
        step = float(rng.choice((5.0, 5.0, 6.0, 4.0, 7.0)))
        y = rng.uniform(10, 16)
        while y * K < H:
            x = rng.uniform(6, 12)
            while x * K < W:
                DRAW and cv2.circle(
                    img, (int(x * K), int(y * K)), 1, tuple(float(c) for c in col * 0.75), -1, cv2.LINE_AA
                )
                x += step
            y += step
        pitch = step * (int(rng.choice((1, 2))) if step >= 6 else 2)
        first = 18 + pitch * rng.integers(0, 2)
        left, right, bottom = rng.uniform(8, 20) * F, rng.uniform(8, 18) * F, rng.uniform(8, 20) * F
        if dense:
            pitch = step  # one line of small writing per row of dots
    if kind == "spiral":
        n = int(SIZE[1] / rng.uniform(7.5, 10.0))
        side = "left" if rng.random() < 0.8 else "top"
        if side == "left":
            holes(img, rng, n, rng.uniform(4, 7), rng.uniform(1.6, 2.2), 30, H - 30)
            left += 4
        else:
            img = np.ascontiguousarray(img)
            for i in range(n):
                x = 25 + i * (W - 50) / (n - 1)
                cv2.circle(
                    img,
                    (int(x), int(rng.uniform(8, 13) * K / 2)),
                    int(1.8 * K / 2),
                    (0.40, 0.40, 0.42),
                    -1,
                    cv2.LINE_AA,
                )
            first += 4
        meta["spiral"] = side
    elif kind in ("lined", "margin", "squared", "legal") and rng.random() < 0.18:
        holes(img, rng, 3 if rng.random() < 0.4 else 2, rng.uniform(7, 10), rng.uniform(2.2, 3.2), H * 0.18, H * 0.82)
        left += 5
        meta["punched"] = True
    return Paper(
        img,
        pitch,
        first,
        left,
        right,
        bottom,
        kind,
        dict(meta, rule=rc_name, fmt=fmt, tone=img[img.shape[0] // 2, img.shape[1] // 2].round(2).tolist()),
        SIZE,
        K,
        fmt,
    )
