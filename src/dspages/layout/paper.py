"""Paper, print and ageing effects on a flat page.

The page is carried as ink coverage a (0 paper .. 1 ink) until it is composited into luminance reflectance:
    lum = paper * texture * (1 - a (1 - ink / paper))
Each effect runs with its config probability and draws its parameters uniformly from the config ranges.
"""
import cv2
import numpy as np

from ..config import LayoutCfg, ShallowCreaseCfg
from ..creases import shallow
from ..noise import fbm, fbm_rotated, fleck_mask, scaled, set_scale, warp

YELLOW_LUMA = 0.24              # Rec. 709 luma of a yellow cast (0, 0.25, 0.8) per unit strength
STAIN_LUMA = 0.57               # of a brown stain (0.3, 0.6, 1.0)
BLUR_LEVELS = (0.0, 0.7, 1.4, 2.4, 3.6, 5.0)    # px, the pre-blurred levels of a variable blur


def _u(rng, r):
    return float(rng.uniform(*r)) if isinstance(r, tuple) else r


def smooth_noise(rng, shape, scale_px):
    """Zero-mean, unit-std Gaussian noise with correlation length ~scale_px, made on a coarse grid."""
    h, w = shape
    s = max(1.0, scaled(scale_px) / 4)
    small = rng.standard_normal((max(2, int(h / s) + 2), max(2, int(w / s) + 2)), dtype=np.float32)
    small = cv2.GaussianBlur(small, (0, 0), 2.0)
    n = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return (n - n.mean()) / (n.std() + 1e-6)


def _grow(box, pad, shape):
    """Box (y0, y1, x0, x1) grown by pad px on every side, within shape."""
    y0, y1, x0, x1 = box
    return max(y0 - pad, 0), min(y1 + pad, shape[0]), max(x0 - pad, 0), min(x1 + pad, shape[1])


def _blur(m, box, sigma):
    """GaussianBlur of m (zero outside box) computed around box only: the same values. -> blurred m, its box."""
    out = np.zeros_like(m)
    big = _grow(box, int(np.ceil(4 * sigma)) + 2, m.shape)
    sl = np.s_[big[0]:big[1], big[2]:big[3]]
    out[sl] = cv2.GaussianBlur(m[sl], (0, 0), sigma)
    return out, big


def _blob_mask(rng, shape, center, radius, harmonics=3):
    """Soft round blob with a wobbly outline, and the box (y0, y1, x0, x1) outside which it is zero."""
    th = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    r = np.ones_like(th)
    for k in range(1, harmonics + 1):
        r += rng.uniform(0.03, 0.12) / k * np.cos(k * th + rng.uniform(0, 2 * np.pi))
    pts = np.stack([center[0] + radius * r * np.cos(th), center[1] + radius * r * np.sin(th)], -1)
    m = np.zeros(shape, np.float32)
    p = np.round(pts).astype(np.int32)
    cv2.fillPoly(m, [p], 1.0)
    return _blur(m, _grow((p[:, 1].min(), p[:, 1].max() + 1, p[:, 0].min(), p[:, 0].max() + 1), 1, shape), max(1.0, 0.15 * radius))


def variable_blur(img, sigma_map, levels=BLUR_LEVELS):
    """Per-pixel Gaussian blur of sigma_map (px), linear between pre-blurred levels."""
    pos = np.interp(sigma_map, levels, np.arange(len(levels))).astype(np.float32)
    if img.ndim == 3:
        pos = pos[..., None]
    out = np.zeros_like(img)
    for k, s in enumerate(levels):
        wk = np.clip(1 - np.abs(pos - k), 0, 1)
        if wk.any():
            out += wk * (img if s == 0 else cv2.GaussianBlur(img, (0, 0), s))
    return out


def old_crease_map(rng, w_mm, h_mm, px, sc: ShallowCreaseCfg):
    """Height map [mm] of old creases at raster size px (w, h), from the shallow crease generator."""
    lines = shallow.sample(rng, w_mm, h_mm, sc.mean_groups, sc.p_isolated, sc.mean_extra, sc.radius_mm,
                           sc.spread_deg, sc.singles_per_group, sc.k)
    h = shallow.render(lines, w_mm, h_mm, sc.pitch_mm, sc.w_max_mm)
    return cv2.resize(h, px, interpolation=cv2.INTER_LINEAR), lines


def _logu(rng, r):
    return float(np.exp(rng.uniform(*np.log(r))))


CLOUD_STEP = 4                              # clouds (>= 2 mm) are evaluated on every 4th pixel


def paper_texture(rng, h, w, px_per_mm, p):
    """Zero-mean paper structure (h, w): fibres (fractal noise stretched along a random direction), formation
    clouds (fractal noise of random scale, octaves and roughness, sometimes domain-warped), sometimes recycled
    flecks (cellular noise), and pixel grain; every scale and amplitude drawn per page."""
    sd = int(rng.integers(1 << 30))
    th, k = rng.uniform(0, np.pi), _logu(rng, p["fiber_stretch"])
    t = _u(rng, p["fiber_amp"]) * fbm_rotated(sd, h, w, 1.0 / px_per_mm, th, k, _u(rng, p["fiber_mm"]), int(rng.integers(2, 4)))
    q = CLOUD_STEP
    cy, cx = (v.astype(np.float32) for v in np.meshgrid(np.arange(0, h, q) / px_per_mm, np.arange(0, w, q) / px_per_mm, indexing="ij"))
    if rng.random() < 0.4:
        cx, cy = warp(sd + 7, cx, cy, _logu(rng, p["cloud_mm"]), rng.uniform(1, 10))
    cloud = fbm(sd + 20, cx, cy, _logu(rng, p["cloud_mm"]), int(rng.integers(2, 7)), rng.uniform(0.35, 0.7))
    t += _u(rng, p["cloud_amp"]) * cv2.resize(cloud, (w, h), interpolation=cv2.INTER_CUBIC)
    if rng.random() < p["fleck_prob"]:
        t -= _u(rng, p["fleck_amp"]) * fleck_mask(sd + 40, h, w, _logu(rng, (0.5, 4.0)) * px_per_mm, rng.uniform(0.04, 0.12), rng.uniform(0.1, 0.5))
    return t + _u(rng, p["grain_amp"]) * rng.standard_normal((h, w), dtype=np.float32)


def apply(rng, gray, px_per_mm, c: LayoutCfg, sc: ShallowCreaseCfg):
    """uint8 gray page (0 ink .. 255 paper) -> uint8 luminance page and the names of the effects applied."""
    set_scale(rng.uniform(*c.noise_scale))
    fx = {k: e for k, e in c.effects.items() if rng.random() < e.prob}
    h, w = gray.shape
    a = 1.0 - gray.astype(np.float32) / 255.0

    if "print_spread" in fx:
        p = fx["print_spread"].params
        a = np.clip(cv2.GaussianBlur(a, (0, 0), _u(rng, p["sigma_px"])) * (1 + _u(rng, p["gain"])), 0, 1)
    if "ink_fade" in fx:
        p = fx["ink_fade"].params
        a *= 1 - _u(rng, p["strength"]) * np.clip(0.5 + 0.5 *
                                                  smooth_noise(rng, (h, w), _u(rng, p["scale_mm"]) * px_per_mm), 0, 1)
    if "toner_speckle" in fx:
        p = fx["toner_speckle"].params
        n = rng.poisson(_u(rng, p["density"]) * h * w)
        ys, xs = rng.integers(0, h, n), rng.integers(0, w, n)
        for y, x, r in zip(ys, xs, rng.integers(p["size_px"][0], p["size_px"][1] + 1, n)):
            cv2.circle(a, (int(x), int(y)), int(r) // 2, float(rng.uniform(0.3, 1.0)), -1)
    if "banding" in fx:
        p = fx["banding"].params
        period = _u(rng, p["period_mm"]) * px_per_mm
        yy = np.arange(h, dtype=np.float32)[:, None] if rng.random() < 0.5 else np.arange(w, dtype=np.float32)[None, :]
        a = np.clip(a * (1 - _u(rng, p["amp"]) * (0.5 + 0.5 *
                    np.sin(2 * np.pi * yy / period + rng.uniform(0, 6.3)))), 0, 1)

    creases = None
    if "old_creases" in fx:
        p = fx["old_creases"].params
        hmap, _ = old_crease_map(rng, w / px_per_mm, h / px_per_mm, (w, h), sc)
        gy, gx = np.gradient(hmap * px_per_mm)
        az = rng.uniform(0, 2 * np.pi)
        creases = (np.cos(az) * gx + np.sin(az) * gy, np.abs(hmap) / (np.abs(hmap).max() + 1e-6), p)
        a *= 1 - _u(rng, p["ink_loss"]) * creases[1] ** 4

    paper, ink = _u(rng, c.paper_tone), _u(rng, c.ink_tone)
    tex = np.ones((h, w), np.float32)
    if "paper_texture" in fx:
        tex += paper_texture(rng, h, w, px_per_mm, fx["paper_texture"].params)
    if "show_through" in fx:
        p = fx["show_through"].params
        back = cv2.GaussianBlur(a[:, ::-1], (0, 0), _u(rng, p["blur_px"]))
        tex -= _u(rng, p["strength"]) * back
    lum = paper * tex * (1 - a * (1 - ink / paper))

    if "yellowing" in fx:
        p = fx["yellowing"].params
        e = _u(rng, p["edge_mm"]) * px_per_mm
        yy, xx = np.arange(h), np.arange(w)
        dist = np.minimum.outer(np.minimum(yy, h - 1 - yy), np.minimum(xx, w - 1 - xx)).astype(np.float32)
        y = _u(rng, p["strength"]) * (0.4 + 0.6 * np.exp(-dist / e))
        lum = lum * (1 - YELLOW_LUMA * y)
    if "stains" in fx:
        p = fx["stains"].params
        for _ in range(rng.integers(p["n"][0], p["n"][1] + 1)):
            r = _u(rng, p["size_mm"]) * px_per_mm / 2
            m, box = _blob_mask(rng, (h, w), rng.uniform([0, 0], [w, h]), r)
            if rng.random() < 0.5:
                b, box = _blur(m, box, 0.15 * r)
                m = np.clip(m - b * 0.8, 0, 1)
            sl = np.s_[box[0]:box[1], box[2]:box[3]]
            lum[sl] = lum[sl] * (1 - STAIN_LUMA * _u(rng, p["strength"]) * m[sl])
    if creases is not None:
        shade, ridge, p = creases
        lum = lum * (1 + _u(rng, p["shade"]) * np.clip(shade, -1, 1)) * (1 - _u(rng, p["darken"]) * ridge)
    if "local_blur" in fx:
        p = fx["local_blur"].params
        smap, boxes = np.zeros((h, w), np.float32), []
        for _ in range(rng.integers(p["n"][0], p["n"][1] + 1)):
            sigma = _u(rng, p["sigma_px"])
            m, box = _blob_mask(rng, (h, w), rng.uniform([0, 0], [w, h]), _u(rng, p["size_mm"]) * px_per_mm / 2)
            smap = np.maximum(smap, sigma * m)
            boxes.append(box)
        y0, y1, x0, x1 = (f(b[k] for b in boxes) for k, f in enumerate((min, max, min, max)))
        big = _grow((y0, y1, x0, x1), int(np.ceil(4 * BLUR_LEVELS[-1])) + 2, (h, w))
        part = variable_blur(lum[big[0]:big[1], big[2]:big[3]], smap[big[0]:big[1], big[2]:big[3]])
        lum[y0:y1, x0:x1] = part[y0 - big[0]:y1 - big[0], x0 - big[2]:x1 - big[2]]
    return np.clip(lum * 255 + 0.5, 0, 255).astype(np.uint8), sorted(fx)
