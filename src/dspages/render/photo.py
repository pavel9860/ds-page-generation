"""Light, shadows, background and camera effects of the photo.

Light: a point light (sharp shadows) or an area light sampled at AREA_SAMPLES points (soft shading and shadows)
above the table. Per point: ambient + (1 - ambient) * mean over light samples of visibility * (lambert + specular);
visibility from a depth map rendered from each light sample, so the page shadows itself and the table.
Everything is single-channel luminance. Camera effects run on linear luminance in [0, 1]: depth-of-field blur,
motion blur, vignetting, exposure and gamma, sensor noise at an ISO, JPEG.
"""
import cv2
import numpy as np
from numba import njit

from ..config import LightCfg, RenderCfg
from ..layout.paper import variable_blur
from ..noise import Field, _lattice, value_noise
from .camera import look_at, project
from .raster import rasterize

AREA_SAMPLES = 6
SHADOW_PX = 384
SHADOW_BIAS_MM = 0.6


def _u(rng, r):
    return float(rng.uniform(*r))


def sample_light(rng, c: LightCfg, target):
    sharp = rng.random() < c.sharp_prob
    el, az, d = np.radians(_u(rng, c.elev_deg)), rng.uniform(0, 2 * np.pi), _u(rng, c.dist_mm)
    centre = target + d * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    if sharp:
        pts = centre[None]
    else:
        r = 0.5 * _u(rng, c.area_mm)
        a = np.linspace(0, 2 * np.pi, AREA_SAMPLES, endpoint=False) + rng.uniform(0, 2 * np.pi)
        rad = r * np.sqrt(rng.uniform(0.2, 1, (AREA_SAMPLES, 1)))
        pts = centre + rad * np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], -1)
    return dict(points=pts, sharp=bool(sharp), ambient=_u(rng, c.ambient), specular=_u(rng, c.specular),
                shininess=_u(rng, c.shininess))


def shadow_maps(light, P, faces, target):
    maps = []
    for L in light["points"]:
        cam = dict(eye=L, R=look_at(L, target, 0.0), f=1.0, cx=0.0, cy=0.0)
        uv, _ = project(P, cam)
        lo, hi = uv.min(0), uv.max(0)
        f = 0.98 * SHADOW_PX / (hi - lo).max()
        cam.update(f=f, cx=SHADOW_PX / 2 - f * (lo[0] + hi[0]) / 2, cy=SHADOW_PX / 2 - f * (lo[1] + hi[1]) / 2)
        pix, z = project(P, cam)
        _, _, depth = rasterize(pix, z, np.zeros((len(P), 3)), faces, (SHADOW_PX, SHADOW_PX))
        maps.append((cam, depth))
    return maps


@njit(cache=True, fastmath=True)
def _shade(Q, N, eye, pts, Rs, cams, depths, ambient, specular, shininess, bias):
    n, m, s = Q.shape[0], pts.shape[0], depths.shape[1]
    out = np.empty(n, np.float32)
    for i in range(n):
        nx, ny, nz = N[i, 0], N[i, 1], N[i, 2]
        vx, vy, vz = eye[0] - Q[i, 0], eye[1] - Q[i, 1], eye[2] - Q[i, 2]
        vn = np.sqrt(vx * vx + vy * vy + vz * vz)
        vx, vy, vz = vx / vn, vy / vn, vz / vn
        acc = 0.0
        for k in range(m):
            lx, ly, lz = pts[k, 0] - Q[i, 0], pts[k, 1] - Q[i, 1], pts[k, 2] - Q[i, 2]
            ln = np.sqrt(lx * lx + ly * ly + lz * lz)
            lx, ly, lz = lx / ln, ly / ln, lz / ln
            qx, qy, qz = -lx * ln, -ly * ln, -lz * ln
            cx = Rs[k, 0, 0] * qx + Rs[k, 0, 1] * qy + Rs[k, 0, 2] * qz
            cy = Rs[k, 1, 0] * qx + Rs[k, 1, 1] * qy + Rs[k, 1, 2] * qz
            cz = Rs[k, 2, 0] * qx + Rs[k, 2, 1] * qy + Rs[k, 2, 2] * qz
            nl = nx * lx + ny * ly + nz * lz
            if nl <= 0.0:
                continue
            tan_t = min(np.sqrt(max(1.0 - nl * nl, 0.0)) / nl, 5.0)
            b = bias + 1.5 * cz / cams[k, 0] * tan_t
            fx, fy = cams[k, 0] * cx / cz + cams[k, 1], cams[k, 0] * cy / cz + cams[k, 2]
            xi, yi = int(fx), int(fy)
            lit = 0.0
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    xx, yy = min(max(xi + dx, 0), s - 1), min(max(yi + dy, 0), s - 1)
                    lit += cz <= depths[k, yy, xx] + b
            if lit == 0.0:
                continue
            hx, hy, hz = lx + vx, ly + vy, lz + vz
            hn = np.sqrt(hx * hx + hy * hy + hz * hz)
            nh = max((nx * hx + ny * hy + nz * hz) / hn, 0.0)
            acc += lit / 9.0 * (nl + specular * nh ** shininess)
        out[i] = ambient + (1 - ambient) * acc / m
    return out


def shade(Q, N, eye, light, maps):
    """Light factor of points Q (n, 3) with normals N (n, 3) seen from eye."""
    Rs = np.stack([cam["R"] for cam, _ in maps])
    cams = np.array([[cam["f"], cam["cx"], cam["cy"]] for cam, _ in maps])
    depths = np.stack([d for _, d in maps]).astype(np.float64)
    return _shade(np.ascontiguousarray(Q, np.float64), np.ascontiguousarray(N, np.float64), np.asarray(eye, float),
                  light["points"].astype(np.float64), Rs, cams, depths, light["ambient"], light["specular"],
                  light["shininess"], SHADOW_BIAS_MM)


def table_points(cam, size, step=1, xs=None, ys=None):
    """Where the rays of every step-th pixel (or of the pixel coordinates xs, ys) meet the table plane z = 0 (NaN
    if they do not)."""
    w, h = size
    if xs is None:
        xs, ys = np.meshgrid(np.arange(0, w, step, dtype=np.float32) + step / 2,
                             np.arange(0, h, step, dtype=np.float32) + step / 2)
    d = np.stack([(xs - cam["cx"]) / cam["f"], (ys - cam["cy"]) / cam["f"],
                 np.ones_like(xs)], -1) @ cam["R"].astype(np.float32)
    t = -cam["eye"][2] / d[..., 2]
    Q = cam["eye"] + t[..., None] * d
    Q[t <= 0] = np.nan
    return Q


def _cells(x, y, w, h, off):
    """Brick / grid cells: (index u, index v, local u, local v) for cells w x h, rows shifted by off * w."""
    v = np.floor(y / h)
    xs = x + off * w * v
    u = np.floor(xs / w)
    return u, v, xs - u * w, y - v * h


def _wood(rng, x, y, sd, n):
    pw = rng.uniform(60, 250)
    u, v, lu, lv = _cells(y, x, pw, rng.uniform(400, 2000), rng.choice((0.0, 0.5, rng.uniform(0, 1))))
    tone = _lattice(sd, u, v)
    warp = n.fbm(sd + 7, x * 0.3, y + 1000 * u, 40) * rng.uniform(8, 30) + n.fbm(sd + 8, x, y, 300) * 40
    rings = 0.5 + 0.5 * np.sin(2 * np.pi * (lu + warp + 50 * tone) / rng.uniform(3, 10))
    fine = n.fbm(sd + 3, x * 0.03, y * 2, 1.0, 4)
    g = 0.6 - 0.35 * rings ** rng.uniform(2, 6) + 0.15 * fine + 0.35 * (tone - 0.5)
    seam = np.minimum(lu, pw - lu) < rng.uniform(0.3, 1.2)
    return np.where(seam, 0.25 * g, g)


def _fabric(rng, x, y, sd, n):
    p = rng.uniform(0.5, 2.5)
    th = rng.uniform(0, np.pi)
    a, b = x * np.cos(th) + y * np.sin(th), -x * np.sin(th) + y * np.cos(th)
    ia, ib = np.floor(a / p), np.floor(b / p)
    twill = rng.integers(1, 4)
    over = (ia + ib * twill) % (twill + 1) < 1
    thread = np.where(over, np.sin(np.pi * (b / p - ib)), np.sin(np.pi * (a / p - ia)))
    jitter = np.where(over, _lattice(sd, ia, 0 * ib), _lattice(sd + 1, 0 * ia, ib))
    return 0.45 + 0.4 * thread + 0.25 * (jitter - 0.5) + 0.15 * n.fbm(sd + 2, x, y, 20, 3)


def _marble(rng, x, y, sd, n):
    th = rng.uniform(0, np.pi)
    wx, wy = n.warp(sd + 3, x, y, rng.uniform(80, 300), rng.uniform(20, 120))
    t = (wx * np.cos(th) + wy * np.sin(th)) / rng.uniform(40, 200) + rng.uniform(1, 4) * n.fbm(sd, wx, wy, 120, 6)
    vein = np.abs(np.sin(np.pi * t)) ** rng.uniform(0.2, 0.6)
    return 0.3 + 0.6 * vein + 0.08 * n.fbm(sd + 9, x, y, 5, 3)


def _granite(rng, x, y, sd, n):
    g = 0.5 + 0.35 * n.fbm(sd, x, y, rng.uniform(2, 10), 4)
    for k, (sz, d) in enumerate(((0.6, 0.12), (1.5, 0.06), (3.0, 0.03))):
        n = value_noise(sd + 20 + k, x / sz, y / sz)
        g = np.where(n > 1 - d, rng.uniform(0.0, 1.0), g)
    return g


def _tiles(rng, x, y, sd, n):
    w = rng.uniform(40, 400)
    h = w * rng.choice((1.0, 1.0, 0.5, 2.0))
    u, v, lu, lv = _cells(x, y, w, h, rng.choice((0.0, 0.5)))
    tone = 0.5 + 0.45 * (_lattice(sd, u, v) - 0.5) + 0.2 * n.fbm(sd + 1, x, y, rng.uniform(5, 60), 4)
    grout = rng.uniform(1, 5)
    edge = np.minimum(np.minimum(lu, w - lu), np.minimum(lv, h - lv))
    return np.where(edge < grout / 2, rng.choice((0.1, 0.9)), tone)


def _felt(rng, x, y, sd, n):
    return 0.5 + 0.3 * n.fbm(sd, x, y, rng.uniform(0.3, 1.5), 3) + 0.15 * n.fbm(sd + 5, x, y, 50, 3)


def _plain(rng, x, y, sd, n):
    g = 0.5 + 0.12 * n.fbm(sd, x, y, rng.uniform(30, 300), 4)
    for k in range(rng.integers(5, 60)):
        th = rng.uniform(0, np.pi)
        c = rng.uniform(-300, 300)
        d = np.abs(x * np.cos(th) + y * np.sin(th) - c)
        along = -x * np.sin(th) + y * np.cos(th)
        m = (d < rng.uniform(0.15, 0.6)) & (np.abs(along - rng.uniform(-300, 300)) < rng.uniform(5, 80))
        g = np.where(m, g + rng.choice((-1, 1)) * rng.uniform(0.15, 0.35), g)
    return g


def _leather(rng, x, y, sd, n):
    wx, wy = n.warp(sd, x, y, 10, rng.uniform(0.3, 1.5))
    d1, d2, _ = n.worley(sd + 1, wx, wy, rng.uniform(0.8, 3.0))
    crease = np.clip((d2 - d1) * rng.uniform(3, 8), 0, 1)
    return 0.35 + 0.4 * crease + 0.15 * n.fbm(sd + 2, x, y, 40, 3)


def _terrazzo(rng, x, y, sd, n):
    d1, d2, cid = n.worley(sd, *n.warp(sd + 1, x, y, 8, rng.uniform(0.5, 3)), rng.uniform(3, 15))
    chip = (d2 - d1) > rng.uniform(0.15, 0.4)
    ground = 0.5 + 0.1 * n.fbm(sd + 2, x, y, 3, 3)
    return np.where(chip & (cid < rng.uniform(0.3, 0.8)), cid * 1.4 - 0.1, ground)


def _cork(rng, x, y, sd, n):
    d1, _, cid = n.worley(sd, x, y, rng.uniform(0.6, 2.0))
    return 0.35 + 0.35 * cid + 0.25 * d1 + 0.1 * n.fbm(sd + 1, x, y, 20, 3)


def _pebbled(rng, x, y, sd, n):
    d1, _, _ = n.worley(sd, x, y, rng.uniform(0.3, 1.2))
    return 0.5 + 0.3 * (0.5 - np.clip(d1, 0, 1)) + 0.05 * n.fbm(sd + 1, x, y, 60, 3)


def _stripes(x, y, period, th):
    a = x * np.cos(th) + y * np.sin(th)
    return np.floor(a / period), a / period - np.floor(a / period)


def _gingham(rng, x, y, sd, n):
    p = rng.uniform(3, 25)
    th = rng.uniform(0, np.pi)
    _, fa = _stripes(x, y, p, th)
    _, fb = _stripes(x, y, p, th + np.pi / 2)
    duty = rng.uniform(0.3, 0.6)
    return 0.8 - 0.3 * (fa < duty) - 0.3 * (fb < duty) + 0.1 * _fabric(rng, x, y, sd + 1, n)


def _plaid(rng, x, y, sd, n):
    th = rng.uniform(0, np.pi)
    g = 0.5 + 0.1 * _fabric(rng, x, y, sd, n)
    for k in range(rng.integers(2, 5)):
        p, w, v = rng.uniform(15, 80), rng.uniform(0.05, 0.3), rng.uniform(-0.3, 0.3)
        for t in (th, th + np.pi / 2):
            _, f = _stripes(x, y, p, t)
            g = g + v * (f < w)
    return g


def _striped(rng, x, y, sd, n):
    i, f = _stripes(x, y, rng.uniform(2, 40), rng.uniform(0, np.pi))
    return 0.3 + 0.5 * _lattice(sd, i, 0 * i) * (f < rng.uniform(0.2, 0.8)) + 0.05 * n.fbm(sd, x, y, 10, 3)


def _brushed(rng, x, y, sd, n):
    th = rng.uniform(0, np.pi)
    a, b = x * np.cos(th) + y * np.sin(th), -x * np.sin(th) + y * np.cos(th)
    return 0.55 + 0.15 * n.fbm(sd, a * 0.01, b * 8, 1.0, 4) + 0.2 * n.fbm(sd + 1, x, y, 200, 2)


def _perforated(rng, x, y, sd, n):
    p = rng.uniform(2, 10)
    u, v, lu, lv = _cells(x, y, p, p, rng.choice((0.0, 0.5)))
    hole = np.hypot(lu - p / 2, lv - p / 2) < p * rng.uniform(0.15, 0.4)
    return np.where(hole, rng.uniform(0.0, 0.3), 0.6 + 0.1 * n.fbm(sd, x, y, 100, 3))


def _concrete(rng, x, y, sd, n):
    g = 0.5 + 0.15 * n.fbm(sd, x, y, 30, 6) + 0.08 * n.fbm(sd + 1, x, y, 1.5, 3)
    d1, _, _ = n.worley(sd + 2, x, y, rng.uniform(1, 4))
    return np.where(d1 < rng.uniform(0.03, 0.1), g - 0.25, g)


def _cardboard(rng, x, y, sd, n):
    _, f = _stripes(x, y, rng.uniform(3, 8), rng.uniform(0, np.pi))
    return 0.5 + 0.06 * np.sin(2 * np.pi * f) + 0.12 * n.fbm(sd, x, y, 3, 4) + 0.1 * n.fbm(sd + 1, x, y, 80, 3)


def _speckle(rng, x, y, sd, n):
    g = np.full(x.shape, 0.5, np.float32) + 0.05 * n.fbm(sd, x, y, 50, 3)
    for k in range(rng.integers(2, 5)):
        n = value_noise(sd + 10 + k, x / rng.uniform(0.3, 2), y / rng.uniform(0.3, 2))
        g = np.where(n > 1 - rng.uniform(0.02, 0.15), rng.uniform(0, 1), g)
    return g


def _rattan(rng, x, y, sd, n):
    p = rng.uniform(3, 12)
    th = rng.uniform(0, np.pi)
    ia, fa = _stripes(x, y, p, th)
    ib, fb = _stripes(x, y, p, th + np.pi / 2)
    over = (ia + ib) % 2 == 0
    return 0.3 + 0.5 * np.where(over, np.sin(np.pi * fa), np.sin(np.pi * fb)) + 0.1 * n.fbm(sd, x, y, 5, 3)


def _knit(rng, x, y, sd, n):
    p = rng.uniform(2, 6)
    th = rng.uniform(0, np.pi)
    a, b = x * np.cos(th) + y * np.sin(th), -x * np.sin(th) + y * np.cos(th)
    v = np.abs(((a / p) % 1) - 0.5) * 2 + ((b / (1.5 * p)) % 1)
    return 0.35 + 0.4 * np.abs(np.sin(np.pi * v)) + 0.1 * n.fbm(sd, x, y, 40, 3)


def _wallpaper(rng, x, y, sd, n):
    p = rng.uniform(20, 120)
    u, v, lu, lv = _cells(x, y, p, p, rng.choice((0.0, 0.5)))
    mx, my = np.abs(lu - p / 2), np.abs(lv - p / 2)
    motif = n.fbm(sd, mx, my, p * rng.uniform(0.1, 0.4), 4) > rng.uniform(-0.1, 0.2)
    return np.where(motif, rng.uniform(0.2, 0.8), 0.5) + 0.04 * n.fbm(sd + 1, x, y, 5, 3)


def _carpet(rng, x, y, sd, n):
    d1, _, cid = n.worley(sd, x, y, rng.uniform(0.8, 2.5))
    return 0.4 + 0.2 * cid + 0.25 * d1 + 0.2 * n.fbm(sd + 1, x, y, rng.uniform(30, 200), 4)


TEXTURES = (_wood, _fabric, _marble, _granite, _tiles, _felt, _plain, _leather, _terrazzo, _cork, _pebbled, _gingham,
            _plaid, _striped, _brushed, _perforated, _concrete, _cardboard, _speckle, _rattan, _knit, _wallpaper,
            _carpet)


def _surface(rng, x, y, n):
    sd = int(rng.integers(1 << 30))
    t = TEXTURES[rng.integers(len(TEXTURES))](rng, x, y, sd, n)
    base, contrast = rng.uniform(0.15, 0.85), rng.uniform(0.6, 1.6)
    stain = 1 + rng.uniform(0, 0.25) * n.fbm(sd + 99, x, y, rng.uniform(100, 600), 3)
    return np.clip(base * (1 + contrast * (t - 0.5)) * stain, 0, 1)


def _clutter(rng, x, y, g, reach):
    """Things on the table around the page: dark or light objects (mugs, phones, pens) as discs, rounded boxes and
    bars."""
    for k in range(rng.poisson(2.0)):
        cx, cy = rng.uniform(-reach, reach, 2)
        th = rng.uniform(0, np.pi)
        a, b = (x - cx) * np.cos(th) + (y - cy) * np.sin(th), -(x - cx) * np.sin(th) + (y - cy) * np.cos(th)
        kind = rng.integers(1, 4)
        if kind == 1:
            r = rng.uniform(25, 60)
            d = np.hypot(a, b)
            g = np.where(d < r, np.where(d > r * 0.85, rng.uniform(0.6, 1), rng.uniform(0.05, 0.9)), g)
        elif kind == 2:
            w, h, rr = rng.uniform(50, 90), rng.uniform(100, 170), rng.uniform(3, 10)
            q = np.hypot(np.maximum(np.abs(a) - w / 2 + rr, 0), np.maximum(np.abs(b) - h / 2 + rr, 0))
            g = np.where(q < rr, rng.uniform(0.02, 0.25), g)
        else:
            g = np.where((np.abs(a) < rng.uniform(3, 6)) & (np.abs(b) < rng.uniform(50, 90)), rng.uniform(0, 1), g)
    return g


PLAIN_PAPER_PROB = 0.05
PLAIN_GRAY_PROB = 0.05


def _uniform(rng, x, y, lvl, paper, n):
    """Uniform level, or a linear or radial gradient from it; faint paper structure when paper."""
    kind = rng.integers(3)
    th = rng.uniform(0, 2 * np.pi)
    if kind == 1:
        g = lvl * (1 + rng.uniform(0.05, 0.3) * np.tanh((x * np.cos(th) + y * np.sin(th)) / rng.uniform(100, 500)))
    elif kind == 2:
        c = rng.uniform(-200, 200, 2)
        g = lvl * (1 - rng.uniform(0.05, 0.35) * np.clip(np.hypot(x - c[0], y - c[1]) / rng.uniform(200, 600), 0, 1))
    else:
        g = np.full(x.shape, lvl, np.float32)
    if not paper:
        return np.clip(g, 0, 1).astype(np.float32)
    sd = int(rng.integers(1 << 30))
    g = g * (1 + rng.uniform(0, 0.02) * n.fbm(sd, x, y, rng.uniform(0.5, 3), 3)
             + rng.uniform(0, 0.02) * n.fbm(sd + 1, x, y, rng.uniform(10, 80), 4))
    return np.clip(g, 0, 1).astype(np.float32)


def table_pixel(cam, size):
    """Median size [mm] of a photo pixel on the table: the finest detail the table can show."""
    q = table_points(cam, size, 32)
    d = np.hypot(*np.gradient(q[..., 0], axis=(1, 0))) / 32
    return float(np.nanmedian(d)) if np.isfinite(d).any() else 0.0


def table_texture(rng, Q, px, ns):
    """Table luminance at table points Q (h, w, 3) mm, evaluated per point (sharp at any distance). One of the
    TEXTURES (fractal, cellular, domain-warped, woven, printed and tiled patterns) at a random base level, contrast
    and stains; sometimes a second surface beyond a straight table edge, things lying around the page, a broad
    brightness gradient and a random tone curve. Noise feature sizes are times ns, those under 2 px (px: the table's pixel
    size) left out."""
    x, y = np.nan_to_num(Q[..., 0]).astype(np.float32), np.nan_to_num(Q[..., 1]).astype(np.float32)
    n = Field(ns, 2 * px)
    r = rng.random()
    if r < PLAIN_PAPER_PROB:
        return _uniform(rng, x, y, rng.uniform(0.55, 0.97), True, n)
    if r < PLAIN_PAPER_PROB + PLAIN_GRAY_PROB:
        return _uniform(rng, x, y, rng.uniform(0.1, 0.9), False, n)
    g = _surface(rng, x, y, n)
    if rng.random() < 0.3:
        th, off = rng.uniform(0, 2 * np.pi), rng.uniform(150, 400)
        g = np.where(x * np.cos(th) + y * np.sin(th) > off, _surface(rng, x, y, n), g)
    if rng.random() < 0.4:
        g = _clutter(rng, x, y, g, 400)
    if rng.random() < 0.5:
        th = rng.uniform(0, 2 * np.pi)
        g = g * (1 + rng.uniform(0, 0.4) * np.tanh((x * np.cos(th) + y * np.sin(th)) / rng.uniform(200, 800)))
    return (np.clip(g, 0, 1) ** rng.uniform(0.7, 1.4)).astype(np.float32)


TONE_LEVELS = 4096
SRGB_TO_LINEAR = ((np.arange(256) / 255.0) ** 2.2).astype(np.float32)


def camera_effects(rng, img, depth, page_mask, c: RenderCfg):
    """Linear luminance (h, w) -> uint8 photo and the effects applied. Order of a camera: optics (defocus,
    motion, vignetting), exposure, sensor noise, tone curve, JPEG."""
    fx = {k: e for k, e in c.effects.items() if k != "shading" and rng.random() < e.prob}
    h, w = img.shape[:2]
    if "defocus" in fx:
        coc = _u(rng, fx["defocus"].params["coc_px"])
        z = np.where(np.isfinite(depth), depth, np.nan)
        focus = np.nanpercentile(z, rng.uniform(10, 90)) if page_mask.any() else 1.0
        zz = np.nan_to_num(z, nan=np.nanmax(z) if page_mask.any() else 1.0)
        img = variable_blur(img, (coc * np.abs(1 - focus / zz) * 8).astype(np.float32))
    if "motion_blur" in fx:
        L = _u(rng, fx["motion_blur"].params["length_px"])
        k = np.zeros((int(L) | 1, int(L) | 1), np.float32)
        k[k.shape[0] // 2] = 1
        rot = cv2.getRotationMatrix2D((k.shape[1] / 2 - 0.5, k.shape[0] / 2 - 0.5), rng.uniform(0, 180), 1)
        k = cv2.warpAffine(k, rot, k.shape[::-1])
        img = cv2.filter2D(img, -1, k / k.sum())
    gain = np.ones((h, w), np.float32)
    if "vignetting" in fx:
        yy, xx = np.ogrid[0:h, 0:w]
        gain = gain - _u(rng, fx["vignetting"].params["strength"]) * (
            ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / (0.25 * (w * w + h * h))).astype(np.float32)
    gamma = 1.0
    if "exposure" in fx:
        p = fx["exposure"].params
        gain = gain * 2 ** _u(rng, p["ev"])
        gamma = _u(rng, p["gamma"])
    img = img * gain
    if "iso_noise" in fx:
        g = float(np.exp(rng.uniform(*np.log(fx["iso_noise"].params["iso"])))) / 100.0
        sd = np.sqrt(np.clip(img, 0, 1) * 0.0004 * g + (0.002 * g) ** 2)
        img = img + sd * rng.standard_normal(img.shape, dtype=np.float32)
        if g > 8:
            img = cv2.bilateralFilter(img, 5, 0.05 * g / 8, 3)
    tone = (np.linspace(0, 1, TONE_LEVELS) ** (gamma / 2.2) * 255 + 0.5).astype(np.uint8)
    out = tone[(np.clip(img, 0, 1) * (TONE_LEVELS - 1) + 0.5).astype(np.int32)]
    if "jpeg" in fx:
        q = int(rng.integers(*fx["jpeg"].params["quality"]))
        out = cv2.imdecode(cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_GRAYSCALE)
    return np.ascontiguousarray(out), sorted(fx)
