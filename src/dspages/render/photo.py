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
            xi = min(max(int(cams[k, 0] * cx / cz + cams[k, 1]), 0), s - 1)
            yi = min(max(int(cams[k, 0] * cy / cz + cams[k, 2]), 0), s - 1)
            if cz > depths[k, yi, xi] + bias:
                continue
            hx, hy, hz = lx + vx, ly + vy, lz + vz
            hn = np.sqrt(hx * hx + hy * hy + hz * hz)
            nh = max((nx * hx + ny * hy + nz * hz) / hn, 0.0)
            acc += max(nx * lx + ny * ly + nz * lz, 0.0) + specular * nh ** shininess
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


def table_points(cam, size, step=1):
    """Where the rays of every step-th pixel meet the table plane z = 0 (NaN if they do not)."""
    w, h = size
    xs, ys = np.meshgrid(np.arange(0, w, step, dtype=np.float32) + step / 2,
                         np.arange(0, h, step, dtype=np.float32) + step / 2)
    d = np.stack([(xs - cam["cx"]) / cam["f"], (ys - cam["cy"]) / cam["f"],
                 np.ones_like(xs)], -1) @ cam["R"].astype(np.float32)
    t = -cam["eye"][2] / d[..., 2]
    Q = cam["eye"] + t[..., None] * d
    Q[t <= 0] = np.nan
    return Q


def _lattice(seed, i, j):
    h = (i.astype(np.int64) * 374761393 + j.astype(np.int64) * 668265263 + seed * 2246822519) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((h ^ (h >> 16)) & 0xFFFF).astype(np.float32) / 65535.0


def value_noise(seed, x, y):
    """Smooth value noise in [0, 1] at arbitrary coordinates (lattice pitch 1)."""
    i, j = np.floor(x), np.floor(y)
    fx, fy = x - i, y - j
    fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    a, b = _lattice(seed, i, j), _lattice(seed, i + 1, j)
    c, d = _lattice(seed, i, j + 1), _lattice(seed, i + 1, j + 1)
    return (a + (b - a) * fx) * (1 - fy) + (c + (d - c) * fx) * fy


def fbm(seed, x, y, scale, octaves=5, gain=0.5):
    """Fractal noise in about [-1, 1], feature size `scale` (same units as x, y)."""
    out, amp, f, tot = 0.0, 1.0, 1.0 / scale, 0.0
    for o in range(octaves):
        out = out + amp * (value_noise(seed + o, x * f, y * f) * 2 - 1)
        tot += amp
        amp, f = amp * gain, f * 2.03
    return out / tot


def worley(seed, x, y, scale):
    """Cellular noise: distances to the nearest and second nearest jittered lattice point (feature size `scale`)
    and the nearest point's cell id in [0, 1]."""
    x, y = x / scale, y / scale
    i0, j0 = np.floor(x), np.floor(y)
    d1 = np.full(x.shape, 9.0, np.float32)
    d2, cid = d1.copy(), np.zeros(x.shape, np.float32)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            i, j = i0 + di, j0 + dj
            d = np.hypot(i + _lattice(seed, i, j) - x, j + _lattice(seed + 1, i, j) - y)
            near = d < d1
            d2 = np.where(near, d1, np.minimum(d2, d))
            cid = np.where(near, _lattice(seed + 2, i, j), cid)
            d1 = np.minimum(d1, d)
    return d1, d2, cid


def warp(seed, x, y, scale, amount):
    """Domain warping: coordinates displaced by fractal noise."""
    return x + amount * fbm(seed, x, y, scale, 4), y + amount * fbm(seed + 50, x, y, scale, 4)


def _cells(x, y, w, h, off):
    """Brick / grid cells: (index u, index v, local u, local v) for cells w x h, rows shifted by off * w."""
    v = np.floor(y / h)
    xs = x + off * w * v
    u = np.floor(xs / w)
    return u, v, xs - u * w, y - v * h


def _wood(rng, x, y, sd):
    pw = rng.uniform(60, 250)
    u, v, lu, lv = _cells(y, x, pw, rng.uniform(400, 2000), rng.choice((0.0, 0.5, rng.uniform(0, 1))))
    tone = _lattice(sd, u, v)
    warp = fbm(sd + 7, x * 0.3, y + 1000 * u, 40) * rng.uniform(8, 30) + fbm(sd + 8, x, y, 300) * 40
    rings = 0.5 + 0.5 * np.sin(2 * np.pi * (lu + warp + 50 * tone) / rng.uniform(3, 10))
    fine = fbm(sd + 3, x * 0.03, y * 2, 1.0, 4)
    g = 0.6 - 0.35 * rings ** rng.uniform(2, 6) + 0.15 * fine + 0.35 * (tone - 0.5)
    seam = np.minimum(lu, pw - lu) < rng.uniform(0.3, 1.2)
    return np.where(seam, 0.25 * g, g)


def _fabric(rng, x, y, sd):
    p = rng.uniform(0.5, 2.5)
    th = rng.uniform(0, np.pi)
    a, b = x * np.cos(th) + y * np.sin(th), -x * np.sin(th) + y * np.cos(th)
    ia, ib = np.floor(a / p), np.floor(b / p)
    twill = rng.integers(1, 4)
    over = ((ia + ib * (twill if twill > 1 else 1)) % (twill + 1)) < (1 if twill > 1 else 1)
    thread = np.where(over, np.sin(np.pi * (b / p - ib)), np.sin(np.pi * (a / p - ia)))
    jitter = np.where(over, _lattice(sd, ia, 0 * ib), _lattice(sd + 1, 0 * ia, ib))
    return 0.45 + 0.4 * thread + 0.25 * (jitter - 0.5) + 0.15 * fbm(sd + 2, x, y, 20, 3)


def _marble(rng, x, y, sd):
    th = rng.uniform(0, np.pi)
    wx, wy = warp(sd + 3, x, y, rng.uniform(80, 300), rng.uniform(20, 120))
    t = (wx * np.cos(th) + wy * np.sin(th)) / rng.uniform(40, 200) + rng.uniform(1, 4) * fbm(sd, wx, wy, 120, 6)
    vein = np.abs(np.sin(np.pi * t)) ** rng.uniform(0.2, 0.6)
    return 0.3 + 0.6 * vein + 0.08 * fbm(sd + 9, x, y, 5, 3)


def _granite(rng, x, y, sd):
    g = 0.5 + 0.35 * fbm(sd, x, y, rng.uniform(2, 10), 4)
    for k, (sz, d) in enumerate(((0.6, 0.12), (1.5, 0.06), (3.0, 0.03))):
        n = value_noise(sd + 20 + k, x / sz, y / sz)
        g = np.where(n > 1 - d, rng.uniform(0.0, 1.0), g)
    return g


def _tiles(rng, x, y, sd):
    w = rng.uniform(40, 400)
    h = w * rng.choice((1.0, 1.0, 0.5, 2.0))
    u, v, lu, lv = _cells(x, y, w, h, rng.choice((0.0, 0.5)))
    tone = 0.5 + 0.45 * (_lattice(sd, u, v) - 0.5) + 0.2 * fbm(sd + 1, x, y, rng.uniform(5, 60), 4)
    grout = rng.uniform(1, 5)
    edge = np.minimum(np.minimum(lu, w - lu), np.minimum(lv, h - lv))
    return np.where(edge < grout / 2, rng.choice((0.1, 0.9)), tone)


def _felt(rng, x, y, sd):
    return 0.5 + 0.3 * fbm(sd, x, y, rng.uniform(0.3, 1.5), 3) + 0.15 * fbm(sd + 5, x, y, 50, 3)


def _plain(rng, x, y, sd):
    g = 0.5 + 0.12 * fbm(sd, x, y, rng.uniform(30, 300), 4)
    for k in range(rng.integers(5, 60)):
        th = rng.uniform(0, np.pi)
        c = rng.uniform(-300, 300)
        d = np.abs(x * np.cos(th) + y * np.sin(th) - c)
        along = -x * np.sin(th) + y * np.cos(th)
        m = (d < rng.uniform(0.15, 0.6)) & (np.abs(along - rng.uniform(-300, 300)) < rng.uniform(5, 80))
        g = np.where(m, g + rng.choice((-1, 1)) * rng.uniform(0.15, 0.35), g)
    return g


def _leather(rng, x, y, sd):
    wx, wy = warp(sd, x, y, 10, rng.uniform(0.3, 1.5))
    d1, d2, _ = worley(sd + 1, wx, wy, rng.uniform(0.8, 3.0))
    crease = np.clip((d2 - d1) * rng.uniform(3, 8), 0, 1)
    return 0.35 + 0.4 * crease + 0.15 * fbm(sd + 2, x, y, 40, 3)


def _terrazzo(rng, x, y, sd):
    d1, d2, cid = worley(sd, *warp(sd + 1, x, y, 8, rng.uniform(0.5, 3)), rng.uniform(3, 15))
    chip = (d2 - d1) > rng.uniform(0.15, 0.4)
    ground = 0.5 + 0.1 * fbm(sd + 2, x, y, 3, 3)
    return np.where(chip & (cid < rng.uniform(0.3, 0.8)), cid * 1.4 - 0.1, ground)


def _cork(rng, x, y, sd):
    d1, _, cid = worley(sd, x, y, rng.uniform(0.6, 2.0))
    return 0.35 + 0.35 * cid + 0.25 * d1 + 0.1 * fbm(sd + 1, x, y, 20, 3)


def _pebbled(rng, x, y, sd):
    d1, _, _ = worley(sd, x, y, rng.uniform(0.3, 1.2))
    return 0.5 + 0.3 * (0.5 - np.clip(d1, 0, 1)) + 0.05 * fbm(sd + 1, x, y, 60, 3)


TEXTURES = (_wood, _fabric, _marble, _granite, _tiles, _felt, _plain, _leather, _terrazzo, _cork, _pebbled)


def table_texture(rng, Q):
    """Table surface luminance at table points Q (h, w, 3) mm, evaluated per point (sharp at any distance): wood
    planks, woven fabric, marble, granite, tiles, felt, plain with scratches, leather, terrazzo, cork or pebbled
    plastic (fractal, cellular and domain-warped noise), at a random base level and
    contrast, with broad stains."""
    x, y = np.nan_to_num(Q[..., 0]).astype(np.float32), np.nan_to_num(Q[..., 1]).astype(np.float32)
    sd = int(rng.integers(1 << 30))
    t = TEXTURES[rng.integers(len(TEXTURES))](rng, x, y, sd)
    base, contrast = rng.uniform(0.05, 0.85), rng.uniform(0.3, 1.2)
    stain = 1 + rng.uniform(0, 0.25) * fbm(sd + 99, x, y, rng.uniform(100, 600), 3)
    return np.clip(base * (1 + contrast * (t - 0.5)) * stain, 0, 1).astype(np.float32)


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
