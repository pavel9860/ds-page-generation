"""Light, shadows, background and camera effects of the photo.

Light: a point light (sharp shadows) or an area light sampled at AREA_SAMPLES points (soft shading and shadows)
above the table. Per point: ambient + (1 - ambient) * mean over light samples of visibility * (lambert + specular);
visibility from a depth map rendered from each light sample, so the page shadows itself and the table.
Camera effects run on linear RGB in [0, 1]: depth-of-field blur, motion blur, vignetting, white balance to a
colour temperature, exposure and gamma, sensor noise at an ISO, JPEG.
"""
import cv2
import numpy as np

from ..config import LightCfg, RenderCfg
from ..layout.paper import smooth_noise, variable_blur
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


def shade(Q, N, eye, light, maps):
    """Light factor of points Q (n, 3) with normals N seen from eye."""
    v = eye - Q
    v /= np.linalg.norm(v, axis=-1, keepdims=True)
    acc = np.zeros(len(Q))
    for L, (cam, depth) in zip(light["points"], maps):
        to_l = L - Q
        to_l /= np.linalg.norm(to_l, axis=-1, keepdims=True)
        pix, z = project(Q, cam)
        xi = np.clip(pix[:, 0].astype(int), 0, SHADOW_PX - 1)
        yi = np.clip(pix[:, 1].astype(int), 0, SHADOW_PX - 1)
        vis = z <= depth[yi, xi] + SHADOW_BIAS_MM
        hv = to_l + v
        hv /= np.linalg.norm(hv, axis=-1, keepdims=True)
        spec = light["specular"] * np.clip((N * hv).sum(-1), 0, 1) ** light["shininess"]
        acc += vis * (np.clip((N * to_l).sum(-1), 0, 1) + spec)
    return light["ambient"] + (1 - light["ambient"]) * acc / len(light["points"])


def table_points(cam, size):
    """Where each pixel's ray meets the table plane z = 0 (NaN if it does not)."""
    w, h = size
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32) + 0.5, np.arange(h, dtype=np.float32) + 0.5)
    d = np.stack([(xs - cam["cx"]) / cam["f"], (ys - cam["cy"]) / cam["f"],
                 np.ones_like(xs)], -1) @ cam["R"].astype(np.float32)
    t = -cam["eye"][2] / d[..., 2]
    Q = cam["eye"] + t[..., None] * d
    Q[t <= 0] = np.nan
    return Q


def table_texture(rng, Q):
    """Procedural table surface at table points Q (h, w, 3) mm: base colour, low-frequency variation, wood-like
    grain or fabric weave."""
    hue = rng.uniform(0.0, 1.0)
    sat = rng.uniform(0.0, 0.35) if rng.random() < 0.7 else rng.uniform(0.35, 0.6)
    val = rng.uniform(0.12, 0.8)
    base = cv2.cvtColor(np.array([[[hue * 179, sat * 255, val * 255]]], np.uint8), cv2.COLOR_HSV2RGB)[0, 0] / 255.0
    base = base.astype(np.float32)
    x, y = np.nan_to_num(Q[..., 0]), np.nan_to_num(Q[..., 1])
    lowf = smooth_noise(rng, Q.shape[:2], 60)
    kind = rng.integers(3)
    if kind == 0:
        th = rng.uniform(0, np.pi)
        s = x * np.cos(th) + y * np.sin(th)
        pattern = 0.5 + 0.5 * np.sin(2 * np.pi * s / rng.uniform(4, 15) + 3 * lowf)
    elif kind == 1:
        p = rng.uniform(0.8, 3.0)
        pattern = 0.5 + 0.25 * (np.sin(2 * np.pi * x / p) + np.sin(2 * np.pi * y / p))
    else:
        pattern = 0.5 + 0.5 * np.clip(smooth_noise(rng, Q.shape[:2], 8), -1, 1)
    tex = base * (1 + 0.25 * (pattern[..., None] - 0.5) + 0.08 * lowf[..., None])
    return np.clip(tex, 0, 1).astype(np.float32)


def kelvin_rgb(k):
    """Relative RGB of a black body at k kelvin (Tanner Helland fit), 1 at 6500 K."""
    t = k / 100.0

    def rgb(t):
        r = 255.0 if t <= 66 else 329.698727446 * (t - 60) ** -0.1332047592
        g = 99.4708025861 * np.log(t) - 161.1195681661 if t <= 66 else 288.1221695283 * (t - 60) ** -0.0755148492
        b = 255.0 if t >= 66 else (0.0 if t <= 19 else 138.5177312231 * np.log(t - 10) - 305.0447927307)
        return np.clip([r, g, b], 1, 255)
    return (rgb(t) / rgb(65.0)).astype(np.float32)


TONE_LEVELS = 4096
SRGB_TO_LINEAR = ((np.arange(256) / 255.0) ** 2.2).astype(np.float32)


def camera_effects(rng, img, depth, page_mask, c: RenderCfg):
    """Linear RGB (h, w, 3) -> uint8 RGB photo and the effects applied. Order of a camera: optics (defocus,
    motion, vignetting), exposure, sensor noise, white balance, tone curve, JPEG."""
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
    img = img * gain[..., None]
    if "iso_noise" in fx:
        g = float(np.exp(rng.uniform(*np.log(fx["iso_noise"].params["iso"])))) / 100.0
        sd = np.sqrt(np.clip(img, 0, 1) * 0.0004 * g + (0.002 * g) ** 2)
        img = img + sd * rng.standard_normal(img.shape, dtype=np.float32)
        chroma = rng.standard_normal((h // 2, w // 2, 3), dtype=np.float32) * 0.002 * g
        img = img + cv2.resize(cv2.GaussianBlur(chroma, (0, 0), 0.75), (w, h))
        if g > 8:
            img = cv2.bilateralFilter(img, 5, 0.05 * g / 8, 3)
    if "white_balance" in fx:
        p = fx["white_balance"].params
        wb = kelvin_rgb(_u(rng, p["kelvin"])) * np.array([1.0, 1.0 + _u(rng, p["tint"]), 1.0], np.float32)
        img = img * (wb / wb.mean())
    tone = (np.linspace(0, 1, TONE_LEVELS) ** (gamma / 2.2) * 255 + 0.5).astype(np.uint8)
    out = tone[(np.clip(img, 0, 1) * (TONE_LEVELS - 1) + 0.5).astype(np.int32)]
    if "jpeg" in fx:
        q = int(rng.integers(*fx["jpeg"].params["quality"]))
        out = cv2.imdecode(cv2.imencode(".jpg", out[..., ::-1],
                           [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)[..., ::-1]
    return np.ascontiguousarray(out), sorted(fx)
