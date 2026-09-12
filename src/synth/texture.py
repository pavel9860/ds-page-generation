"""Printed-grid raster: 2.5mm fine grid (3px@300dpi) + heavy 10-cell grid (6px)."""
import cv2
import numpy as np

from . import config as cfg

FINE_MM = cfg.FINE_MM
HEAVY_MM = cfg.HEAVY_MM
FINE_PX_300 = cfg.FINE_PX_300
HEAVY_PX_300 = cfg.HEAVY_PX_300
PX_PER_MM_300 = cfg.PX_PER_MM_300


def make_old_creases(rng, span):
    """Print-stage crease: a local blur in the flat print (ink smear), 2D
    texture-space, distinct from geometry.py's 3D folds/creases."""
    if rng.random() >= cfg.OLD_CREASE_PROB:
        return []
    length_mm = rng.uniform(*cfg.OLD_CREASE_LENGTH_MM)
    th = rng.uniform(0, np.pi)
    cx, cy = rng.uniform(*cfg.OLD_CREASE_CENTER_FRAC, 2) * span
    s = rng.uniform(*cfg.OLD_CREASE_S_FRAC) * span / 3.0
    return [dict(th=th, cx=cx, cy=cy, length_mm=length_mm, s=s)]


def _old_crease_field(c, X, Y):
    du, dv = X - c['cx'], Y - c['cy']
    t_perp = du * np.cos(c['th']) + dv * np.sin(c['th'])
    t_along = -du * np.sin(c['th']) + dv * np.cos(c['th'])
    return np.exp(-(t_perp / c['s']) ** 2) * np.exp(-(t_along / (c['length_mm'] / 2)) ** 2)


def _fbm_noise(H, W, rng, octaves=5, base_res=8, persistence=0.55, lacunarity=2.5,
              min_res_grid=256, max_res_grid=256):
    """Sum of `octaves` value-noise layers, each generated on a coarse grid
    and upsampled with bicubic interpolation -- a standard fBm approximation
    without an external Perlin library. `min_res_grid`: the coarsest octave
    grid is generated at at least this resolution before upsampling.
    `max_res_grid`: the finest octave is CAPPED there too -- generating a
    near-full-resolution random grid for the last octave and upsampling it
    (a near no-op resize) is the dominant cost for no visible benefit, since
    that octave's own contribution is already tiny (persistence**(octaves-1))."""
    base_res = max(base_res, min_res_grid / (2.0 ** (octaves - 1)))
    out = np.zeros((H, W), np.float32)
    amp, total_amp, res = 1.0, 0.0, base_res
    for _ in range(octaves):
        res = min(res, max_res_grid)
        gh, gw = max(2, int(res)), max(2, int(res))
        layer = rng.uniform(-1, 1, (gh, gw)).astype(np.float32)
        layer = cv2.resize(layer, (W, H), interpolation=cv2.INTER_CUBIC)
        out += amp * layer
        total_amp += amp
        amp *= persistence
        res *= lacunarity
    return out / total_amp


def add_fbm_creases(tex, rng, ridge_frac=cfg.FBM_CREASE_RIDGE_FRAC,
                    sharpen=cfg.FBM_CREASE_SHARPEN, darken=cfg.FBM_CREASE_DARKEN):
    """Fine crumple-crease texture, painted directly on the flat print (2D,
    texture-space -- never touches z(U,V) or the Newton warp, so it can't
    be aliased away by the warp's downscaled height-field grid). Gradient
    magnitude of multi-octave fBm noise reads as crease/ridge boundaries;
    keep only the sharpest `ridge_frac` fraction, sharpen, then darken the
    texture proportionally. Paired 1:1 with geometry.py's 3D crease network
    by render.py (same trigger draw), not an independent probability."""
    H, W = tex.shape
    field = _fbm_noise(H, W, rng)
    gx = cv2.Sobel(field, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(field, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    thresh = np.percentile(mag, 100 * (1 - ridge_frac))
    mask = np.clip((mag - thresh) / (mag.max() - thresh + 1e-6), 0, 1)
    if sharpen != 1.0:
        mask = mask ** sharpen
    return np.clip(tex - darken * mask * tex, 0, 1).astype(np.float32)


def render_grid_texture(mm_w, mm_h, ppm, rng, old_creases=None, ink=cfg.INK, paper=cfg.PAPER,
                        margin_mm=0.0, noise=True, add_fbm_creases_flag=False):
    """Render the flat print at `ppm` px/mm. `margin_mm`: blank paper border
    around the grid, same tone as between lines."""
    W, H = int(mm_w * ppm), int(mm_h * ppm)
    fine_hw = 0.5 * FINE_PX_300 / PX_PER_MM_300 * ppm
    heavy_hw = 0.5 * HEAVY_PX_300 / PX_PER_MM_300 * ppm

    xs = np.arange(W, dtype=np.float32) / ppm
    ys = np.arange(H, dtype=np.float32) / ppm

    def line_field(coord_mm, hw_px, every_mm):
        k = np.round(coord_mm / every_mm)
        d = np.abs(coord_mm - k * every_mm) * ppm
        return np.clip(hw_px - d + 0.5, 0.0, 1.0)

    cov_x = np.maximum(line_field(xs, fine_hw, FINE_MM), line_field(xs, heavy_hw, HEAVY_MM))
    cov_y = np.maximum(line_field(ys, fine_hw, FINE_MM), line_field(ys, heavy_hw, HEAVY_MM))
    cov = np.maximum(cov_x[None, :], cov_y[:, None])
    if margin_mm > 0:
        in_x = (xs >= margin_mm) & (xs <= mm_w - margin_mm)
        in_y = (ys >= margin_mm) & (ys <= mm_h - margin_mm)
        cov = cov * (in_x[None, :] & in_y[:, None])

    tex = paper * (1.0 - cov) + ink * cov

    if old_creases:
        MX, MY = np.meshgrid(xs, ys)
        sigma = max(ppm * cfg.OLD_CREASE_BLUR_SIGMA_SCALE, cfg.OLD_CREASE_BLUR_SIGMA_MIN)
        blurred = cv2.GaussianBlur(tex, (0, 0), sigma)
        blend = np.zeros_like(tex)
        for c in old_creases:
            blend = np.maximum(blend, _old_crease_field(c, MX, MY))
        tex = tex * (1 - blend) + blurred * blend

    if add_fbm_creases_flag:
        tex = add_fbm_creases(tex, rng)

    if noise:
        fib = cv2.GaussianBlur(rng.standard_normal(tex.shape, dtype=np.float32), (0, 0),
                               cfg.NOISE_FIB_BLUR_SIGMA)
        grain = rng.standard_normal(tex.shape, dtype=np.float32)
        edge_zone = 4 * cov * (1 - cov)   # peaks at ink/paper edge (cov=0.5)
        tex = tex + cfg.NOISE_FIB_AMP * fib + cfg.NOISE_GRAIN_AMP * grain + \
            cfg.NOISE_EDGE_AMP * edge_zone * grain

    return np.clip(tex, 0, 1).astype(np.float32)


def lattice_points(mm_w, mm_h, margin_mm=0.0):
    """All fine-grid crossings in mm."""
    nx, ny = int(mm_w / FINE_MM) + 1, int(mm_h / FINE_MM) + 1
    cx, cy = np.arange(nx) * FINE_MM, np.arange(ny) * FINE_MM
    U, V = np.meshgrid(cx, cy)
    U, V = U.ravel(), V.ravel()
    if margin_mm > 0:
        keep = (U >= margin_mm) & (U <= mm_w - margin_mm) & \
            (V >= margin_mm) & (V <= mm_h - margin_mm)
        U, V = U[keep], V[keep]
    return U, V
