"""Deep creases: the ridge network of a crumpled and flattened sheet, light to heavy, varying over the page.

Per page: a coverage c (share of the sheet with creases, up to all of it), a largest crease length L_max, and
a ratio r between the largest and the smallest local length. Two smooth random fields of correlation length
region_mm set where the creases are (the field above its (1 - c) quantile) and the local crease length
L(x) = L_max r^(-u(x)), u in [0, 1]. Ridge starts are scattered with density ~ density / L(x)^2 over the covered
part, each ridge of log-normal length around L(x) in a random direction with one gentle kink, valley or mountain
at random. Each ridge is rendered with the physics-fitted crease profile of `shallow` at the level's severity;
the total height saturates at the level's w_max. All lengths in mm.
"""
import numpy as np

from . import shallow
from ..config import DeepCreaseCfg
from ..layout.paper import smooth_noise

FIELD_MM = 2.0


def _ridge(rng, p, th, L, kink):
    """Polyline from p along th, length L, with one gentle kink."""
    a = rng.uniform(0.3, 0.7)
    th2 = th + rng.normal(0.0, kink)
    q = p + a * L * np.array([np.cos(th), np.sin(th)])
    e = q + (1 - a) * L * np.array([np.cos(th2), np.sin(th2)])
    return np.vstack([np.linspace(p, q, max(int(a * L / 0.1), 4)),
                      np.linspace(q, e, max(int((1 - a) * L / 0.1), 4))[1:]])


def _unit(f):
    return (f - f.min()) / max(float(np.ptp(f)), 1e-9)


BAYER = (np.array([[0, 32, 8, 40, 2, 34, 10, 42], [48, 16, 56, 24, 50, 18, 58, 26],
                   [12, 44, 4, 36, 14, 46, 6, 38], [60, 28, 52, 20, 62, 30, 54, 22],
                   [3, 35, 11, 43, 1, 33, 9, 41], [51, 19, 59, 27, 49, 17, 57, 25],
                   [15, 47, 7, 39, 13, 45, 5, 37], [63, 31, 55, 23, 61, 29, 53, 21]]) + 0.5) / 64


def _starts(rng, spacing, covered):
    """Ridge starts spread regularly with local spacing (FIELD_MM grid): a jittered grid at the finest spacing
    thinned by ordered (Bayer) dithering of (finest / local spacing)^2."""
    g = max(float(spacing[covered].min()) if covered.any() else 1.0, FIELD_MM)
    h, w = spacing.shape
    ys, xs = np.meshgrid(np.arange(0, h * FIELD_MM, g), np.arange(0, w * FIELD_MM, g), indexing="ij")
    iy, ix = np.minimum((ys / FIELD_MM).astype(int), h - 1), np.minimum((xs / FIELD_MM).astype(int), w - 1)
    keep = covered[iy, ix] & ((g / spacing[iy, ix]) ** 2 > BAYER[np.arange(ys.shape[0])[:, None] % 8,
                                                                 np.arange(ys.shape[1])[None, :] % 8])
    jit = rng.uniform(-0.3, 0.3, (int(keep.sum()), 2)) * g
    return np.stack([xs[keep], ys[keep]], -1) + jit, iy[keep], ix[keep]


def sample(rng, W, H, level, c: DeepCreaseCfg):
    """Crease dicts (line, k, sign[, spread]) of one page's network: small creases spread regularly by the local
    scale, and a few broad, deep, low-angle creases across the page."""
    shape = (int(H / FIELD_MM) + 1, int(W / FIELD_MM) + 1)
    region = rng.uniform(*c.region_mm) / FIELD_MM
    cover = rng.uniform(*c.coverage[level])
    where = smooth_noise(rng, shape, region)
    covered = where >= np.quantile(where, 1 - cover) if cover < 1 else np.ones(shape, bool)
    L_max = rng.uniform(*c.max_scale_mm[level])
    local = L_max * rng.uniform(*c.scale_ratio) ** -_unit(smooth_noise(rng, shape, region))
    pts, iy, ix = _starts(rng, local / np.sqrt(c.density), covered)
    lengths = local[iy, ix] * np.exp(rng.normal(0.0, 0.4, len(pts)))
    kink = np.radians(c.kink_deg)
    out = [dict(line=_ridge(rng, p, rng.uniform(0, 2 * np.pi), L, kink), k=rng.uniform(*c.k[level]),
                sign=rng.choice([-1, 1])) for p, L in zip(pts, lengths)]
    for _ in range(rng.integers(c.broad_n[level][0], c.broad_n[level][1] + 1)):
        L = rng.uniform(*c.broad_length_mm)
        depth = rng.uniform(*c.broad_depth_frac) * c.max_depth_mm[level]
        th = rng.uniform(0, np.pi)
        line = _ridge(rng, rng.uniform([0, 0], [W, H]) - 0.5 * L * np.array([np.cos(th), np.sin(th)]), th, L, kink / 3)
        out.append(dict(line=line, k=depth / shallow.height(50.0), sign=rng.choice([-1, 1]),
                        spread=rng.uniform(*c.broad_spread)))
    return out


def render(creases, W, H, level, c: DeepCreaseCfg):
    return shallow.render(creases, W, H, h=c.pitch_mm, w_max=c.max_depth_mm[level] / 2)
