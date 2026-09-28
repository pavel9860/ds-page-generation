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


def sample(rng, W, H, level, c: DeepCreaseCfg):
    """Crease dicts (line, k, sign) of one page's network."""
    shape = (int(H / FIELD_MM) + 1, int(W / FIELD_MM) + 1)
    region = rng.uniform(*c.region_mm) / FIELD_MM
    cover = rng.uniform(*c.coverage[level])
    where = smooth_noise(rng, shape, region)
    covered = where >= np.quantile(where, 1 - cover) if cover < 1 else np.ones(shape, bool)
    L_max = rng.uniform(*c.max_scale_mm[level])
    local = L_max * rng.uniform(*c.scale_ratio) ** -_unit(smooth_noise(rng, shape, region))
    rate = np.where(covered, c.density / local ** 2, 0.0) * FIELD_MM ** 2
    n = rng.poisson(rate)
    iy, ix = np.nonzero(n)
    reps = n[iy, ix]
    iy, ix = np.repeat(iy, reps), np.repeat(ix, reps)
    pts = (np.stack([ix, iy], -1) + rng.random((len(ix), 2))) * FIELD_MM
    lengths = local[iy, ix] * np.exp(rng.normal(0.0, 0.4, len(ix)))
    kink = np.radians(c.kink_deg)
    return [dict(line=_ridge(rng, p, rng.uniform(0, 2 * np.pi), L, kink), k=rng.uniform(*c.k[level]),
                 sign=rng.choice([-1, 1])) for p, L in zip(pts, lengths)]


def render(creases, W, H, level, c: DeepCreaseCfg):
    return shallow.render(creases, W, H, h=c.pitch_mm, w_max=c.w_max_mm[level])
