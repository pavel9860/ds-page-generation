"""Deep creases: the ridge network of a crumpled and flattened sheet, from light to heavy.

Network statistics after Blair & Kudrolli, "Geometry of Crumpled Paper" (PRL 94, 166107, 2005): log-normal
ridge lengths, low vertex degree (capped), most growth attempts ending without a branch. Ridges start at
existing vertices, leave them in any direction and bend slightly along the way; valley or mountain at random.
Each ridge is rendered with the physics-fitted crease profile of `shallow` at the level's severity; the total
height saturates at the level's w_max. All lengths in mm.
"""
import numpy as np

from . import shallow
from ..config import DeepCreaseCfg


def _ridge(rng, p, th, L, kink):
    """Polyline from p along th, length L, with one gentle kink."""
    a = rng.uniform(0.3, 0.7)
    th2 = th + rng.normal(0.0, kink)
    q = p + a * L * np.array([np.cos(th), np.sin(th)])
    e = q + (1 - a) * L * np.array([np.cos(th2), np.sin(th2)])
    return np.vstack([np.linspace(p, q, max(int(a * L / 0.1), 4)),
                     np.linspace(q, e, max(int((1 - a) * L / 0.1), 4))[1:]])


def sample(rng, W, H, level, c: DeepCreaseCfg):
    """Crease dicts (line, k, sign) of one network, level in c.levels."""
    diag = np.hypot(W, H)
    n = rng.integers(*c.n_ridges[level], endpoint=True)
    kink = np.radians(c.kink_deg)
    nodes = [rng.uniform([0, 0], [W, H]) for _ in range(max(2, n // 8))]
    degree = [0] * len(nodes)
    out = []
    while len(out) < n:
        open_ = [i for i, d in enumerate(degree) if d < c.max_degree]
        i = open_[rng.integers(len(open_))]
        L = float(np.clip(rng.lognormal(np.log(c.length_median_frac * diag), c.length_sigma),
                          *(np.array(c.length_frac) * diag)))
        line = _ridge(rng, nodes[i], rng.uniform(0, 2 * np.pi), L, kink)
        out.append(dict(line=line, k=rng.uniform(*c.k[level]), sign=rng.choice([-1, 1])))
        degree[i] += 1
        if rng.random() < c.branch_prob:
            nodes.append(line[-1])
            degree.append(1)
    return out


def render(creases, W, H, level, c: DeepCreaseCfg):
    return shallow.render(creases, W, H, h=c.pitch_mm, w_max=c.w_max_mm[level])
