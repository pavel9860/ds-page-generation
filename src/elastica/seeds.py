import math

import numpy as np


def _bend(th):
    return 2.17 * math.sin(th) ** 0.69 / (0.25 * (24.0 * th) ** (2.0 / 3.0))


def flat_seed(dev, h, q, d):
    th = abs(dev)
    roots = [r.real for r in np.roots([1.0, 0.0, 0.0, -24.0 * th, -72.0 * h * (q / d) ** (1.0 / 3.0)])
             if abs(r.imag) < 1e-9 and r.real > 0.0]
    if not roots:
        return None
    G = max(roots)
    lc = (d / q) ** (1.0 / 3.0)
    g = G * lc
    v0 = -math.copysign(_bend(th) * (G * G / 6.0 + 2.0 * th / G) / lc, dev)
    return np.array([v0, g, -q * g / 3.0 + 2.0 * d * th / (g * g)])


def tip_seed(dev, h, ell, q, d):
    th = abs(dev)
    B = -q / (24.0 * d)
    c = (-th - 3.0 * B * ell ** 3 - h / ell) / (2.0 * ell * ell)
    v0 = -math.copysign(-_bend(th) * (6.0 * c * ell + 12.0 * B * ell * ell), dev)
    return np.array([v0, -6.0 * d * c])


def free_touches(dev, h, ell, q, d):
    return h + abs(dev) * ell - q * ell ** 4 / (8.0 * d) < 0.0
