import math

import numpy as np

from .arm import solve_arm
from .ode import rk4_3
from .two_support import solve_two_support

N_ARM = 400


def _arm_sxz(arm, length, q, d):
    tr = rk4_3(arm["theta0"], arm["v0"], arm["ell"], q, d, n_steps=N_ARM,
                      r_tip=arm.get("r_tip", 0.0))
    s = np.linspace(0.0, arm["ell"], N_ARM + 1)
    x = np.concatenate([[0.0], np.cumsum(0.5 * (np.cos(tr[1:, 0]) + np.cos(tr[:-1, 0])) * np.diff(s))])
    z = tr[:, 2]
    if arm["ell"] < length:
        s = np.append(s, length)
        x = np.append(x, x[-1] + (length - arm["ell"]) * math.cos(tr[-1, 0]))
        z = np.append(z, z[-1])
    return s, x, z


def clamp(length, theta0, q, d):
    return _arm_sxz(solve_arm(theta0, length, q=q, d=d), length, q, d)


def center_support(length, h, q, d):
    half = 0.5 * length
    s, x, z = _arm_sxz(solve_arm(0.0, half, plane=-h, q=q, d=d), half, q, d)
    return (np.concatenate([half - s[::-1], half + s[1:]]),
            np.concatenate([-x[::-1], x[1:]]) + half,
            np.concatenate([z[::-1], z[1:]]) + h)


def two_support(length, p1, h1, p2, h2, q, d):
    x, z = solve_two_support(p1, h1, p2, h2, length, q=q, d=d)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(z)))])
    return s, x, z


KINDS = {"clamp": clamp, "center_support": center_support, "two_support": two_support}


def solve_case(kind, params, length, q, d):
    return KINDS[kind](length, *params, q, d)
