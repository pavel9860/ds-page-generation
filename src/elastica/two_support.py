import math

import numpy as np
from scipy.optimize import fsolve

from .arm import solve_arm
from .ode import D, Q, rk4_3

N = 60


def _outer(th, ell, h, left, q, d):
    if left:
        return solve_arm(th + math.pi, ell, plane=-h, flat_theta=math.pi, q=q, d=d)
    return solve_arm(th, ell, plane=-h, q=q, d=d)


def _shoot(th, v, z, t, a, b, q, d, n=N):
    tr = rk4_3(th, v, b - a, q, d, n_steps=n, r_tip=t - q * (b - a))
    tr[:, 2] += z
    return tr


def solve_two_support(p1, h1, p2, h2, L, q=Q, d=D):
    span = p2 - p1

    def middle(s, contact=None):
        th1, th2, t1 = s[:3]
        v1 = -_outer(th1, p1, h1, True, q, d)["v0"]
        if contact is None:
            return [_shoot(th1, v1, h1, t1, p1, p2, q, d)]
        sc, rc = contact
        a = _shoot(th1, v1, h1, t1, p1, sc, q, d)
        b = _shoot(*a[-1], t1 - q * (sc - p1) + rc, sc, p2, q, d)
        return [a, b]

    def F_free(s):
        e = middle(s)[-1][-1]
        return np.array([e[2] - h2, e[0] - s[1], e[1] - _outer(s[1], L - p2, h2, False, q, d)["v0"]])

    def F_point(s):
        a, b = middle(s, (s[3], s[4]))
        e = b[-1]
        return np.array([e[2] - h2, e[0] - s[1], e[1] - _outer(s[1], L - p2, h2, False, q, d)["v0"],
                         a[-1, 2], a[-1, 0]])

    def rod_flat(th, h, out_len, left):
        o = _outer(th, out_len, h, left, q, d)
        i = _outer(th, span, h, not left, q, d)
        return o, i

    slope = math.atan2(h2 - h1, span)
    s = fsolve(F_free, [slope, slope, 0.5 * q * span], xtol=1e-12)
    mid = middle(s)
    state = ("free", s)
    if mid[0][:, 2].min() < -1e-7:
        th1 = fsolve(lambda y: [sum(a["v0"] for a in rod_flat(y[0], h1, p1, True))], [s[0]], xtol=1e-12)[0]
        th2 = fsolve(lambda y: [sum(a["v0"] for a in rod_flat(y[0], h2, L - p2, False))], [s[1]], xtol=1e-12)[0]
        i1, i2 = rod_flat(th1, h1, p1, True)[1], rod_flat(th2, h2, L - p2, False)[1]
        if i1["branch"] == "flat" and i2["branch"] == "flat" and i1["ell"] + i2["ell"] <= span:
            state = ("flat", (th1, th2, i1, i2))
        else:
            g1 = i1["ell"] if i1["branch"] == "flat" else 0.5 * span
            g2 = i2["ell"] if i2["branch"] == "flat" else 0.5 * span
            t1 = q * g1 + i1.get("r_tip", 0.0)
            rc = q * span - t1 - (q * g2 + i2.get("r_tip", 0.0))
            s = fsolve(F_point, [th1, th2, t1, p1 + span * g1 / (g1 + g2), rc], xtol=1e-12)
            state = ("point", s)
    return _profile(state, p1, h1, p2, h2, L, q, d, middle)


def _arm_xz(a, q, d, n=200):
    tr = rk4_3(a["theta0"], a["v0"], a["ell"], q, d, n_steps=n, r_tip=a.get("r_tip", 0.0))
    ds = a["ell"] / n
    return tr, ds


def _profile(state, p1, h1, p2, h2, L, q, d, middle):
    kind, s = state
    o1 = _outer(s[0], p1, h1, True, q, d)
    o2 = _outer(s[1], L - p2, h2, False, q, d)

    def pts(tr, ds, z0, x0, rev):
        th, z = tr[:, 0], tr[:, 2] + z0
        x = x0 + np.concatenate([[0.0], np.cumsum(0.5 * (np.cos(th[1:]) + np.cos(th[:-1])) * ds)])
        return (x[::-1], z[::-1]) if rev else (x, z)

    xs, zs = [], []
    tr, ds = _arm_xz(o1, q, d)
    x, z = pts(tr, ds, h1, 0.0, True)
    if o1["ell"] < p1:
        x, z = np.concatenate([[x[0] - (p1 - o1["ell"])], x]), np.concatenate([[z[0]], z])
    xs.append(x - x[-1]); zs.append(z)
    if kind == "flat":
        tr, ds = _arm_xz(s[2], q, d)
        x, z = pts(tr, ds, h1, 0.0, False)
        tr2, ds2 = _arm_xz(s[3], q, d)
        x2, z2 = pts(tr2, ds2, h2, 0.0, False)
        gap = (p2 - p1) - s[2]["ell"] - s[3]["ell"]
        xm = np.concatenate([x, x[-1] + gap + (x2 - x2[-1])[::-1]]); zm = np.concatenate([z, z2[::-1]])
    else:
        segs = middle(s, None if kind == "free" else (s[3], s[4]))
        lens = [p2 - p1] if kind == "free" else [s[3] - p1, p2 - s[3]]
        xm, zm, x0 = [], [], 0.0
        for tr, ln in zip(segs, lens):
            x, z = pts(tr, ln / (len(tr) - 1), 0.0, x0, False)
            xm.append(x); zm.append(z); x0 = x[-1]
        xm, zm = np.concatenate(xm), np.concatenate(zm)
    xs.append(xm); zs.append(zm)
    tr, ds = _arm_xz(o2, q, d)
    x, z = pts(tr, ds, h2, xm[-1], False)
    if o2["ell"] < L - p2:
        x, z = np.append(x, x[-1] + (L - p2 - o2["ell"])), np.append(z, z[-1])
    xs.append(x); zs.append(z)
    return np.concatenate(xs), np.concatenate(zs)
