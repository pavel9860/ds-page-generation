"""Paper strip on the table with line supports, an optional clamped edge and plastic folds: discrete elastica.

Unknowns y = (phi_0 .. phi_{n-1}, z_0): segment angles and the height of node 0; z_j = z_0 + h sum_{i<j} sin phi_i.
    E = sum_j D_j / (2h) (dphi_j - h kbar_j)^2 + sum_j w_j z_j
Bending stiffness D = d, except at a fold (s, theta, k): a hinge of stiffness k h and rest jump theta
(theta > 0 valley). Gravity node weights w are trapezoidal. Obstacles: every node z_j >= 0 (table) and every
support z(s_i) >= h_i, linear between the two nodes around s_i. A clamp holds z_0 = 0 and the tangent theta0
at s = 0 through a half-segment joint.
Solved by SQP: each step is a convex QP (quadprog) over all obstacle inequalities, so contacts open and close
inside the step. The gravity curvature enters the QP Hessian with its absolute value (a descent direction far
from equilibrium); steps rotate no segment by more than MAX_DPHI.
"""
import numpy as np
import quadprog

MAX_STEPS = 200
MAX_DPHI = 0.2
STEP_TOL = 1e-6                 # m: stop when no node moves more than this in a step
REG = 1e-9


def _nodes(y, h):
    return y[-1] + h * np.concatenate([[0.0], np.cumsum(np.sin(y[:-1]))])


def _xs(y, h):
    return h * np.concatenate([[0.0], np.cumsum(np.cos(y[:-1]))])


def _lower_hull_support(x, gap):
    """Two lower-hull nodes of (x, gap) bracketing the centre of mass: where the rigid strip rests."""
    hull = []
    for i in np.argsort(x):
        while len(hull) > 1 and ((x[hull[-1]] - x[hull[-2]]) * (gap[i] - gap[hull[-2]])
                                 - (gap[hull[-1]] - gap[hull[-2]]) * (x[i] - x[hull[-2]])) <= 0:
            hull.pop()
        hull.append(i)
    k = next((m for m in range(len(hull) - 1) if x[hull[m + 1]] >= x.mean()), len(hull) - 2)
    return hull[k], hull[k + 1]


def _rest(folds, b, h, n, clamp):
    """Initial state: the rigidly folded strip, clamped or resting on its two supporting obstacle nodes."""
    phi = np.zeros(n)
    for s, th, _ in folds:
        phi[int(round(s / h)):] += th
    if clamp is not None:
        return np.append(phi + clamp, 0.0)
    y = np.append(phi, 0.0)
    x, gap = _xs(y, h), _nodes(y, h) - b
    a, c = _lower_hull_support(x, gap)
    y[:-1] -= np.arctan2(gap[c] - gap[a], x[c] - x[a])
    y[-1] = b[a] - _nodes(y, h)[a]
    return y


def solve_strip(length, q, d, folds=(), supports=(), clamp=None, n=150, y0=None):
    """folds [(s, theta, k)], supports [(s, h)], clamp: edge tangent angle or None; y0: start state (a nearby
    solution), else the rigid rest. -> s, x, z node arrays and the state y."""
    h = length / n
    D = np.full(n - 1, d)
    kbar = np.zeros(n - 1)
    for s, th, k in folds:
        j = min(max(int(round(s / h)) - 1, 0), n - 2)
        D[j], kbar[j] = k * h, th / h
    b = np.zeros(n + 1)
    sj = np.array([min(int(s / h), n - 1) for s, _ in supports], int)
    sw = np.array([s / h for s, _ in supports]) - sj
    sh = np.array([hh for _, hh in supports])
    for j, w_, hh in zip(sj, sw, sh):
        b[j + (w_ >= 0.5)] = max(b[j + (w_ >= 0.5)], hh)
    y = _rest(folds, b, h, n, clamp) if y0 is None else y0.copy()

    free = np.arange(1 if clamp is not None else 0, n + 1)
    idx, jj = np.arange(n), np.arange(n - 1)
    H0 = np.zeros((n + 1, n + 1))
    H0[jj, jj] += D / h
    H0[jj + 1, jj + 1] += D / h
    H0[jj, jj + 1] -= D / h
    H0[jj + 1, jj] -= D / h
    H0[np.arange(n + 1), np.arange(n + 1)] += REG
    E = np.eye(n + 1)[[n]] if clamp is not None else np.zeros((0, n + 1))
    if clamp is not None:
        H0[0, 0] += 2 * d / h
    w = np.full(n + 1, q * h)
    w[[0, n]] *= 0.5
    wtail = np.cumsum(w[::-1])[::-1][1:]
    nf = len(free)
    m = nf + len(sj)
    rows = np.zeros((m, n + 1))                    # constraint rows over node heights: table nodes, supports
    rows[np.arange(nf), free] = 1.0
    rows[nf + np.arange(len(sj)), sj] = 1 - sw
    rows[nf + np.arange(len(sj)), sj + 1] = sw
    B = rows[:, 1:][:, ::-1].cumsum(1)[:, ::-1]   # d(row) / d(sin phi_i) / h: weight of nodes beyond segment i
    bound = np.concatenate([np.zeros(nf), sh])
    lam = np.zeros(m)
    Ct = np.zeros((n + 1, len(E) + m))
    Ct[:, :len(E)] = E.T
    Ct[n, len(E):] = 1.0
    for _ in range(MAX_STEPS):
        c, s = np.cos(y[:-1]), np.sin(y[:-1])
        r = D / h * (np.diff(y[:-1]) - h * kbar)
        g = np.append(h * c * wtail, q * length)
        g[:-2] -= r
        g[1:-1] += r
        if clamp is not None:
            g[0] += 2 * d / h * (y[0] - clamp)
        tail = wtail - (lam @ B)
        H = H0.copy()
        H[idx, idx] += np.abs(h * s * tail)
        Ct[:n, len(E):] = (B * (h * c)).T
        dy, _, _, _, mult, _ = quadprog.solve_qp(H, -g, Ct, np.concatenate([-E @ y, bound - rows @ _nodes(y, h)]),
                                                 len(E))
        a = min(1.0, MAX_DPHI / max(np.abs(dy[:-1]).max(), 1e-300))
        y = y + a * dy
        lam = (1 - a) * lam + a * mult[len(E):]
        dz = dy[-1] + h * np.concatenate([[0.0], np.cumsum(c * dy[:-1])])
        dx = h * np.concatenate([[0.0], np.cumsum(-s * dy[:-1])])
        if a * max(np.abs(dz).max(), np.abs(dx).max()) < STEP_TOL:
            return h * np.arange(n + 1), _xs(y, h), _nodes(y, h), y
    raise RuntimeError(f"strip: no equilibrium for folds {folds}, supports {supports}, clamp {clamp}")
