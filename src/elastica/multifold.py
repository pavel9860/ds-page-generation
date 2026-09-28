"""Strip with plastic folds on line supports, an optional clamped edge and the table, as a discrete elastica.

Unknowns y = (phi_0..phi_{n-1}, z_0); nodes z_j = z_0 + h sum_{i<j} sin phi_i.
    E = sum_j D_j/(2h) (dphi_j - h kbar_j)^2 + q h sum_j z_j
A fold (s_c, theta, k) is a hinge: joint stiffness k*h, rest jump theta, i.e.
dphi = theta - M/k. theta > 0 valley, < 0 mountain.
Obstacles are node lower bounds z_j >= b_j: the table (b = 0) and supports (s_i, h_i)
(b = h_i at the node over the support, fixed in arc length). A clamp fixes z_0 = 0 and the tangent theta0 at s = 0 (half-segment joint).
Solved by SQP: each Newton step is a convex QP (quadprog) with all node bounds as inequalities.
"""
import numpy as np
import quadprog

N_SEG = 150
N_NEWTON = 200
REG = 1e-9
STEP_TOL = 1e-6    # m: stop when a step moves no node more than 1 um (a lift-off point between
                   # two nodes makes the contact set toggle there with sub-um moves)
MAX_DPHI = 0.2
Z_TOL = 2e-5        # m, allowed penetration: contact edge falls between 2 mm nodes; << paper thickness


def _joints(length, creases, d, n):
    h = length / n
    D = np.full(n - 1, d)
    kbar = np.zeros(n - 1)
    for sc, th, k in creases:
        j = min(max(int(round(sc / h)) - 1, 0), n - 2)
        D[j], kbar[j] = k * h, th / h
    return h, D, kbar


def _nodes(y, h):
    return y[-1] + h * np.concatenate([[0.0], np.cumsum(np.sin(y[:-1]))])


def _xs(y, h):
    return h * np.concatenate([[0.0], np.cumsum(np.cos(y[:-1]))])


def _support(x, gap):
    """Two lower-hull nodes of (x, z - b) bracketing the centre of mass: where a rigid body rests."""
    hull = []                                      # lower convex hull (monotone chain)
    for i in np.argsort(x):
        while len(hull) > 1 and ((x[hull[-1]] - x[hull[-2]]) * (gap[i] - gap[hull[-2]])
                                 - (gap[hull[-1]] - gap[hull[-2]]) * (x[i] - x[hull[-2]])) <= 0:
            hull.pop()
        hull.append(i)
    xc = x.mean()
    k = next((m for m in range(len(hull) - 1) if x[hull[m + 1]] >= xc), len(hull) - 2)
    return int(hull[k]), int(hull[k + 1])


def _rest(creases, b, h, n, clamp):
    """Rigid folded strip: clamped at theta0, or resting on its two supporting obstacle nodes."""
    phi = np.zeros(n)
    for sc, th, _ in creases:
        phi[int(round(sc / h)):] += th
    if clamp is not None:
        return np.append(phi + clamp, 0.0), set()   # the clamp tangent at s = 0
    y = np.append(phi, 0.0)
    x, gap = _xs(y, h), _nodes(y, h) - b
    a, c = _support(x, gap)
    y[:-1] -= np.arctan2(gap[c] - gap[a], x[c] - x[a])
    y[-1] = b[a] - _nodes(y, h)[a]
    return y, {a, c}


def solve_strip(length, q, d, creases=(), supports=(), clamp=None, n=N_SEG):
    """creases: [(s_c, theta, k)] folds; supports: [(s_i, h_i)] line supports at arc length s_i,
    height h_i; clamp: edge angle theta0 at s = 0, z = 0, or None. Returns s, x, z node arrays.
    SQP: each Newton step is a convex QP with every node's bound z_j >= b_j as an inequality,
    so contacts open and close inside the step (no separate contact loop)."""
    h, D, kbar = _joints(length, creases, d, n)
    b = np.zeros(n + 1)                            # obstacle under each node (for the initial rest only)
    sj, sw = [], []                                # supports: nodes j, j+1 and weight w of the exact position
    for si, hi in supports:
        j = min(int(si / h), n - 1)
        sj.append(j)
        sw.append(si / h - j)
        b[j if sw[-1] < 0.5 else j + 1] = max(b[j if sw[-1] < 0.5 else j + 1], hi)
    sj, sw, sh = np.array(sj, int), np.array(sw), np.array([hi for _, hi in supports])
    y, _ = _rest(creases, b, h, n, clamp)
    free = np.arange(1 if clamp is not None else 0, n + 1)   # nodes that may touch the table
    idx, jj = np.arange(n), np.arange(n - 1)
    H0 = np.zeros((n + 1, n + 1))                  # bending part of the Hessian: constant
    H0[jj, jj] += D / h
    H0[jj + 1, jj + 1] += D / h
    H0[jj, jj + 1] -= D / h
    H0[jj + 1, jj] -= D / h
    H0[n, n] = REG                                 # z_0 has no stiffness: regularise the step only
    E = np.zeros((0, n + 1)) if clamp is None else np.eye(n + 1)[[n]]   # clamp: z_0 = 0; angle at s = 0 via
    e0 = np.zeros(len(E))                          # a half-segment joint to the clamp tangent (below)
    if clamp is not None:
        H0[0, 0] += 2 * d / h
    w = np.full(n + 1, q * h)                      # trapezoid node weights: total q * length
    w[[0, n]] *= 0.5
    wtail = np.cumsum(w[::-1])[::-1][1:]           # weight beyond segment i
    nodes = np.concatenate([free, sj, sj + 1])     # nodes in constraint rows
    Z = np.zeros((len(free) + len(sj), len(nodes)))   # constraint rows = Z @ (node heights)
    Z[np.arange(len(free)), np.arange(len(free))] = 1.0
    Z[len(free) + np.arange(len(sj)), len(free) + np.arange(len(sj))] = 1 - sw
    Z[len(free) + np.arange(len(sj)), len(free) + len(sj) + np.arange(len(sj))] = sw
    bound = np.concatenate([np.zeros(len(free)), sh])
    below = idx[None, :] < nodes[:, None]          # z_j depends on phi_i for i < j
    lam = np.zeros(len(bound))                     # contact reactions per constraint row
    for _ in range(N_NEWTON):
        c, s = np.cos(y[:-1]), np.sin(y[:-1])
        z = _nodes(y, h)
        r = D / h * (np.diff(y[:-1]) - h * kbar)
        g = np.append(h * c * wtail, q * length)   # gravity
        g[:-2] -= r
        g[1:-1] += r
        if clamp is not None:
            g[0] += 2 * d / h * (y[0] - clamp)
        f = np.zeros(n + 1)                        # reactions spread to nodes
        np.add.at(f, nodes, Z.T @ lam)
        tail = wtail - np.cumsum(f[::-1])[::-1][1:]   # net vertical load beyond i
        H = H0.copy()
        H[idx, idx] += np.abs(h * s * tail)       # |curvature of gravity + reactions|: convex QP
        J = Z @ np.hstack([h * c * below, np.ones((len(nodes), 1))])   # rows: d (constraint) / d y
        C = np.vstack([E, J]).T
        bq = np.concatenate([e0 - E @ y, bound - Z @ z[nodes]])
        dy, _, _, _, mult, _ = quadprog.solve_qp(H, -g, C, bq, len(E))
        a = min(1.0, MAX_DPHI / max(np.abs(dy[:-1]).max(), 1e-300))   # cap rotation: gravity is nonlinear
        y = y + a * dy
        lam = (1 - a) * lam + a * mult[len(E):]
        dz = a * (dy[-1] + h * np.concatenate([[0.0], np.cumsum(c * dy[:-1])]))   # node moves of this step
        dx = a * h * np.concatenate([[0.0], np.cumsum(-s * dy[:-1])])
        if max(np.abs(dz).max(), np.abs(dx).max()) < STEP_TOL:
            break
    else:
        raise RuntimeError(f"strip: no equilibrium for creases {creases}, supports {supports}, clamp {clamp}")
    return h * np.arange(n + 1), _xs(y, h), _nodes(y, h)


def solve_multifold(length, creases, q, d, n=N_SEG):
    return solve_strip(length, q, d, creases=creases, n=n)


def folded(length, *params, q, d):
    """params: (s_1..s_M, theta_1..theta_M, k_1..k_M), unused creases s = nan."""
    m = len(params) // 3
    creases = [(s, th, k) for s, th, k in zip(params[:m], params[m:2 * m], params[2 * m:])
               if np.isfinite(s)]
    return solve_multifold(length, creases, q, d)
