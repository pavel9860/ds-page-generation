"""Strip with N plastic creases resting on a table, as a discrete elastica.

Unknowns y = (phi_0..phi_{n-1}, z_0); nodes z_j = z_0 + h sum_{i<j} sin phi_i.
    E = sum_j D_j/(2h) (dphi_j - h kbar_j)^2 + q h sum_j z_j
A crease (s_c, theta, k) is a hinge: joint stiffness k*h, rest jump theta, i.e.
dphi = theta - M/k. theta > 0 valley, < 0 mountain.
Table: exact constraints z_j = 0 on a contact set (Newton-KKT), updated until
no node is below the table and every contact pushes up.
"""
import numpy as np
from threadpoolctl import threadpool_limits

N_SEG = 150
N_NEWTON = 30
N_CONTACT = 60
MAX_DPHI = 0.2


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


def _rest(length, creases, n):
    """Rigid folded strip placed on the two lower-hull nodes that bracket its centre of mass."""
    h = length / n
    phi = np.zeros(n)
    for sc, th, _ in creases:
        phi[int(round(sc / h)):] += th
    x = h * np.concatenate([[0.0], np.cumsum(np.cos(phi))])
    z = h * np.concatenate([[0.0], np.cumsum(np.sin(phi))])
    hull = []                                      # lower convex hull (monotone chain)
    for i in np.argsort(x):
        while len(hull) > 1 and ((x[hull[-1]] - x[hull[-2]]) * (z[i] - z[hull[-2]])
                                 - (z[hull[-1]] - z[hull[-2]]) * (x[i] - x[hull[-2]])) <= 0:
            hull.pop()
        hull.append(i)
    xc = x.mean()
    k = next((m for m in range(len(hull) - 1) if x[hull[m + 1]] >= xc), len(hull) - 2)
    a, b = hull[k], hull[k + 1]
    phi -= np.arctan2(z[b] - z[a], x[b] - x[a])
    y = np.append(phi, 0.0)
    y[-1] = -_nodes(y, h)[a]
    return y, {a, b}


def solve_multifold(length, creases, q, d, n=N_SEG):
    """creases: [(s_c, theta, k)]. Returns s, x, z node arrays.
    Newton-KKT for a fixed contact set; then add the lowest node of each run below
    the table and release contacts that pull (lam < 0), until neither happens."""
    h, D, kbar = _joints(length, creases, d, n)
    y, act = _rest(length, creases, n)
    idx, jj = np.arange(n), np.arange(n - 1)
    with threadpool_limits(1):                     # small dense systems: BLAS threads only add overhead
        for _ in range(N_CONTACT):
            A = np.array(sorted(act))
            lam = np.zeros(len(A))
            for _ in range(N_NEWTON):
                phi = y[:-1]
                c, s = np.cos(phi), np.sin(phi)
                r = D / h * (np.diff(phi) - h * kbar)
                lam_n = np.zeros(n + 1)
                lam_n[A] = lam
                tail = np.cumsum((q * h - lam_n)[::-1])[::-1][1:]   # net vertical load beyond i
                g = np.append(h * c * tail, (n + 1) * q * h - lam.sum())
                g[:-2] -= r
                g[1:-1] += r
                H = np.zeros((n + 1, n + 1))
                H[jj, jj] += D / h
                H[jj + 1, jj + 1] += D / h
                H[jj, jj + 1] -= D / h
                H[jj + 1, jj] -= D / h
                H[idx, idx] -= h * s * tail
                J = np.zeros((len(A), n + 1))
                J[:, :-1] = h * c * (idx[None, :] < A[:, None])
                J[:, -1] = 1.0
                K = np.block([[H, -J.T], [J, np.zeros((len(A), len(A)))]])
                sol = np.linalg.solve(K, -np.concatenate([g, _nodes(y, h)[A]]))
                a = min(1.0, MAX_DPHI / np.abs(sol[:n]).max())   # cap rotation per step: gravity is nonlinear
                y, lam = y + a * sol[:n + 1], lam + a * sol[n + 1:]
                if np.abs(sol[:n + 1]).max() < 1e-10:
                    break
            z = _nodes(y, h)
            below = z < -1e-7
            runs = np.split(np.nonzero(below)[0], np.nonzero(np.diff(np.nonzero(below)[0]) > 1)[0] + 1)
            add = {int(r[np.argmin(z[r])]) for r in runs if len(r)}
            pull = set(A[lam < 0].tolist())
            if not add and not pull:
                break
            act = (act | add) - pull
    x = h * np.concatenate([[0.0], np.cumsum(np.cos(y[:-1]))])
    return h * np.arange(n + 1), x, _nodes(y, h)


def folded(length, *params, q, d):
    """params: (s_1..s_M, theta_1..theta_M, k_1..k_M), unused creases s = nan."""
    m = len(params) // 3
    creases = [(s, th, k) for s, th, k in zip(params[:m], params[m:2 * m], params[2 * m:])
               if np.isfinite(s)]
    return solve_multifold(length, creases, q, d)
