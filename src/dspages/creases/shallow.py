"""Shallow creases on a flat page: fast height map (mm) from a template fitted to a physics solution.

Template (fold -> press flat at yield strain 0.5 % -> release under gravity on the table; FvK shell +
fine-scale Kirchhoff plate with table contact; straight mountain creases L = 4..50 mm, fit error <= 10 %):
    w(d, s) = h(L) * A(s) * exp(-(sqrt(d^2 + r^2) - r) / lam(L))
    h = 0.0547 L^0.80 mm,  lam = 1.91 L^0.45 mm,  r = max(0.05, 0.07 L - 0.6) mm   (fitted for L <= 50 mm;
    longer creases use the L = 50 values: no extrapolation),
    A = 1 - 0.35 u^2 (2 - u^2), u = 2 s / L - 1 along the crease (tip value 0.65 as fitted, zero slope at the
    tips so the field has no kink there), d = distance to the centerline (radial past the tips).
Distance-to-curve fields have a slope kink where two parts of a curve are equally near (inside an arch);
the elastic solution has none, so away from the ridge each crease field is Gaussian-smoothed (sigma lam/4).
Valley = the same profile negated. Crease severity k scales h (residual angle ~ yield strain).
Overlapping creases add, the total saturates smoothly: w_max tanh(w / w_max). All lengths in mm."""
import cv2
import numpy as np
from numba import njit, prange


def height(L):
    return 0.0547 * L ** 0.80


def lam(L):
    return 1.91 * L ** 0.45


def r_ridge(L):
    return max(0.05, 0.07 * L - 0.6)


# ---------------- lines ----------------
def arch(rng, p0, p1, sag=(0.0, 0.25), peak=(0.15, 0.85)):
    """Cubic Bezier p0 -> p1 with both controls on one side: a smooth one-sided arch, apex at fraction
    `peak` of the chord, sagitta sag * chord (0 = straight)."""
    c = p1 - p0
    L = np.linalg.norm(c)
    t = c / L
    n = np.array([-t[1], t[0]]) * rng.choice([-1, 1])
    s, a = rng.uniform(*sag) * L, rng.uniform(*peak)
    k1 = p0 + t * L * (2 * a / 3) + n * s * 4 / 3
    k2 = p0 + t * L * (a + (1 - a) / 3) + n * s * 4 / 3
    u = np.linspace(0, 1, max(int(L / 0.1), 8))[:, None]
    return (1 - u) ** 3 * p0 + 3 * (1 - u) ** 2 * u * k1 + 3 * (1 - u) * u ** 2 * k2 + u ** 3 * p1


def length(rng, median=30.0, sigma=0.6, lo=5.0, hi=120.0):
    return float(np.clip(rng.lognormal(np.log(median), sigma), lo, hi))


def ray(rng, p, th, L, **kw):
    return arch(rng, p, p + L * np.array([np.cos(th), np.sin(th)]), **kw)


def _tangent(line, k):
    d = line[min(k + 1, len(line) - 1)] - line[max(k - 1, 0)]
    return np.arctan2(d[1], d[0])


def motif(rng, c, th, kind):
    """Centerlines of one crease unit at c with main direction th."""
    if kind == "single":                                # straight to gently curved, or a hook
        L = length(rng)
        e = L / 2 * np.array([np.cos(th), np.sin(th)])
        hook = rng.random() < 0.2
        return [arch(rng, c - e, c + e, sag=(0.25, 0.45) if hook else (0.0, 0.12))]
    if kind == "chain":                                 # 2-3 creases end to end, kinked
        lines = [ray(rng, c, th, length(rng) * 0.6, sag=(0.0, 0.12))]
        for _ in range(rng.integers(1, 3)):
            a = lines[-1]
            lines.append(ray(rng, a[-1], _tangent(a, len(a) - 1) + rng.choice([-1, 1]) * rng.uniform(0.2, 1.0),
                             length(rng) * 0.6, sag=(0.0, 0.12)))
        return lines
    if kind == "star":                                  # 3-4 creases meeting at a vertex
        n = rng.integers(3, 5)
        a = th + np.sort(rng.uniform(0, 2 * np.pi, n))
        return [ray(rng, c, x, length(rng) * 0.7, sag=(0.0, 0.12)) for x in a]
    if kind == "V":
        return [ray(rng, c, th, length(rng) * 0.7), ray(rng, c, th + rng.uniform(0.35, 2.9), length(rng) * 0.7)]
    if kind == "T":                                     # a crease branching off another
        a = motif(rng, c, th, "single")[0]
        k = rng.integers(len(a) // 5, 4 * len(a) // 5)
        return [a, ray(rng, a[k], _tangent(a, k) + rng.choice([-1, 1]) * rng.uniform(0.35, 2.8), length(rng) * 0.6)]
    # "X": two creases crossing
    return [motif(rng, c, th, "single")[0], motif(rng, c, th + rng.uniform(0.4, 2.7), "single")[0]]


KINDS = {"single": 0.45, "chain": 0.15, "T": 0.15, "star": 0.10, "V": 0.10, "X": 0.05}


def sample(rng, W, H, mean_groups=1.8, p_isolated=0.3, mean_extra=1.5, radius=(15.0, 45.0), spread_deg=25.0,
           singles_per_group=(3.0, 4.0), k=(0.15, 0.9)):
    """Creases of one page. Groups ~ 1 + Poisson(mean_groups - 1) at random places. A group is one isolated
    unit (p_isolated) or a loose group of 2 + Poisson(mean_extra) units within its radius sharing a main
    direction (+- spread_deg). A unit is a single crease, a kinked chain, a T, a star vertex, a V or an X.
    Outside the groups: Poisson(U(singles_per_group) * n_groups) single creases, anywhere, any direction.
    Each crease gets severity k and valley/mountain sign; group = -1 for the stand-alone singles."""
    kinds, prob = list(KINDS), list(KINDS.values())
    out = []
    n_groups = 1 + rng.poisson(mean_groups - 1)
    for g in range(n_groups):
        m = 1 if rng.random() < p_isolated else 2 + rng.poisson(mean_extra)
        R = rng.uniform(*radius)
        c0 = rng.uniform([R, R], [W - R, H - R])
        th0 = rng.uniform(0, np.pi)
        for j in range(m):
            c = c0 if j == 0 else c0 + rng.normal(0, R / 2, 2)
            th = th0 + rng.normal(0, np.radians(spread_deg)) + rng.choice([0, np.pi])
            for line in motif(rng, c, th, kinds[rng.choice(len(kinds), p=prob)]):
                out.append(dict(line=line, k=rng.uniform(*k), sign=rng.choice([-1, 1]), group=g))
    for _ in range(rng.poisson(rng.uniform(*singles_per_group) * n_groups)):
        line = motif(rng, rng.uniform([0, 0], [W, H]), rng.uniform(0, np.pi), "single")[0]
        out.append(dict(line=line, k=rng.uniform(*k), sign=rng.choice([-1, 1]), group=-1))
    return out


# ---------------- height map ----------------
CH = 8                         # segments per culling chunk


@njit(parallel=True, fastmath=True, cache=True)
def _fields(P, S, B, cp, cb, box, prm, off, rows, f, keep):
    """All crease boxes of a page in one parallel pass over their rows. Crease q: polyline P[cp[q]:cp[q+1]]
    (arclength S), culling chunks B[cb[q]:cb[q+1]], box (x0, y0, ny, nx) with pitch h in prm, output at
    f[off[q]:]. Per pixel: exact distance / arclength to the polyline, template value f and ridge weight
    keep = exp(-(d / (lam / 2))^2). Culling: chunks of CH segments with bounding boxes; the previous pixel's
    nearest chunk is tried first (neighbours share it), then only chunks nearer than the best so far."""
    for rr in prange(rows.shape[0]):
        q = rows[rr, 0]
        i = rows[rr, 1]
        p0 = cp[q]
        nseg = cp[q + 1] - p0 - 1
        b0 = cb[q]
        nc = cb[q + 1] - b0
        x0 = box[q, 0]
        py = box[q, 1] + i * prm[q, 0]
        nx = int(box[q, 3])
        h = prm[q, 0]
        L = prm[q, 1]
        lm = prm[q, 2]
        r = prm[q, 3]
        dmax = prm[q, 4]
        amp = prm[q, 5]
        gx0 = B[b0:b0 + nc, 0].min()
        gy0 = B[b0:b0 + nc, 1].min()
        gx1 = B[b0:b0 + nc, 2].max()
        gy1 = B[b0:b0 + nc, 3].max()
        o = off[q] + i * nx
        cprev = 0
        for j in range(nx):
            px = x0 + j * h
            ex = max(gx0 - px, 0.0, px - gx1)
            ey = max(gy0 - py, 0.0, py - gy1)
            if ex * ex + ey * ey >= dmax * dmax:            # whole line beyond the cutoff
                f[o + j] = 0.0
                keep[o + j] = 0.0
                continue
            best = 1e30
            bs = 0.0
            cbest = cprev
            for cc in range(-1, nc):
                c = cprev if cc < 0 else cc
                if cc >= 0:
                    if c == cprev:
                        continue
                    ex = max(B[b0 + c, 0] - px, 0.0, px - B[b0 + c, 2])
                    ey = max(B[b0 + c, 1] - py, 0.0, py - B[b0 + c, 3])
                    if ex * ex + ey * ey >= best:
                        continue
                for k in range(CH * c, min(CH * c + CH, nseg)):
                    ax = P[p0 + k, 0]
                    ay = P[p0 + k, 1]
                    bx = P[p0 + k + 1, 0] - ax
                    by = P[p0 + k + 1, 1] - ay
                    qx = px - ax
                    qy = py - ay
                    t = min(max((qx * bx + qy * by) / max(bx * bx + by * by, 1e-12), 0.0), 1.0)
                    dx = qx - t * bx
                    dy = qy - t * by
                    d2 = dx * dx + dy * dy
                    if d2 < best:
                        best = d2
                        bs = S[p0 + k] + t * (S[p0 + k + 1] - S[p0 + k])
                        cbest = c
            cprev = cbest
            d = np.sqrt(best)
            if d >= dmax:
                f[o + j] = 0.0
                keep[o + j] = 0.0
                continue
            u = 2.0 * bs / L - 1.0
            fade = 1.0 - (d / dmax) ** 2
            f[o + j] = (amp * (1.0 - 0.35 * u * u * (2.0 - u * u)) * fade * fade
                        * np.exp(-(np.sqrt(d * d + r * r) - r) / lm))
            keep[o + j] = np.exp(-(2.0 * d / lm) ** 2)


def render(creases, W, H, h=0.5, w_max=1.0, reach=4.0, step=0.5):
    """Height map [mm] on a W x H mm page, pitch h; +: mountain. Per crease, in its box (reach * lam):
    exact distance to the centerline polyline (resampled every `step` mm; chord error < 0.01 mm), template
    value, tail faded to 0 at reach * lam by (1 - (d / d_max)^2)^2 so the box edge leaves no step.
    Away from the ridge each crease field is blended with its Gaussian blur (sigma lam / 4)."""
    ny, nx = int(round(H / h)) + 1, int(round(W / h)) + 1
    w = np.zeros((ny, nx), np.float32)
    Ps, Ss, Bs, cp, cb, box, prm, off, rows, ij = [], [], [], [0], [0], [], [], [0], [], []
    for c in creases:
        ln = c['line']
        seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(ln, axis=0), axis=1))]
        L = seg[-1]
        Lf = min(L, 50.0)
        lm = lam(Lf)
        dmax = reach * lm
        j0, j1 = max(int((ln[:, 0].min() - dmax) / h), 0), min(int((ln[:, 0].max() + dmax) / h) + 2, nx)
        i0, i1 = max(int((ln[:, 1].min() - dmax) / h), 0), min(int((ln[:, 1].max() + dmax) / h) + 2, ny)
        if i1 <= i0 or j1 <= j0:
            continue
        idx = np.unique(np.r_[np.searchsorted(seg, np.arange(0, L, step)), len(ln) - 1])
        P = ln[idx]
        Bq = np.array([np.r_[P[k:k + CH + 1].min(0), P[k:k + CH + 1].max(0)] for k in range(0, len(P) - 1, CH)])
        q = len(box)
        Ps.append(P)
        Ss.append(seg[idx])
        Bs.append(Bq)
        cp.append(cp[-1] + len(P))
        cb.append(cb[-1] + len(Bq))
        box.append((j0 * h, i0 * h, i1 - i0, j1 - j0))
        off.append(off[-1] + (i1 - i0) * (j1 - j0))
        prm.append((h, L, lm, r_ridge(Lf), dmax, c['sign'] * c['k'] * height(Lf)))
        rows.append(np.c_[np.full(i1 - i0, q), np.arange(i1 - i0)])
        ij.append((i0, i1, j0, j1, lm))
    if not box:
        return w
    f = np.empty(off[-1], np.float32)
    keep = np.empty_like(f)
    _fields(np.concatenate(Ps), np.concatenate(Ss), np.concatenate(Bs), np.array(cp), np.array(cb),
            np.array(box), np.array(prm), np.array(off), np.concatenate(rows), f, keep)
    for q, (i0, i1, j0, j1, lm) in enumerate(ij):
        fq = f[off[q]:off[q + 1]].reshape(i1 - i0, j1 - j0)
        kq = keep[off[q]:off[q + 1]].reshape(fq.shape)
        sm = cv2.GaussianBlur(fq, (0, 0), 0.25 * lm / h, borderType=cv2.BORDER_CONSTANT)
        w[i0:i1, j0:j1] += kq * fq + (1 - kq) * sm                  # sharp ridge kept, smoothed elsewhere
    return w_max * np.tanh(w / w_max)
