"""Fast engineered features of a 3D page map (page-frame grid G (360, 256, 3) [mm], from maps3d_emb.load): everything
is measured on the map itself.

Blocks (shared smoothing / curvature computed once per scale):
  height   Z level, spread, table contact, sides / corners, radial spectrum, low DCT, row / column profiles
  normal   tilt histogram, normal-field gradient |dn|, Gaussian image (normal histogram entropy, spread)
  curv     principal curvatures, K, H, curvedness and shape index (Koenderink) by scale
  band     band-pass (DoG) residuals of Z by scale, top-hat / median residuals (creases)
  ridge    Hessian line strength by scale, ridge components (count, extent)
  lines    straight lines of the ridge map (Radon peaks away from the page edge): count by strength, axis angle to
           the page side, bend angle across each line (normals on both sides), length; bend axis of the smooth surface
"""
import time

import cv2
import numpy as np
from scipy import ndimage
from scipy.fft import dctn

STEP = (210.0 / 255, 297.0 / 359)           # mm: nominal node steps across, down
DZ_MAX = 73.13                              # mm: largest frame Z over all maps
SIGMAS = (0, 1, 2, 4, 8, 12)                # nodes: surface smoothing of the curvature scales
SMALL = (128, 180)                          # px (w, h): working size of the line search
BAND = (4, 20)                              # px at SMALL: side bands of a line for its bend angle


def _blur(x, s, nearest=False):
    """Gaussian smoothing (OpenCV: much faster than scipy for large sigma)."""
    if not s:
        return x
    k = int(2 * round(4 * s) + 1)
    return cv2.GaussianBlur(np.ascontiguousarray(x, np.float32), (k, k), s, borderType=cv2.BORDER_REPLICATE if nearest else cv2.BORDER_REFLECT)


def _q(a):
    """|a|: mean, std, p50, p90, p99, max."""
    a = np.abs(a).ravel()
    return [a.mean(), a.std(), *np.percentile(a, (50, 90, 99)), a.max()]


def _forms(G, sigma):
    """Normals, k1 >= k2, K, H [1/mm] of G smoothed by sigma nodes, on a grid subsampled by sub(sigma) (smooth enough)."""
    f = sub(sigma)
    if sigma:
        G = _blur(G, sigma, True)[::f, ::f]
    du, dv = STEP[0] * f, STEP[1] * f
    Gu, Gv = np.gradient(G, du, axis=1), np.gradient(G, dv, axis=0)
    Guu, Guv, Gvv = np.gradient(Gu, du, axis=1), np.gradient(Gu, dv, axis=0), np.gradient(Gv, dv, axis=0)
    n = np.cross(Gu, -Gv)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    E, F, Gg = (Gu * Gu).sum(-1), (Gu * Gv).sum(-1), (Gv * Gv).sum(-1)
    L, M, N = (Guu * n).sum(-1), (Guv * n).sum(-1), (Gvv * n).sum(-1)
    det = np.maximum(E * Gg - F ** 2, 1e-12)
    K = (L * N - M ** 2) / det
    H = (E * N - 2 * F * M + Gg * L) / (2 * det)
    r = np.sqrt(np.maximum(H ** 2 - K, 0))
    return n, H + r, H - r, K, H


def sub(sigma):
    return max(1, min(4, int(sigma // 2)))


def _hessian_line(Z, s):
    """Line strength (strong curvature in one direction only) and the scale-normalised Hessian of Z at s."""
    gr, gc = np.gradient(_blur(Z, s))
    Hrr, Hrc, Hcc = s * s * np.gradient(gr, axis=0), s * s * np.gradient(gr, axis=1), s * s * np.gradient(gc, axis=1)
    tmp = np.sqrt((Hrr - Hcc) ** 2 + 4 * Hrc ** 2)
    l1, l2 = 0.5 * (Hrr + Hcc + tmp), 0.5 * (Hrr + Hcc - tmp)
    big = np.abs(l1) > np.abs(l2)
    lam, oth = np.where(big, l1, l2), np.where(big, l2, l1)
    return np.abs(lam) * np.exp(-4 * (oth / (lam + 1e-12)) ** 2), (Hrr, Hrc, Hcc)


def _radon(img, n_ang=90):
    """Mean of img along lines: (angles [deg], A (n_ang, offsets), N pixels per line, pad, offsets of the image)."""
    h, w = img.shape
    pad = int(np.ceil(np.hypot(h, w))) + 2
    P = np.zeros((pad, pad), np.float32)
    y0, x0 = (pad - h) // 2, (pad - w) // 2
    P[y0:y0 + h, x0:x0 + w] = img
    O = np.zeros_like(P)
    O[y0:y0 + h, x0:x0 + w] = 1
    angs = np.linspace(0, 180, n_ang, endpoint=False)
    S, N = [], []
    for a in angs:
        R = cv2.getRotationMatrix2D((pad / 2, pad / 2), a, 1)
        S.append(cv2.warpAffine(P, R, (pad, pad)).sum(0))
        N.append(cv2.warpAffine(O, R, (pad, pad)).sum(0))
    S, N = np.array(S), np.array(N)
    return angs, np.where(N > 30, S / np.maximum(N, 1), 0), N, pad, (y0, x0)


def _side(a):
    """Angle of a line at image angle a [deg] to the page side (90: square across the page)."""
    return 90 - min(a % 180, 180 - a % 180)


def lines(Z, n, thr=0.01):
    """Straight ridge lines: list of dict(axis, bend, strength, length), strongest first. Peaks whose side band
    leaves the page (edge artefacts) are dropped."""
    small = lambda x: cv2.resize(x.astype(np.float32), SMALL, interpolation=cv2.INTER_AREA)
    r = small(_hessian_line(Z, 2)[0])
    angs, A, N, pad, (y0, x0) = _radon(r)
    pk = (A == ndimage.maximum_filter(A, (15, 9), mode="wrap")) & (A > thr)
    ns = np.stack([small(n[..., k]) for k in range(3)], -1)
    yy, xx = np.mgrid[0:r.shape[0], 0:r.shape[1]]
    out = []
    for ia, io in zip(*np.nonzero(pk)):
        R = cv2.getRotationMatrix2D((pad / 2, pad / 2), angs[ia], 1)
        d = R[0, 0] * (xx + x0) + R[0, 1] * (yy + y0) + R[0, 2] - io
        a_, b_ = (d >= BAND[0]) & (d <= BAND[1]), (d <= -BAND[0]) & (d >= -BAND[1])
        if a_.sum() < 20 or b_.sum() < 20:
            continue
        na, nb = ns[a_].mean(0), ns[b_].mean(0)
        bend = np.degrees(np.arccos(np.clip(na @ nb / np.linalg.norm(na) / np.linalg.norm(nb), -1, 1)))
        out.append(dict(axis=_side(angs[ia]), bend=float(bend), strength=float(A[ia, io]), length=float(N[ia, io] / SMALL[1])))
    return sorted(out, key=lambda x: -x["strength"])


def feats(G, timing=False):
    """-> (float32 vector, names) [, seconds per block]."""
    G = G.astype(np.float32)
    Z = G[..., 2]
    F, names, dt = [], [], {}

    def add(block, keys, vals):
        F.extend(vals)
        names.extend(f"{block}.{k}" for k in keys)

    t = time.time()
    fm = {s: _forms(G, s) for s in SIGMAS}
    dt["forms"] = time.time() - t

    # height
    t = time.time()
    add("height", ["max", "mean", "std", "p10", "p50", "p90", "lt1", "lt5"],
        [Z.max(), Z.mean(), Z.std(), *np.percentile(Z, (10, 50, 90)), (Z < 1).mean(), (Z < 5).mean()])
    add("height", [f"side{i}" for i in range(4)] + [f"corner{i}" for i in range(4)] + ["centre"],
        [*np.sort([Z[0].mean(), Z[-1].mean(), Z[:, 0].mean(), Z[:, -1].mean()]),
         *np.sort([Z[0, 0], Z[0, -1], Z[-1, 0], Z[-1, -1]]), Z[140:220, 100:156].mean()])
    P = np.abs(np.fft.rfft2(Z - Z.mean())) ** 2
    fy, fx = np.meshgrid(np.fft.fftfreq(Z.shape[0]), np.fft.rfftfreq(Z.shape[1]), indexing="ij")
    rr, band = np.hypot(fy, fx), np.geomspace(0.003, 0.5, 11)
    add("height", [f"psd{i}" for i in range(10)], [np.log10(P[(rr >= a) & (rr < b)].mean() + 1e-9) for a, b in zip(band[:-1], band[1:])])
    add("height", [f"dct{i}" for i in range(36)],
        list((dctn(cv2.resize(Z.astype(np.float32), (32, 32), interpolation=cv2.INTER_AREA), norm="ortho")[:6, :6] / DZ_MAX).ravel()))
    add("height", ["colprof_std", "rowprof_std", "colprof_tv", "rowprof_tv"],
        [Z.mean(0).std(), Z.mean(1).std(), np.abs(np.diff(Z.mean(0))).sum(), np.abs(np.diff(Z.mean(1))).sum()])
    dt["height"] = time.time() - t

    # normal
    t = time.time()
    for s in (0, 2, 8):
        n = fm[s][0]
        tilt = np.degrees(np.arccos(np.clip(n[..., 2], -1, 1)))
        add("normal", [f"s{s}.tilt_{k}" for k in ("mean", "p90", "max", *range(6))],
            [tilt.mean(), np.percentile(tilt, 90), tilt.max(), *np.histogram(tilt, (0, 5, 10, 20, 30, 45, 90))[0] / tilt.size])
        dn = np.sqrt(sum(np.gradient(n[..., k], axis=a) ** 2 for k in range(3) for a in (0, 1))) / sub(s)
        add("normal", [f"s{s}.dn_{k}" for k in ("mean", "std", "p50", "p90", "p99", "max", "gt.01", "gt.03", "gt.1", "gt.3")],
            _q(dn) + [(dn > v).mean() for v in (0.01, 0.03, 0.1, 0.3)])
    for s in (1, 4):
        n = fm[s][0]
        h = np.histogram2d(n[..., 0].ravel(), n[..., 1].ravel(), 24, ((-1, 1), (-1, 1)))[0] / n[..., 0].size
        p = h[h > 0]
        ev = np.linalg.eigvalsh(np.cov(np.stack([n[..., 0].ravel(), n[..., 1].ravel()])))
        add("normal", [f"s{s}.gauss_{k}" for k in ("entropy", "bins01", "bins05", "max", "ev0", "ev1", "ratio")],
            [-(p * np.log(p)).sum(), (h > 0.01).sum(), (h > 0.05).sum(), h.max(), *ev, ev[0] / (ev[1] + 1e-12)])
    dt["normal"] = time.time() - t

    # curv
    t = time.time()
    for s in SIGMAS:
        _, k1, k2, K, H = fm[s]
        c = np.sqrt((k1 ** 2 + k2 ** 2) / 2)
        kk = np.maximum(np.abs(k1), np.abs(k2))
        si = 2 / np.pi * np.arctan2(k1 + k2, k1 - k2)
        add("curv", [f"s{s}.kmax_{k}" for k in ("mean", "std", "p50", "p90", "p99", "max")], _q(kk))
        add("curv", [f"s{s}.K_{k}" for k in ("mean", "std", "p50", "p90", "p99", "max")], _q(K))
        add("curv", [f"s{s}.{k}" for k in ("absH", "kmin_kmax", "c_gt.01", "c_gt.03", "c_gt.1", "c_gt.3")],
            [np.abs(H).mean(), np.minimum(np.abs(k1), np.abs(k2)).mean() / (kk.mean() + 1e-12), *[(c > v).mean() for v in (0.01, 0.03, 0.1, 0.3)]])
        add("curv", [f"s{s}.si{i}" for i in range(9)], list(np.histogram(si, 9, (-1, 1), weights=c)[0] / max(c.sum(), 1e-12)))
    dt["curv"] = time.time() - t

    # band: DoG of Z, top-hat and median residuals (median on a reduced grid for the large kernels)
    t = time.time()
    sm = [_blur(Z, s) for s in (0.5, 1, 2, 4, 8, 16, 32)]
    for i, (a, b) in enumerate(zip(sm[:-1], sm[1:])):
        r = a - b
        add("band", [f"dog{i}_{k}" for k in ("mean", "std", "p50", "p90", "p99", "max", "gt.02", "gt.05", "gt.1", "gt.2", "gt.5", "kurt")],
            _q(r) + [(np.abs(r) > v).mean() for v in (0.02, 0.05, 0.1, 0.2, 0.5)] + [((r - r.mean()) ** 4).mean() / (r.var() ** 2 + 1e-12)])
    for k in (5, 9, 17, 33):
        f = max(1, k // 4)
        Zs = Z[::f, ::f] if f > 1 else Z
        kk = k // f | 1
        up = (lambda x: cv2.resize(x.astype(np.float32), Z.shape[::-1], interpolation=cv2.INTER_LINEAR)) if f > 1 else (lambda x: x)
        for nm, r in (("tophat", Z - up(ndimage.grey_opening(Zs, (kk, kk)))), ("blackhat", up(ndimage.grey_closing(Zs, (kk, kk))) - Z),
                      ("median", Z - up(ndimage.median_filter(Zs, (kk, kk))))):
            add("band", [f"{nm}{k}_{q}" for q in ("mean", "std", "p50", "p90", "p99", "max", "gt1e-3", "gt3e-3", "gt1e-2", "gt3e-2")],
                _q(r) + [(np.abs(r) > v).mean() for v in (1e-3, 3e-3, 1e-2, 3e-2)])
    dt["band"] = time.time() - t

    # ridge
    t = time.time()
    for s in (1, 2, 4, 8):
        ln = _hessian_line(Z, s)[0]
        add("ridge", [f"s{s}_{k}" for k in ("mean", "std", "p50", "p90", "p99", "max")], _q(ln))
        for v in (0.02, 0.1):
            mask = ln > v
            lab, k = ndimage.label(mask)
            if k:
                sizes = np.bincount(lab.ravel())[1:]
                ext = np.array([max(o[0].stop - o[0].start, o[1].stop - o[1].start) for o in ndimage.find_objects(lab)])
                big = sizes >= 20
                vals = [k, mask.mean(), sizes.max(), ext.max(), (ext > 100).sum(), (ext > 200).sum(),
                        (ext[big] ** 2 / sizes[big]).max() if big.any() else 0, big.sum()]
            else:
                vals = [0] * 8
            add("ridge", [f"s{s}_t{v}_{q}" for q in ("n", "area", "size", "ext", "ext100", "ext200", "elong", "nbig")], vals)
    dt["ridge"] = time.time() - t

    # lines
    t = time.time()
    L = lines(Z, fm[1][0])  # sub(1) == 1: full grid
    st = [x["strength"] for x in L]
    top = (L + [dict(axis=-1.0, bend=0.0, strength=0.0, length=0.0)] * 3)[:3]
    _, (Hrr, Hrc, Hcc) = _hessian_line(Z, 16)
    T = np.array([[(Hrr ** 2 + Hrc ** 2).mean(), (Hrc * (Hrr + Hcc)).mean()], [(Hrc * (Hrr + Hcc)).mean(), (Hcc ** 2 + Hrc ** 2).mean()]])
    ev, V = np.linalg.eigh(T)
    add("lines", ["n.02", "n.05", "n.1", "n.2"], [sum(v > q for v in st) for q in (0.02, 0.05, 0.1, 0.2)])
    add("lines", [f"l{i}_{k}" for i in range(3) for k in ("strength", "axis", "bend", "length")],
        [x[k] for x in top for k in ("strength", "axis", "bend", "length")])
    add("lines", ["bend_sum", "axis_spread", "bend_axis", "bend_aniso"],
        [sum(x["bend"] for x in L if x["strength"] > 0.05), np.std([x["axis"] for x in L]) if len(L) > 1 else 0.0,
         _side(np.degrees(np.arctan2(V[0, 1], V[1, 1])) + 90), np.log1p(ev[1] / (ev[0] + 1e-12))])
    dt["lines"] = time.time() - t

    v = np.nan_to_num(np.array(F, np.float32), posinf=0, neginf=0)
    return (v, names, dt) if timing else (v, names)
