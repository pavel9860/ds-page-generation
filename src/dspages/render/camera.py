"""Pinhole camera over the page: pose from the view pool, focal length and principal point so the page fills
the frame. Table plane z = 0, page above it, all lengths in mm; camera axes x right, y down, z forward."""

import numpy as np
from numba import njit

from ..config import CameraCfg


def vertex_normals(X, Y, Z):
    P = np.stack([X, Y, Z], -1)
    du = np.gradient(P, axis=1)
    dv = np.gradient(P, axis=0)
    n = np.cross(du, dv)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    return n * np.sign(n[..., 2:3] + 1e-12)


def look_at(eye, target, roll):
    """Rotation (rows: camera x right, y down, z forward) looking at target, image down along the page's -Y
    (page top at the image top), then turned by roll."""
    f = target - eye
    f /= np.linalg.norm(f)
    down = np.array([0.0, -1.0, 0.0]) if abs(f[1]) < 0.99 else np.array([0.0, 0.0, -1.0])
    y = down - f * (down @ f)
    y /= np.linalg.norm(y)
    x = np.cross(y, f)
    c, s = np.cos(roll), np.sin(roll)
    return np.stack([c * x + s * y, -s * x + c * y, f])


def project(P, cam):
    q = (P - cam["eye"]) @ cam["R"].T
    return np.stack([cam["f"] * q[:, 0] / q[:, 2] + cam["cx"], cam["f"] * q[:, 1] / q[:, 2] + cam["cy"]], -1), q[:, 2]


@njit(cache=True, fastmath=True)
def _worst_cos(eyes, P, N):
    """Smallest cosine between the ray to each eye and the surface normal over all points."""
    out = np.empty(eyes.shape[0])
    for t in range(eyes.shape[0]):
        m = 1.0
        ex, ey, ez = eyes[t, 0], eyes[t, 1], eyes[t, 2]
        for i in range(P.shape[0]):
            rx, ry, rz = ex - P[i, 0], ey - P[i, 1], ez - P[i, 2]
            c = (rx * N[i, 0] + ry * N[i, 1] + rz * N[i, 2]) / np.sqrt(rx * rx + ry * ry + rz * rz)
            if c < m:
                m = c
        out[t] = m
    return out


@njit(cache=True, fastmath=True)
def _nearest(dirs, target, P, N, cos_max, cap, first):
    """Per view direction d the smallest distance D from target with every point P (unit normals N) seen within
    arccos(cos_max) of its normal from target + D d and from any farther eye on that ray (per point the condition
    N.(w + D d) >= cos_max |w + D d|, w = target - P, holds beyond the larger root of a quadratic in D, and only when
    N.d > cos_max); inf when no distance up to cap[d] is (the points stop being checked then). first: stop at the first
    possible direction (the others stay inf)."""
    c2 = cos_max * cos_max
    out = np.full(dirs.shape[0], np.inf)
    for i in range(dirs.shape[0]):
        dx, dy, dz = dirs[i, 0], dirs[i, 1], dirs[i, 2]
        need = 0.0
        for j in range(P.shape[0]):
            wx, wy, wz = target[0] - P[j, 0], target[1] - P[j, 1], target[2] - P[j, 2]
            a = N[j, 0] * dx + N[j, 1] * dy + N[j, 2] * dz
            if a <= cos_max:
                need = np.inf
                break
            b = N[j, 0] * wx + N[j, 1] * wy + N[j, 2] * wz
            dw = dx * wx + dy * wy + dz * wz
            A, B, Cq = a * a - c2, 2.0 * (a * b - c2 * dw), b * b - c2 * (wx * wx + wy * wy + wz * wz)
            disc = B * B - 4.0 * A * Cq
            d = max((-B + np.sqrt(disc)) / (2.0 * A) if disc >= 0 else -np.inf, -b / a)
            if d > need:
                need = d
                if need > cap[i]:
                    need = np.inf
                    break
        out[i] = need
        if first and need < np.inf:
            break
    return out


def views(P, N, c: CameraCfg, offset, first=False):
    """The possible views of a page grid (P, N: (nv, nu, 3) points [mm] and unit normals): view_dirs directions spread
    evenly over the tilts (up to max_tilt_deg) an offset of offset x the page size allows from dist_mm[0], overhead first, each with its
    distances [lo, hi] in dist_mm where every view_step-th node of the grid is seen within incidence_deg of its normal and
    the camera's horizontal offset from the page centre stays within the offset. first: only up to the first possible
    direction. -> target, dirs (m, 3), tilts, azimuths, lo, hi (lo > hi: the direction is not possible)."""
    target = 0.5 * (P.min((0, 1)) + P.max((0, 1)))
    reach = offset * float(max(np.ptp(P[..., 0]), np.ptp(P[..., 1])))
    k = np.arange(c.view_dirs)
    azs = k * np.pi * (3 - np.sqrt(5))
    tilts = min(np.arcsin(min(1.0, reach / c.dist_mm[0])), np.radians(c.max_tilt_deg)) * np.sqrt(
        (k + 0.5) / c.view_dirs
    )
    dirs = np.stack([np.sin(tilts) * np.cos(azs), np.sin(tilts) * np.sin(azs), np.cos(tilts)], -1)
    hi = np.minimum(c.dist_mm[1], reach / np.maximum(np.sin(tilts), 1e-9))
    s = c.view_step
    sub = lambda A: np.ascontiguousarray(A[::s, ::s].reshape(-1, 3), np.float64)
    lo = np.maximum(
        _nearest(dirs, target, sub(P), sub(N), np.cos(np.radians(c.incidence_deg)), hi, first), c.dist_mm[0]
    )
    return target, dirs, tilts, azs, lo, hi


def _frame_region(rng, content, c: CameraCfg):
    """The page part (u0, v0, u1, v1) the frame is fitted to when it cuts the page and its cut sides: all four for the
    frame "inside", else a nonempty set drawn (each half the time); a cut side crop_margin (share of the content box's
    side, drawn per side) beyond the content, the others at the page edge; its shorter span grown (centred, within the
    page) to the longer: the sheet's aspect, which the flat has."""
    content = np.asarray(content, float)
    cut = np.ones(4, bool) if c.frame == "inside" else np.zeros(4, bool)
    while not cut.any():
        cut = rng.random(4) < 0.5
    size = np.tile(content[2:] - content[:2], 2)
    inner = np.clip(content + np.array([-1, -1, 1, 1]) * rng.uniform(*c.crop_margin, 4) * size, 0, 1)
    r = np.where(cut, inner, [0.0, 0.0, 1.0, 1.0])
    k = int(r[2] - r[0] > r[3] - r[1])  # the shorter axis: 0 u, 1 v
    length = r[3 - k] - r[1 - k]
    r[k] = np.clip(0.5 * (r[k] + r[k + 2] - length), 0.0, 1.0 - length)
    r[k + 2] = r[k] + length
    return r, cut


def make_camera(rng, P, N, size, c: CameraCfg, content=(0.0, 0.0, 1.0, 1.0)):
    """P, N: (nv, nu, 3) page grid points and unit normals; content: the content box (u0, v0, u1, v1) on the page.
    -> camera dict (eye, R, f, cx, cy) and view parameters.
    The frame fits the page bbox at the drawn fill (frame "page"), or exactly the page part _frame_region draws along
    the tighter axis; along the other one its free room goes away from the cut side (centred when both are cut).
    A view drawn among the possible ones (views, offset max_offset, else side_offset: a side view): a direction uniformly,
    then a distance log-uniformly in its range, moved out to where every node of the grid is within incidence_deg; the
    overhead view at the far end of dist_mm when none is possible (the geometry avoids it). Roll and fill are drawn; the focal length sets the fill. min_px_per_mm reports the smallest local
    image scale f cos(incidence) / depth on the page."""
    w, h = size
    roll = np.radians(rng.uniform(*c.roll_deg))
    fill = rng.uniform(*c.fill)
    target, dirs, tilts, azs, lo, hi = views(P, N, c, c.max_offset)
    ok = np.flatnonzero(lo <= hi)
    if not ok.size:
        target, dirs, tilts, azs, lo, hi = views(P, N, c, c.side_offset)
        ok = np.flatnonzero(lo <= hi)
    nv, nu = P.shape[:2]
    P, N = P.reshape(-1, 3).astype(np.float64), N.reshape(-1, 3).astype(np.float64)
    if ok.size:
        k = int(rng.choice(ok))
        dist = float(np.exp(rng.uniform(np.log(lo[k]), np.log(hi[k]))))
        full = _nearest(dirs[k : k + 1], target, P, N, np.cos(np.radians(c.incidence_deg)), np.array([np.inf]), False)[
            0
        ]
        dist = max(dist, full) if np.isfinite(full) else dist
    else:
        k, dist = 0, float(c.dist_mm[1])
    eye = target + dist * dirs[k]
    worst = _worst_cos(eye[None], P, N)[0]
    ray = eye - P
    cos_k = np.einsum("nk,nk->n", ray, N) / np.linalg.norm(ray, axis=1)
    cam = dict(eye=eye, R=look_at(eye, target, roll), f=1.0, cx=0.0, cy=0.0)
    xy, z = project(P, cam)
    cut = c.frame != "page"
    sides = np.zeros(4, bool)
    if cut:
        region, sides = _frame_region(rng, content, c)
        (i0, j0), (i1, j1) = (
            np.floor(region[:2] * [nu - 1, nv - 1]).astype(int),
            np.ceil(region[2:] * [nu - 1, nv - 1]).astype(int),
        )
        fill = 1.0
        sub = xy.reshape(nv, nu, 2)[j0 : j1 + 1, i0 : i1 + 1].reshape(-1, 2)
        lo, hi = sub.min(0), sub.max(0)
    else:
        lo, hi = xy.min(0), xy.max(0)
    f = fill * min(w / (hi[0] - lo[0]), h / (hi[1] - lo[1]))
    free = np.array([w, h]) - f * (hi - lo)
    shift = np.where(
        sides[:2] == sides[2:], 0.0 if cut else c.shift * rng.uniform(-1, 1, 2), np.where(sides[:2], -1.0, 1.0)
    )
    cx, cy = np.array([w, h]) / 2 - f * (lo + hi) / 2 + 0.5 * free * shift
    cam.update(f=float(f), cx=float(cx), cy=float(cy))
    view = dict(
        dist_mm=float(dist),
        tilt_deg=float(np.degrees(tilts[k])),
        azimuth_deg=float(np.degrees(azs[k]) % 360),
        roll_deg=float(np.degrees(roll)),
        fill=float(fill),
        fov_deg=float(np.degrees(2 * np.arctan(h / 2 / f))),
        max_incidence_deg=float(np.degrees(np.arccos(np.clip(worst, -1, 1)))),
        min_px_per_mm=float((f * np.clip(cos_k, 0, 1) / z).min()),
        cut=bool(cut),
        cut_sides="".join(n for n, k in zip("LTRB", sides) if k),
    )
    return cam, view
