"""Pinhole camera over the page: pose from the view pool, focal length and principal point so the page fills
the frame. Table plane z = 0, page above it, all lengths in mm; camera axes x right, y down, z forward."""
import numpy as np
from numba import njit

from ..config import CameraCfg

DIRECTIONS = 64
DIST_STEPS = 12
DIST_GROWTH = 1.3


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


def make_camera(rng, P, N, size, c: CameraCfg):
    """P, N: (n, 3) surface points and unit normals. -> camera dict (eye, R, f, cx, cy) and view parameters.
    Distance, tilt, azimuth, roll and fill are drawn. The pose is the one nearest to the drawn direction, among
    DIRECTIONS view directions within tilt_deg with the camera's horizontal offset from the page centre <=
    max_offset x the page size, at the smallest of DIST_STEPS distances from the drawn one up
    (x DIST_GROWTH each) where every point is seen within incidence_deg of its normal (the least oblique pose if
    none is). The focal length sets the fill. min_px_per_mm reports the smallest local image scale
    f cos(incidence) / depth on the page."""
    w, h = size
    target = 0.5 * (P.min(0) + P.max(0))
    dist0 = float(np.exp(rng.uniform(*np.log(c.dist_mm))))
    az0 = rng.uniform(0, 2 * np.pi)
    tilt0 = np.radians(rng.uniform(*c.tilt_deg))
    roll = np.radians(rng.uniform(*c.roll_deg))
    fill = rng.uniform(*c.fill)
    tmax = np.radians(c.tilt_deg[1])
    k = np.arange(DIRECTIONS)
    azs = np.concatenate([[az0], k * np.pi * (3 - np.sqrt(5))])
    d0 = np.array([np.sin(tilt0) * np.cos(az0), np.sin(tilt0) * np.sin(az0), np.cos(tilt0)])
    reach = c.max_offset * float(np.ptp(P[:, :2], 0).max())
    cos_max = np.cos(np.radians(c.incidence_deg))
    best = None
    for dist in dist0 * DIST_GROWTH ** np.arange(DIST_STEPS):
        t_d = min(tmax, np.arcsin(min(1.0, reach / dist)))      # directions spread over the tilts the offset allows
        tilts = np.concatenate([[tilt0], t_d * np.sqrt((k + 0.5) / DIRECTIONS)])
        dirs = np.stack([np.sin(tilts) * np.cos(azs), np.sin(tilts) * np.sin(azs), np.cos(tilts)], -1)
        eyes = target + dist * dirs
        worst = np.where(dist * np.sin(tilts) <= reach * (1 + 1e-9), _worst_cos(eyes, P, N), -1.0)
        ok = np.flatnonzero(worst >= cos_max)
        if ok.size:
            best = (dist, int(ok[np.argmax(dirs[ok] @ d0)]), eyes, worst, tilts)
            break
        j = int(worst.argmax())
        if best is None or worst[j] > best[3][best[1]]:
            best = (dist, j, eyes, worst, tilts)
    dist, k, eyes, worst, tilts = best
    ray = eyes[k] - P
    cos_k = np.einsum("nk,nk->n", ray, N) / np.linalg.norm(ray, axis=1)
    cam = dict(eye=eyes[k], R=look_at(eyes[k], target, roll), f=1.0, cx=0.0, cy=0.0)
    xy, z = project(P, cam)
    lo, hi = xy.min(0), xy.max(0)
    f = fill * min(w / (hi[0] - lo[0]), h / (hi[1] - lo[1]))
    free = np.array([w, h]) - f * (hi - lo)
    cx, cy = np.array([w, h]) / 2 - f * (lo + hi) / 2 + 0.5 * c.shift * free * rng.uniform(-1, 1, 2)
    cam.update(f=float(f), cx=float(cx), cy=float(cy))
    view = dict(dist_mm=float(dist), tilt_deg=float(np.degrees(tilts[k])), azimuth_deg=float(np.degrees(azs[k])),
                roll_deg=float(np.degrees(roll)), fill=float(fill), fov_deg=float(np.degrees(2 * np.arctan(h / 2 / f))),
                max_incidence_deg=float(np.degrees(np.arccos(np.clip(worst[k], -1, 1)))),
                min_px_per_mm=float((f * np.clip(cos_k, 0, 1) / z).min()))
    return cam, view
