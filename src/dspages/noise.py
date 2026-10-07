"""Procedural noise evaluated per point (no grids, sharp at any scale): hashed value noise, fractal noise,
cellular (Worley) noise and domain warping, compiled with numba."""
import cv2
import numpy as np
from numba import njit, vectorize

_SCALE = [1.0]
C_COARSE = 4.0                                      # feature size, in pixels, from which noise is evaluated on a coarser grid


def set_scale(k):
    """Factor of every feature size below: one value per page, drawn by the caller, independent of the page's geometry."""
    _SCALE[0] = float(k)


def scaled(size):
    return float(size) * _SCALE[0]


@njit(cache=True, inline="always")
def _hash(seed, i, j):
    h = (i * 374761393 + j * 668265263 + seed * 2246822519) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return np.float32((h ^ (h >> 16)) & 0xFFFF) / np.float32(65535.0)


@vectorize(["float32(int64, float64, float64)"], cache=True)
def _lattice(seed, i, j):
    return _hash(seed, np.int64(i), np.int64(j))


@njit(cache=True, inline="always")
def _value(seed, x, y):
    fi, fj = np.floor(x), np.floor(y)
    i, j = np.int64(fi), np.int64(fj)
    fx, fy = x - fi, y - fj
    fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    a, b = _hash(seed, i, j), _hash(seed, i + 1, j)
    c, d = _hash(seed, i, j + 1), _hash(seed, i + 1, j + 1)
    return (a + (b - a) * fx) * (1 - fy) + (c + (d - c) * fx) * fy


@vectorize(["float32(int64, float64, float64)"], cache=True)
def value_noise(seed, x, y):
    """Smooth value noise in [0, 1] at arbitrary coordinates (lattice pitch 1)."""
    return _value(seed, x, y)


@njit(cache=True, fastmath=True)
def _fbm(seed, x, y, scale, octaves, gain):
    out = np.empty(x.size, np.float32)
    for k in range(x.size):
        acc, amp, f, tot = 0.0, 1.0, 1.0 / scale, 0.0
        for o in range(octaves):
            acc += amp * (_value(seed + o, x[k] * f, y[k] * f) * 2 - 1)
            tot += amp
            amp, f = amp * gain, f * 2.03
        out[k] = acc / tot
    return out


@njit(cache=True, fastmath=True)
def _fbm_rotated(seed, h, w, step, c, s, k, scale, octaves, gain):
    out = np.empty((h, w), np.float32)
    for j in range(h):
        for i in range(w):
            x, y = i * step, j * step
            a, b = x * c + y * s, (y * c - x * s) * k
            acc, amp, f, tot = 0.0, 1.0, 1.0 / scale, 0.0
            for o in range(octaves):
                acc += amp * (_value(seed + o, a * f, b * f) * 2 - 1)
                tot += amp
                amp, f = amp * gain, f * 2.03
            out[j, i] = acc / tot
    return out


def fbm_rotated(seed, h, w, step, theta, k, scale, octaves=5, gain=0.5):
    """Fractal noise of an (h, w) pixel grid of pitch `step`, the coordinates turned by theta and the second stretched by k:
    fbm of the rotated coordinates without the coordinate arrays."""
    sc = scaled(scale)
    q = 2 if sc >= C_COARSE * step else 1                         # features of a few pixels are evaluated on every 2nd pixel
    field = _fbm_rotated(int(seed), -(-h // q), -(-w // q), float(step) * q, np.cos(theta), np.sin(theta), float(k), sc, int(octaves), float(gain))
    return field if q == 1 else cv2.resize(field, (w, h), interpolation=cv2.INTER_LINEAR)


def fbm(seed, x, y, scale, octaves=5, gain=0.5):
    """Fractal noise in about [-1, 1], feature size `scale` (same units as x, y)."""
    x, y = np.broadcast_arrays(np.asarray(x, np.float64), np.asarray(y, np.float64))
    return _fbm(int(seed), x.ravel(), y.ravel(), scaled(scale), int(octaves), float(gain)).reshape(x.shape)


@njit(cache=True, fastmath=True)
def _worley(seed, x, y, scale):
    n = x.size
    d1, d2, cid = np.empty(n, np.float32), np.empty(n, np.float32), np.empty(n, np.float32)
    for k in range(n):
        px, py = x[k] / scale, y[k] / scale
        i0, j0 = np.int64(np.floor(px)), np.int64(np.floor(py))
        a, b, c = 9.0, 9.0, 0.0
        for di in range(-1, 2):
            for dj in range(-1, 2):
                i, j = i0 + di, j0 + dj
                d = np.hypot(i + _hash(seed, i, j) - px, j + _hash(seed + 1, i, j) - py)
                if d < a:
                    a, b, c = d, a, _hash(seed + 2, i, j)
                elif d < b:
                    b = d
        d1[k], d2[k], cid[k] = a, b, c
    return d1, d2, cid


@njit(cache=True)
def _flecks(seed, h, w, scale, radius, cid_max):
    mask = np.zeros((h, w), np.uint8)
    r = radius * scale
    for j in range(-1, int(h / scale) + 2):
        for i in range(-1, int(w / scale) + 2):
            if _hash(seed + 2, i, j) < cid_max:
                cx, cy = (i + _hash(seed, i, j)) * scale, (j + _hash(seed + 1, i, j)) * scale
                for y in range(max(int(cy - r), 0), min(int(cy + r) + 1, h)):
                    for x in range(max(int(cx - r), 0), min(int(cx + r) + 1, w)):
                        if (x - cx) ** 2 + (y - cy) ** 2 < r * r:
                            mask[y, x] = 1
    return mask


def fleck_mask(seed, h, w, scale, radius, cid_max):
    """Cellular flecks: a disc of `radius` (cell sizes) at the jittered point of every cell of `scale` px whose id is below
    cid_max, painted cell by cell (what worley's d1 < radius and id < cid_max gives, without a distance per pixel)."""
    return _flecks(int(seed), h, w, scaled(scale), float(radius), float(cid_max)).astype(bool)


def worley(seed, x, y, scale):
    """Cellular noise: distances to the nearest and second nearest jittered lattice point (feature size `scale`)
    and the nearest point's cell id in [0, 1]."""
    x, y = np.broadcast_arrays(np.asarray(x, np.float64), np.asarray(y, np.float64))
    return tuple(r.reshape(x.shape) for r in _worley(int(seed), x.ravel(), y.ravel(), scaled(scale)))


def warp(seed, x, y, scale, amount):
    """Domain warping: coordinates displaced by fractal noise."""
    return x + amount * fbm(seed, x, y, scale, 4), y + amount * fbm(seed + 50, x, y, scale, 4)
