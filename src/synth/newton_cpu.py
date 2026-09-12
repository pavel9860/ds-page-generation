"""Numpy-only per-pixel Newton UV inversion (no torch, no mesh-locate).
Downscale-then-correct against a precomputed low-res height-field grid --
proven safe under multiprocessing (unlike gpu.py's torch path on CPU).
Shared by text_render.py and render.py's grid pipeline."""
import cv2
import numpy as np

from .geometry import project


def _fixed_z_seed(qx, qy, Z, cam):
    """Closed-form inverse assuming a fixed height Z -- 2x2 linear solve."""
    R, f, cx, cy, dist = cam['R'], cam['f'], cam['cx'], cam['cy'], cam['dist']
    cx0, cy0 = cam['centre'][0], cam['centre'][1]
    a, b, c, e = R[0, 0], R[0, 1], R[1, 0], R[1, 1]
    rp, rq = R[2, 0], R[2, 1]
    r02, r12, r22 = R[0, 2], R[1, 2], R[2, 2]
    X = (qx - cx) / f
    Y = (qy - cy) / f
    A11, A12 = a - X * rp, b - X * rq
    A21, A22 = c - Y * rp, e - Y * rq
    rhs1 = X * dist - Z * (r02 - X * r22)
    rhs2 = Y * dist - Z * (r12 - Y * r22)
    det = A11 * A22 - A12 * A21
    dU = (rhs1 * A22 - A12 * rhs2) / det
    dV = (A11 * rhs2 - rhs1 * A21) / det
    return dU + cx0, dV + cy0


def _height_grad(zf, U, V, eps=1e-3):
    Z = zf(U, V)
    dU = (zf(U + eps, V) - zf(U - eps, V)) / (2 * eps)
    dV = (zf(U, V + eps) - zf(U, V - eps)) / (2 * eps)
    return Z, dU, dV


def _project_grad(U, V, Z, dZ_dU, dZ_dV, cam):
    R, centre, f, cx, cy, dist = cam['R'], cam['centre'], cam['f'], cam['cx'], cam['cy'], cam['dist']
    P = np.stack([U - centre[0], V - centre[1], Z], -1) @ R.T
    d = np.maximum(P[..., 2] + dist, 1e-3)
    x = f * P[..., 0] / d + cx
    y = f * P[..., 1] / d + cy
    r00, r01, r02 = R[0]
    r10, r11, r12 = R[1]
    r20, r21, r22 = R[2]
    dPx_dU, dPx_dV = r00 + r02 * dZ_dU, r01 + r02 * dZ_dV
    dPy_dU, dPy_dV = r10 + r12 * dZ_dU, r11 + r12 * dZ_dV
    dd_dU, dd_dV = r20 + r22 * dZ_dU, r21 + r22 * dZ_dV
    Jxu = f * (dPx_dU * d - P[..., 0] * dd_dU) / (d * d)
    Jxv = f * (dPx_dV * d - P[..., 0] * dd_dV) / (d * d)
    Jyu = f * (dPy_dU * d - P[..., 1] * dd_dU) / (d * d)
    Jyv = f * (dPy_dV * d - P[..., 1] * dd_dV) / (d * d)
    return x, y, d, Jxu, Jxv, Jyu, Jyv


def _height_field_sampler(zf, mm_w, n_grid):
    """Z, dZ/dU, dZ/dV precomputed once on a regular n_grid x n_grid
    lattice; returns a closure bilinearly interpolating them anywhere."""
    lin = np.linspace(0, mm_w, n_grid)
    Ug, Vg = np.meshgrid(lin, lin)
    Zg, dUg, dVg = _height_grad(zf, Ug, Vg)
    stacked = np.stack([Zg, dUg, dVg], -1).astype(np.float32)

    def sample(U, V):
        gx = (U / mm_w * (n_grid - 1)).astype(np.float32)
        gy = (V / mm_w * (n_grid - 1)).astype(np.float32)
        out = cv2.remap(stacked, gx, gy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return out[..., 0].astype(np.float64), out[..., 1].astype(np.float64), out[..., 2].astype(np.float64)
    return sample


def newton_uv_invert(zf, cam, mm_w, out_size, n_iters=2, downscale=3):
    """Newton solve of project(U,V,z(U,V),cam)=(qx,qy) at 1/downscale
    resolution against a precomputed LOW-RES height-field grid (the
    precompute itself is the expensive part on CPU), upsampled, then one
    full-res correction against the same low-res grid. Final depth/page
    reuse the same interpolated height sampler (no exact zf() call)."""
    N = out_size
    N_low = max(N // downscale, 8)
    sample_height = _height_field_sampler(zf, mm_w, N_low)

    lin_low = np.linspace(0, N - 1, N_low)
    qy_low, qx_low = np.meshgrid(lin_low, lin_low, indexing='ij')

    U, V = _fixed_z_seed(qx_low, qy_low, np.zeros_like(qx_low), cam)
    Z0, _, _ = sample_height(U, V)
    U, V = _fixed_z_seed(qx_low, qy_low, Z0, cam)
    for _ in range(n_iters):
        Z, dU, dV = sample_height(U, V)
        gx, gy, _, Jxu, Jxv, Jyu, Jyv = _project_grad(U, V, Z, dU, dV, cam)
        det = Jxu * Jyv - Jxv * Jyu
        rx, ry = qx_low - gx, qy_low - gy
        U = U + (rx * Jyv - Jxv * ry) / det
        V = V + (Jxu * ry - rx * Jyu) / det

    U = cv2.resize(U.astype(np.float32), (N, N), interpolation=cv2.INTER_LINEAR).astype(np.float64)
    V = cv2.resize(V.astype(np.float32), (N, N), interpolation=cv2.INTER_LINEAR).astype(np.float64)

    qy, qx = np.mgrid[0:N, 0:N].astype(np.float64)
    Z, dU, dV = sample_height(U, V)
    gx, gy, _, Jxu, Jxv, Jyu, Jyv = _project_grad(U, V, Z, dU, dV, cam)
    det = Jxu * Jyv - Jxv * Jyu
    rx, ry = qx - gx, qy - gy
    U = U + (rx * Jyv - Jxv * ry) / det
    V = V + (Jxu * ry - rx * Jyu) / det

    Z, _, _ = sample_height(U, V)
    gx, gy, depth = project(U, V, Z, cam)
    resid = np.hypot(qx - gx, qy - gy)
    page = (U >= 0) & (U <= mm_w) & (V >= 0) & (V <= mm_w) & (resid < 0.5)
    return U, V, depth.astype(np.float32), page


def apply_uv_mapping_newton(tex, zf, cam, mm_w, out_size, ppm, n_iters=2, downscale=3):
    U, V, depth, page = newton_uv_invert(zf, cam, mm_w, out_size, n_iters, downscale)
    mapx = (U * ppm).astype(np.float32)
    mapy = (V * ppm).astype(np.float32)
    img = cv2.remap(tex, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    return img, depth, page
