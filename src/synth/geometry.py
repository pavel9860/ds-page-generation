"""Paper surface + camera.

Real rig caps combined incidence at CAP_DEG. Bend/fold/crease angles are
each drawn independently from their own config range; render.py verifies
the ACTUAL combined incidence (camera tilt + local surface slope together,
via incidence_deg on the real composed zf) and resamples the camera's
azimuth if violated, rather than pre-shrinking any individual feature's
range. No rendering-time incidence filter on individual points -- a
steeply foreshortened but unoccluded point is still visible in a real
photo, so masking it out would remove real, valid points. The only
legitimate exclusion is genuine self-occlusion (render.py's bridging check).
"""
import cv2
import numpy as np

from creases.deep import make_ridge_network, crease_field

from . import config as cfg

FINE_MM = cfg.FINE_MM
MAX_CAMERA_TILT_DEG = cfg.MAX_CAMERA_TILT_DEG


def make_fold(rng, span, ppmm):
    """A valley fold: flat, smooth-bottomed V, flat again -- bounded height
    unlike a bend. Spans the whole patch, no along-line taper. Bottom
    rounding radius is a fraction of the fold's own half-width, not an
    independent draw -- real fold curvature tracks bending stiffness, so a
    narrower fold is physically sharper."""
    angle = np.deg2rad(rng.uniform(*cfg.FOLD_ANGLE_DEG))
    th = rng.uniform(0, np.pi)
    d = rng.uniform(-0.4, 0.4) * span
    half_width_mm = rng.uniform(*cfg.FOLD_HALF_WIDTH_PX) / ppmm
    r_mm = half_width_mm * rng.uniform(*cfg.FOLD_R_FRAC_OF_WIDTH)
    return dict(th=th, d=d, r=r_mm, w=half_width_mm,
               slope=np.tan(angle) * rng.choice([-1, 1]))


def _fold_field(f, U, V):
    t = U * np.cos(f['th']) + V * np.sin(f['th']) - f['d']
    v_shape = f['slope'] * (np.sqrt(t ** 2 + f['r'] ** 2) - f['r'])
    envelope = np.exp(-(t / (2 * f['w'])) ** 2)   # decays back to flat
    return v_shape * envelope


def make_surface(rng, span, ppmm, severity=1.0, add_creases=None):
    """z(U,V) from bends + fold(s) + crease mark(s) + low undulation. Each
    feature's own angle is bounded by its own config range (BEND_ANGLE_DEG/
    FOLD_ANGLE_DEG/CREASE_ANGLE_DEG); the combined camera-relative incidence
    is what render.py actually caps, via CAP_DEG + azimuth retry. `add_creases`:
    force the crease-network decision (used by render.py's grid pipeline to
    pair it 1:1 with a texture-space crease effect at the same trigger);
    None (default, used by the text pipeline and standalone tools) draws it
    internally at CREASE_CLUSTER_PROB, same as before."""
    bends, folds, creases = [], [], []
    if rng.random() < cfg.BEND_PROB:
        for _ in range(rng.integers(*cfg.BEND_COUNT)):
            th = rng.uniform(0, np.pi)
            angle = np.deg2rad(rng.uniform(*cfg.BEND_ANGLE_DEG))
            s = rng.uniform(*cfg.BEND_S_FRAC) * span
            A = np.tan(angle) * s * severity * rng.choice([-1, 1])
            bends.append(dict(th=th, d=rng.uniform(-0.5, 0.5) * span, s=s, A=A))
    if rng.random() < cfg.FOLD_PROB:
        for _ in range(rng.integers(*cfg.FOLD_COUNT)):
            folds.append(make_fold(rng, span, ppmm))
    if add_creases is None:
        add_creases = rng.random() < cfg.CREASE_CLUSTER_PROB
    if add_creases:
        creases = make_ridge_network(rng, span, ppmm, severity)
    fx, fy = rng.uniform(-1, 1, 2) / span
    undul = dict(fx=fx, fy=fy, ph=rng.uniform(0, 6.28),
                A=rng.uniform(*cfg.UNDULATION_AMP_FRAC) * span)

    def z(U, V):
        out = np.zeros_like(U, dtype=np.float64)
        for b in bends:
            t = U * np.cos(b['th']) + V * np.sin(b['th']) - b['d']
            out += b['A'] * np.tanh(t / b['s'])
        for c in creases:
            out += crease_field(c, U, V)
        for f in folds:
            out += _fold_field(f, U, V)
        out += undul['A'] * np.sin(2 * np.pi * (undul['fx'] * U + undul['fy'] * V) + undul['ph'])
        return out
    return z


def _resample_rows_by_arclength(P, n_out):
    """Resample each row of P (rows, hi, 3) to n_out points equally spaced
    in cumulative 3D arc length along the row, via cv2.remap."""
    d = np.linalg.norm(np.diff(P, axis=1), axis=-1)
    s = np.concatenate([np.zeros((P.shape[0], 1)), np.cumsum(d, axis=1)], axis=1)
    s_out = np.linspace(0, s[:, -1], n_out, axis=1)
    idx = np.stack([np.interp(s_out[i], s[i], np.arange(s.shape[1])) for i in range(s.shape[0])])
    rows_y = np.repeat(np.arange(P.shape[0])[:, None], n_out, axis=1).astype(np.float32)
    return cv2.remap(P.astype(np.float32), idx.astype(np.float32), rows_y, cv2.INTER_LINEAR)


def isometric_mesh(zf, mm_w, mm_h, n, upsample=4):
    """XYZ mesh with true 3D arc-length spacing, replacing the regular-XY
    Monge patch (X=U, Y=V, Z=zf(U,V)), which stretches wherever it bends.
    Samples zf on a fine regular grid, then resamples rows then columns to
    n points each, equally spaced in cumulative arc length."""
    u = np.linspace(0, mm_w, n * upsample)
    v = np.linspace(0, mm_h, n * upsample)
    U, V = np.meshgrid(u, v)
    P = np.stack([U, V, zf(U, V)], axis=-1)
    rows = _resample_rows_by_arclength(P, n)                      # (v, u_resampled, 3)
    cols = _resample_rows_by_arclength(rows.transpose(1, 0, 2), n)  # (u_resampled, v_resampled, 3)
    out = cols.transpose(1, 0, 2)                                  # (v_resampled, u_resampled, 3)
    return out[..., 0], out[..., 1], out[..., 2]


def surface_normal(zf, U, V, eps=0.3):
    zx = (zf(U + eps, V) - zf(U - eps, V)) / (2 * eps)
    zy = (zf(U, V + eps) - zf(U, V - eps)) / (2 * eps)
    n = np.stack([-zx, -zy, np.ones_like(zx)], -1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def rotation_matrix(az, roll, tilt_deg):
    """Camera R: azimuth-axis tilt (Rodrigues) then in-plane roll."""
    t = np.deg2rad(tilt_deg)
    ax = np.array([np.cos(az), np.sin(az), 0.0])
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    R_tilt = np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)
    cz, sz = np.cos(roll), np.sin(roll)
    R_roll = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return R_roll @ R_tilt


def make_camera(rng, mm_w, mm_h, out_size, ppmm, tilt_deg=None):
    """Pinhole camera; `ppmm` fixes the frontal image scale. Tilt is capped
    at MAX_CAMERA_TILT_DEG so it stacks safely with any feature's own slope."""
    if tilt_deg is None:
        tilt_deg = rng.uniform(0, MAX_CAMERA_TILT_DEG)
    az = rng.uniform(0, 2 * np.pi)
    roll = np.deg2rad(rng.uniform(*cfg.CAMERA_ROLL_DEG))   # image-axis aligned, not 45deg
    R = rotation_matrix(az, roll, tilt_deg)

    dist = rng.uniform(*cfg.CAMERA_DIST_MM)
    f = ppmm * dist
    centre = np.array([mm_w / 2, mm_h / 2, 0.0])
    cxj = rng.uniform(*cfg.CAMERA_CENTER_JITTER_PX)
    cyj = rng.uniform(*cfg.CAMERA_CENTER_JITTER_PX)
    if rng.random() < cfg.CAMERA_BIG_SHIFT_PROB:
        mag = rng.uniform(*cfg.CAMERA_BIG_SHIFT_PX)
        ang = rng.uniform(0, 2 * np.pi)
        cxj += mag * np.cos(ang)
        cyj += mag * np.sin(ang)
    return dict(R=R, f=f, dist=dist, centre=centre,
                cx=out_size / 2 + cxj, cy=out_size / 2 + cyj)


def project(U, V, Z, cam):
    P = np.stack([U - cam['centre'][0], V - cam['centre'][1], Z], -1) @ cam['R'].T
    P[..., 2] += cam['dist']
    d = np.maximum(P[..., 2], 1e-3)
    x = cam['f'] * P[..., 0] / d + cam['cx']
    y = cam['f'] * P[..., 1] / d + cam['cy']
    return x, y, d


def incidence_deg(zf, cam, U, V):
    """Angle between optical axis and local surface normal. Diagnostics only."""
    n = surface_normal(zf, U, V)
    n_cam = n @ cam['R'].T
    cos_i = np.clip(n_cam[..., 2], -1, 1)
    return np.degrees(np.arccos(cos_i))


def retry_camera(build_cam, zf, U, V, tries, cap_deg):
    """Call build_cam() up to `tries` times, keeping the first whose
    incidence to every point on the surface stays within cap_deg. Falls
    back to the last attempt if none qualify -- the page can legitimately
    extend outside the frame, this only bounds obliqueness. build_cam is
    expected to hold tilt fixed and vary only azimuth (and whatever else)
    across calls."""
    cam = None
    for _ in range(tries):
        cam = build_cam()
        if incidence_deg(zf, cam, U, V).max() <= cap_deg:
            return cam
    return cam
