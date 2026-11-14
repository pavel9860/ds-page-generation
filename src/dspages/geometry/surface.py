"""3D page grid from a scene.

The profile is solved along direction bend_dir (to the page's u axis, or its v axis with along_long) on the
page's rotated bounding box: p along the profile, g along its generators. Two-profile pages blend strips solved
at n_slices positions across g. Sag (shell under gravity) acts outside the support strip. Shallow creases and a
deep crease network are height maps in page coordinates. The surface is worked out on a page-aligned grid (the
mesh, work_scale times finer): each node takes the profile at its own p, and passes along the page rows and columns
give them their lengths back; the mesh is every work_scale-th node.
"""
from dataclasses import replace

import cv2
import numpy as np
from numba import njit

from ..config import GeometryCfg, paper_props
from ..creases import cockle, deep, shallow
from .sag import sag_field, solve_sag
from .scenes import lerp_scene, sample_ends, scene_label
from .strip import solve_strip
from ..render.camera import vertex_normals, views

ISO_PASSES = 2                  # row and column passes that give a creased sheet its lengths back (strain < 1e-4)


@njit(cache=True)
def _resample(Q, t):
    """Each curve Q[r] (m, 3) walked to the arc lengths t (from its first point; past its end straight on)."""
    out = np.empty_like(Q)
    n, m = Q.shape[0], Q.shape[1]
    for r in range(n):
        k, s0 = 0, 0.0
        seg = np.sqrt((Q[r, 1, 0] - Q[r, 0, 0]) ** 2 + (Q[r, 1, 1] - Q[r, 0, 1]) ** 2 + (Q[r, 1, 2] - Q[r, 0, 2]) ** 2)
        for j in range(m):
            while k < m - 2 and s0 + seg < t[j]:
                s0 += seg
                k += 1
                seg = np.sqrt((Q[r, k + 1, 0] - Q[r, k, 0]) ** 2 + (Q[r, k + 1, 1] - Q[r, k, 1]) ** 2
                              + (Q[r, k + 1, 2] - Q[r, k, 2]) ** 2)
            w = (t[j] - s0) / max(seg, 1e-12)
            for c in range(3):
                out[r, j, c] = Q[r, k, c] + w * (Q[r, k + 1, c] - Q[r, k, c])
    return out


def _lerp2(axes, values, pts):
    """RegularGridInterpolator's linear interpolation (extrapolating) on evenly spaced axes."""
    idx, frac = [], []
    for k, a in enumerate(axes):
        q = (pts[..., k] - a[0]) / (a[1] - a[0])
        i = np.clip(np.floor(q).astype(np.int64), 0, len(a) - 2)
        idx.append(i)
        frac.append(q - i)
    (i, j), (s, t) = idx, frac
    if values.ndim > 2:
        s, t = s[..., None], t[..., None]
    return ((1 - s) * ((1 - t) * values[i, j] + t * values[i, j + 1])
            + s * ((1 - t) * values[i + 1, j] + t * values[i + 1, j + 1]))


def _spaced(P, t, axis):
    """The curves of grid P (n, m, 3) along `axis` (1: rows, 0: columns) with their points moved along them to the arc
    lengths t from the first: each keeps its shape and its material length."""
    Q = np.ascontiguousarray(np.moveaxis(P, axis, 1))
    return np.moveaxis(_resample(Q, np.asarray(t, float) - t[0]), 1, axis)


def strains(X, Y, Z, u, v):
    su = np.sqrt(np.diff(X, axis=1) ** 2 + np.diff(Y, axis=1) ** 2 + np.diff(Z, axis=1) ** 2) / np.diff(u)[None, :] - 1
    sv = np.sqrt(np.diff(X, axis=0) ** 2 + np.diff(Y, axis=0) ** 2 + np.diff(Z, axis=0) ** 2) / np.diff(v)[:, None] - 1
    return su, sv


def _swap(a, b, along_long):
    """Sheet (a along the width, b along the height) <-> L-frame (l along the profile base axis, m across)."""
    return (b, a) if along_long else (a, b)


def _crease_heights(rng, g: GeometryCfg, cond, fold_ls, lm_to_p):
    """Height map [m] on the sheet (rows b, columns a) of the shallow creases and the deep network at the working
    grid's pitch, its axes, and the crease lines [mm]. Shallow creases keep fold_clear_mm from the fold lines (p = const)."""
    w_mm, h_mm = g.sheet_mm
    mesh_step = min(w_mm / (g.mesh[0] - 1), h_mm / (g.mesh[1] - 1))
    step = mesh_step / g.work_scale
    lines, h, pitch = [], None, None
    if cond["shallow"]:
        c = g.shallow
        folds = np.array(fold_ls)
        cand = shallow.sample(rng, w_mm, h_mm, c.mean_groups, c.p_isolated, c.mean_extra, c.radius_mm, c.spread_deg,
                              c.singles_per_group, c.k)
        lines = [cr for cr in cand if not len(folds) or np.abs(
            lm_to_p(*_swap(cr["line"][:, 0] * 1e-3, cr["line"][:, 1] * 1e-3, cond["along_long"]))[:, None]
            - folds[None, :]).min() > c.fold_clear_mm * 1e-3]
        h, pitch = shallow.render(lines, w_mm, h_mm, step, c.w_max_mm), step
    if cond["deep"]:
        net = deep.sample(rng, w_mm, h_mm, cond["deep"], g.deep, mesh_step)
        hd = deep.render(net, w_mm, h_mm, cond["deep"], g.deep, step)
        h, pitch = (hd if h is None else h + hd), step
        lines = lines + net
    if h is None:
        return None, None, lines
    return 1e-3 * h, (np.arange(h.shape[0]) * pitch * 1e-3, np.arange(h.shape[1]) * pitch * 1e-3), lines


def _open_book(solve, length, pp, n, halves, y):
    """An open book's profile: each half solved from the spine in the middle out (clamped there), the first one mirrored.
    y: the halves' states (None: rest). -> s, x, z, states."""
    (sl, xl, zl, yl), (sr, xr, zr, yr) = (solve(0.5 * length, pp["q"], pp["bending"], (), (), hf["clamp"], n, y0,
                                                hf["base"]) for hf, y0 in zip(halves, y or (None, None)))
    return (np.r_[0.5 * length - sl[::-1], 0.5 * length + sr[1:]], np.r_[xl[-1] - xl[::-1], xl[-1] + xr[1:]],
            np.r_[zl[::-1], zr[1:]], (yl, yr))


def page_points(work, u, v):
    """X, Y, Z [mm] of the page points at sheet coordinates u (across), v (down the rows) [mm]: one bilinear lookup in
    the page-aligned working grid W (3, len(Wv), len(Wu)) [mm] of make_page."""
    return np.moveaxis(_lerp2((work["Wv"], work["Wu"]), np.moveaxis(work["W"], 0, -1), np.stack([v, u], -1)), -1, 0)


def _water(rng, S, g, scale):
    """Cockling of a real water-damaged page (creases.cockle) on the working grid S [m]: field and present band
    energy on a grid of the band tables' pitch over the sheet, the field resized to the working grid."""
    shape = tuple(int(round(L / d)) + 1 for L, d in zip(g.sheet_mm[::-1], cockle.pitch()[::-1]))   # (rows, cols)
    Zt = cv2.resize(1e3 * S[..., 2], shape[::-1], interpolation=cv2.INTER_LINEAR)
    w = g.water
    h = cockle.field(shape, rng, cockle.sample(rng), dict(rms=cockle.bands(Zt).std((1, 2))), scale)
    h = cockle.local(h, rng, rng.uniform(*w.cover), w.region * shape[0] / 360, w.dry)
    h = cv2.resize(h, S.shape[1::-1], interpolation=cv2.INTER_AREA if h.shape[0] > S.shape[0] else cv2.INTER_LINEAR)
    n = np.cross(np.gradient(S, axis=1), np.gradient(S, axis=0))
    return S + 1e-3 * h[..., None] * n / np.linalg.norm(n, axis=-1, keepdims=True) * np.sign(n[..., 2:3] + 1e-12)


def make_surface(rng, g: GeometryCfg, paper, cond, camera=None):
    """make_page drawn again (up to view_tries) until a camera view of it is possible (camera: CameraCfg; None: once)."""
    for _ in range(g.view_tries if camera is not None else 1):
        state = rng.bit_generator.state
        s = make_page(rng, g, paper, cond)
        s["rng_state"] = state
        if camera is None or seen(s, camera):
            break
    return s


def seen(s, camera):
    """Whether a camera view of a page grid (make_page output) is possible within camera.side_offset."""
    k = camera.view_step
    X, Y, Z = s["X"][::k, ::k], s["Y"][::k, ::k], s["Z"][::k, ::k]
    _, _, _, _, lo, hi = views(np.stack([X, Y, Z], -1), vertex_normals(X, Y, Z), replace(camera, view_step=1), camera.side_offset, True)
    return bool((lo <= hi).any())


def replay_surface(state, g: GeometryCfg, paper, cond, strips, sag):
    """The make_surface page drawn from rng state `state` (its rng_state), its strips and sag given."""
    rng = np.random.default_rng()
    rng.bit_generator.state = state
    return make_page(rng, g, paper, cond, strips, sag)


def make_page(rng, g: GeometryCfg, paper, cond, strips=None, sag=None):
    """-> dict with the page grid X, Y, Z [mm] (rows down the sheet, columns across; face up, right-handed: Y points
    up the sheet) and what made it: scenes at both ends, paper, conditions, crease lines, max membrane strain of the
    laid-out box, the solved strips (s, x, z) in solve order and the sag at the shell nodes (None: no sag). strips, sag:
    those of a page of the same rng state, not solved again."""
    solved = [] if strips is None else list(strips)
    replay = strips is not None

    def solve(*a):
        if replay:
            return (*solved.pop(0), None)
        r = solve_strip(*a)
        solved.append(r[:3])
        return r

    pp = paper_props(paper, cond["gsm"])
    sw, sh = (s * 1e-3 for s in g.sheet_mm)
    length, width = _swap(sw, sh, cond["along_long"])
    beta = np.radians(cond["bend_dir"])
    cb, sn = np.cos(beta), np.sin(beta)
    lp, wg = length * cb + width * abs(sn), length * abs(sn) + width * cb
    p0, g0 = (0.0, length * sn) if beta >= 0 else (-width * sn, 0.0)

    def lm_to_p(l_, m_):
        return l_ * cb + m_ * sn + p0

    nu, nv = g.mesh
    pitch = min(sw / (nu - 1), sh / (nv - 1))
    up = np.linspace(0.0, lp, int(np.ceil(lp / pitch)) + 1)
    vg = np.linspace(0.0, wg, int(np.ceil(wg / pitch)) + 1)

    sa, sb = sample_ends(rng, g.scene, lp, wg, cond["two"], cond["combo"])
    t = np.linspace(0.0, 1.0, g.n_slices if cond["two"] else 1)
    raw, y = [], None
    for tk in t:
        sc = lerp_scene(sa, sb, tk)
        if sc["halves"]:
            s, x, z, y = _open_book(solve, lp, pp, g.strip_segments // 2, sc["halves"], y)
        else:
            s, x, z, y = solve(lp, pp["q"], pp["bending"], sc["folds"], sc["supports"], sc["clamp"],
                                     g.strip_segments, y, sc["base"])
        raw.append((s, x, z))
    ws = g.work_scale
    u = np.linspace(0.0, sw, (nu - 1) * ws + 1)
    v = np.linspace(0.0, sh, (nv - 1) * ws + 1)
    l_, m_ = _swap(*np.meshgrid(u, v), cond["along_long"])
    pn, qn = lm_to_p(l_, m_), -l_ * sn + m_ * cb + g0                    # page nodes on the profile and its generators

    def blend(q, f):
        """Slice values f(k) blended across the generators at q."""
        k = np.clip(np.searchsorted(t, q / wg, side="right") - 1, 0, max(len(t) - 2, 0))
        k1 = np.minimum(k + 1, len(t) - 1)
        wt = (q / wg - t[k]) / (t[min(1, len(t) - 1)] - t[0] or 1.0)
        return (1 - wt) * f(k) + wt * f(k1)

    xp = [np.interp(pn, s_, x_) for s_, x_, _ in raw]
    zp = [np.interp(pn, s_, z_) for s_, _, z_ in raw]
    x = blend(qn, lambda k: np.choose(k, xp))
    z = blend(qn, lambda k: np.choose(k, zp))
    span = (0.0, wg)
    if sa["supports"]:
        f = rng.uniform(*g.support_frac)
        a0 = rng.uniform(max(0.0, 0.5 - f), min(0.5, 1 - f)) * wg
        span = (a0, a0 + f * wg)
    if span[1] - span[0] < wg:
        if not replay:
            zs = np.array([np.interp(up, s_, z_) for s_, _, z_ in raw])
            zb = blend(vg[:, None], lambda k: zs[k.ravel()])
            sag = solve_sag(up, vg, zb, span, pp, g.shell_nodes)
        z = z + sag_field(up, vg, g.shell_nodes, sag, pn, qn)

    hmap, axes, lines = _crease_heights(rng, g, cond, [f[0] for f in sa["folds"]], lm_to_p)
    z = np.maximum(z, 0.0)
    X, Y = _swap((x - p0) * cb - (qn - g0) * sn, (x - p0) * sn + (qn - g0) * cb, cond["along_long"])
    S = np.stack([X, sh - Y, z], -1)
    if hmap is not None:
        h = _lerp2(axes, hmap, np.stack(np.meshgrid(v, u, indexing="ij"), -1))
        n = np.cross(np.gradient(S, axis=1), np.gradient(S, axis=0))
        S = S + h[..., None] * n / np.linalg.norm(n, axis=-1, keepdims=True) * np.sign(n[..., 2:3] + 1e-12)
    if cond.get("water", 0.0) > 0:
        S = _water(rng, S, g, cond["water"])
    for _ in range(ISO_PASSES):
        if cond["combo"][3]:
            S[..., 2] -= S[..., 2].min()
        S[..., 2] = np.maximum(S[..., 2], 0.0)
        S = _spaced(_spaced(S, u, 1), v, 0)
    S[..., 2] = np.maximum(S[..., 2], 0.0)
    W = 1e3 * np.moveaxis(S, -1, 0)
    work = dict(W=W, Wu=1e3 * u, Wv=1e3 * v)
    X, Y, Z = W[:, ::ws, ::ws]
    u, v = u[::ws], v[::ws]
    return dict(X=X, Y=Y, Z=Z, u=1e3 * u, v=1e3 * v, **work, scene_a=sa, scene_b=sb,
                label=scene_label(sa), paper=pp, cond=cond, creases=lines, strips=[] if replay else solved, sag=sag,
                box_strain=float(max(np.abs(a).max() for a in strains(*W, work["Wu"], work["Wv"]))))
