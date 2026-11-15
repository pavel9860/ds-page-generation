"""Part 3: a photo of a layout page on a 3D page surface, with its flat image, UV map and 3D map.

The frame (the plan's) holds the whole page, cuts it on some sides or all four, the content in view. The flat is the page
part in view: on a cut side it ends at the smallest page margin beyond the content left in view there, then brought to
the sheet's aspect by trimming margins (never the content). The layout is drawn once, at the scale that makes that part
page_px (the flat is a slice of it); the photo samples the same drawing. All at the sheet's aspect, turned portrait
when it is landscape (an open book): photo and flat page_px, the UV and 3D maps the page grid's size (mesh): node
(i, j) is the point at evenly spaced coordinates of the flat, so neither map has holes.
flat   (h, w) uint8: the layout sheet's part in view (luminance)
warped (h, w) uint8: the photo (luminance)
grid2d (2, mesh h, mesh w) float32: x, y of each page node in the photo, in [-1, 1] with align_corners=True (pixel
       centres at -1 and 1): the grid_sample grid that flattens the photo
grid3d (3, mesh h, mesh w) float32: X, Y, Z [mm] of each page node on the curved surface, table z = 0
mask   (h, w) bool: photo pixels on the flat's part of the page
"""

import cv2
import numpy as np

from ..config import Preset
from ..geometry.surface import page_points
from ..layout.generate import draw_layout
from . import photo
from .camera import make_camera, project, vertex_normals
from .config import BG_STEP, FLAT_ROUNDS, MIP_LEVELS, MIP_STEP, SHADOW_STEP  # noqa: F401
from .raster import grid_faces, rasterize


def _nodes(n, step):
    return np.unique(np.r_[0:n:step, n - 1])


def _page_in_photo(lin, mx, my, mask):
    """The page at photo pixels (mx, my in page px): prefiltered where a photo pixel covers more than one page pixel
    (sigma 0.5 sqrt(s^2 - 1) for s page px per photo px along the most shrunk direction), blended between MIP_LEVELS."""
    q = MIP_STEP
    sx, sy = mx[::q, ::q] / q, my[::q, ::q] / q
    a, b = np.gradient(sx, axis=1), np.gradient(sx, axis=0)
    c, d = np.gradient(sy, axis=1), np.gradient(sy, axis=0)
    s2 = 0.5 * (a * a + b * b + c * c + d * d + np.sqrt(((a - d) ** 2 + (b + c) ** 2) * ((a + d) ** 2 + (b - c) ** 2)))
    inner = cv2.erode(mask[::q, ::q].astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    sigma = np.where(inner, 0.5 * np.sqrt(np.maximum(s2 - 1, 0)), 0.0)
    if sigma.max() <= 0:
        return cv2.remap(lin, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    pos = cv2.resize(
        np.interp(sigma, MIP_LEVELS, np.arange(len(MIP_LEVELS))).astype(np.float32),
        mx.shape[::-1],
        interpolation=cv2.INTER_LINEAR,
    )
    out = np.zeros(mx.shape, np.float32)
    for k, s in enumerate(MIP_LEVELS):
        if pos.min() < k + 1 and pos.max() > k - 1:
            wk = np.clip(1 - np.abs(pos - k), 0, 1)
            src = lin if s == 0 else cv2.GaussianBlur(lin, (0, 0), s)
            out += wk * cv2.remap(src, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out


def _in_view(inside, span, content):
    """-> the page part (u0, v0, u1, v1) the flat may keep given the in-frame nodes inside (nv, nu): per side the page
    edge when the frame shows it along the rows and columns of span, else the nearest-to-content extent of the in-frame
    nodes over them (never inside the content); and their farthest extent (where the flat may grow)."""
    nv, nu = inside.shape
    n = np.array([nu - 1, nv - 1, nu - 1, nv - 1])
    i0, j0 = np.floor(span[:2] * n[:2]).astype(int)
    i1, j1 = np.ceil(span[2:] * n[2:]).astype(int)
    c0, c1 = np.floor(content[:2] * n[:2]).astype(int), np.ceil(content[2:] * n[2:]).astype(int)
    rows, cols = inside[j0 : j1 + 1], inside[:, i0 : i1 + 1]
    idx_u, idx_v = np.arange(nu), np.arange(nv)[:, None]
    first_u, last_u = np.where(rows, idx_u, nu).min(1), np.where(rows, idx_u, -1).max(1)
    first_v, last_v = np.where(cols, idx_v, nv).min(0), np.where(cols, idx_v, -1).max(0)
    tight = np.r_[np.minimum([first_u.max(), first_v.max()], c0), np.maximum([last_u.min(), last_v.min()], c1)]
    loose = np.r_[np.minimum([first_u.min(), first_v.min()], c0), np.maximum([last_u.max(), last_v.max()], c1)]
    return np.clip(tight / n, 0, 1), np.clip(loose / n, 0, 1)


def _flat_part(pix, nv, nu, px, content):
    """The page part the flat keeps: _in_view over the rows and columns of the part itself (from the content's on, until
    it stays the same), brought to the sheet's aspect (_fit_aspect)."""
    inside = ((pix >= 0) & (pix <= np.array(px))).all(1).reshape(nv, nu)
    part = content
    for _ in range(FLAT_ROUNDS):
        new = _fit_aspect(*_in_view(inside, part, content), content)
        if np.allclose(new, part):
            break
        part = new
    return new


def _resize_span(a0, a1, c0, c1, lo, hi, length):
    """Span [a0, a1] brought to `length`: shrunk on its cut sides first (a0 > 0, a1 < 1), then on the others, never
    inside the content [c0, c1]; grown up to [lo, hi] (the page in view), then to the sheet [0, 1]."""
    d = (a1 - a0) - length
    if d > 0:
        for cut in (True, False):
            room = np.array([(a0 > 0) == cut and max(c0 - a0, 0), (a1 < 1) == cut and max(a1 - c1, 0)], float)
            take = room * min(1.0, d / room.sum()) if room.sum() > 0 else room
            a0, a1, d = a0 + take[0], a1 - take[1], d - take.sum()
    for b0, b1 in ((lo, hi), (0.0, 1.0)):
        if d < 0:
            room = np.array([max(a0 - b0, 0), max(b1 - a1, 0)])
            give = room * min(1.0, -d / room.sum()) if room.sum() > 0 else room
            a0, a1, d = a0 - give[0], a1 + give[1], d + give.sum()
    return a0, a1


def _fit_aspect(tight, loose, content):
    """The page part tight (u0, v0, u1, v1) brought to the sheet's aspect (equal shares of both sides): the longer
    span shrunk to the shorter (_resize_span), the shorter grown to what the content leaves of the longer."""
    a = tight.copy()
    k = int(a[2] - a[0] < a[3] - a[1])  # the longer axis: 0 u, 1 v
    o = 1 - k
    a[k], a[k + 2] = _resize_span(a[k], a[k + 2], content[k], content[k + 2], loose[k], loose[k + 2], a[o + 2] - a[o])
    a[o], a[o + 2] = _resize_span(a[o], a[o + 2], content[o], content[o + 2], loose[o], loose[o + 2], a[k + 2] - a[k])
    return np.clip(a, 0.0, 1.0)


def make_sample(rng, layout, surf, P: Preset):
    """layout: layout.generate.make_layout's desc, drawn here once at the flat's scale; surf: geometry.make_surface
    output. -> dict of arrays and meta."""
    c = P.render
    ns = rng.uniform(*P.layout.noise_scale)  # factor of this photo's noise feature sizes
    px = tuple(P.layout.page_px)  # photo and flat (w, h)
    content = np.array(layout["ink"])
    X, Y, Z = surf["X"], surf["Y"], surf["Z"]
    nv, nu = X.shape
    Pts = np.stack([X, Y, Z], -1).reshape(-1, 3)
    N = vertex_normals(X, Y, Z).reshape(-1, 3)
    faces = grid_faces(nv, nu)
    cam, view = make_camera(rng, Pts.reshape(nv, nu, 3), N.reshape(nv, nu, 3), px, c.camera, content)
    pix, depth = project(Pts, cam)
    uu, vv = np.meshgrid(np.linspace(0, 1, nu), np.linspace(0, 1, nv))
    shading = c.effects["shading"].prob > 0 and rng.random() < c.effects["shading"].prob
    centre = 0.5 * (Pts.min(0) + Pts.max(0))
    light = photo.sample_light(rng, c.light, centre) if shading else None
    if shading:
        sv, su = _nodes(nv, SHADOW_STEP), _nodes(nu, SHADOW_STEP)
        coarse = np.stack([X, Y, Z], -1)[sv][:, su].reshape(-1, 3)
        maps = photo.shadow_maps(light, coarse, grid_faces(len(sv), len(su)), centre)
    lit = photo.shade(Pts, N, cam["eye"], light, maps) if shading else np.ones(len(Pts))
    uv, lf, zmap = rasterize(pix, depth, np.stack([uu.ravel(), vv.ravel(), lit], -1), faces, px)
    mask = np.isfinite(uv[..., 0])
    crop = _flat_part(pix, nv, nu, px, content)
    scale = 1.0 / max(crop[2] - crop[0], crop[3] - crop[1])
    page, _, paper_fx = draw_layout(layout, scale, P)
    ph, pw = page.shape
    x0 = int(np.clip(round(crop[0] * pw), 0, pw - px[0]))
    y0 = int(np.clip(round(crop[1] * ph), 0, ph - px[1]))
    flat = page[y0 : y0 + px[1], x0 : x0 + px[0]]
    crop = np.array([x0 / (pw - 1), y0 / (ph - 1), (x0 + px[0] - 1) / (pw - 1), (y0 + px[1] - 1) / (ph - 1)])
    lin = photo.SRGB_TO_LINEAR[page]
    mx = (np.nan_to_num(uv[..., 0]) * (pw - 1)).astype(np.float32)
    my = (np.nan_to_num(uv[..., 1]) * (ph - 1)).astype(np.float32)
    img = _page_in_photo(lin, mx, my, mask) * lf
    if c.background == "texture" and rng.random() >= c.black_prob:
        ys, xs = np.nonzero(~mask)
        bg = np.zeros(mask.shape, np.float32)
        bg[ys, xs] = (
            photo.table_texture(
                rng, photo.table_points(cam, px, xs=xs + 0.5, ys=ys + 0.5), photo.table_pixel(cam, px), ns
            )
            ** 2.2
        )
        if shading:
            q = photo.table_points(cam, px, BG_STEP)
            ok = np.isfinite(q[..., 0])
            t = np.ones(q.shape[:2], np.float32)
            t[ok] = photo.shade(q[ok], np.tile([0.0, 0.0, 1.0], (int(ok.sum()), 1)), cam["eye"], light, maps)
            bg = bg * cv2.resize(t, px, interpolation=cv2.INTER_LINEAR)
        img = np.where(mask, img, bg)
    else:
        img[~mask] = 0.0
    warped, effects = photo.camera_effects(rng, img.astype(np.float32), zmap, mask, c)
    meta = dict(
        view=view,
        light=None if light is None else dict(sharp=light["sharp"], ambient=light["ambient"]),
        effects=effects,
        paper_effects=paper_fx,
        page_frac=float(mask.mean()),
        border_on_page=float(np.r_[mask[0], mask[-1], mask[:, 0], mask[:, -1]].mean()),
        flat_part=crop.tolist(),
        flat_scale=scale,
    )
    grid3d, cpix = _flat_nodes(surf, crop, cam)
    grid2d = ((cpix - 0.5) / (np.array(px) - 1) * 2 - 1).T.reshape(2, nv, nu).astype(np.float32)
    u, v = np.nan_to_num(uv[..., 0], nan=-1), np.nan_to_num(uv[..., 1], nan=-1)
    mask = (u >= crop[0]) & (u <= crop[2]) & (v >= crop[1]) & (v <= crop[3])
    o = dict(flat=flat, warped=warped, grid2d=grid2d, grid3d=grid3d, mask=mask, meta=meta)
    return _portrait(o) if px[0] > px[1] else o


def _flat_nodes(surf, crop, cam):
    """The surface at the page grid's nodes spread over the flat's page part crop (looked up once in the working grid)
    and those nodes in the photo."""
    nv, nu = surf["X"].shape
    gu, gv = np.meshgrid(
        np.linspace(crop[0], crop[2], nu) * surf["u"][-1], np.linspace(crop[1], crop[3], nv) * surf["v"][-1]
    )
    grid3d = np.stack(page_points(surf, gu, gv))
    cpix, _ = project(grid3d.reshape(3, -1).T.astype(np.float64), cam)
    return grid3d.astype(np.float32), cpix


def _portrait(o):
    """A landscape sample (an open book) turned a quarter counterclockwise: images, mask and maps, grid2d's photo
    coordinates (x' = y, y' = -x) and grid3d's X, Y (X' = -Y, Y' = X, from 0) turned with them."""

    def r(a, axes=(0, 1)):
        return np.ascontiguousarray(np.rot90(a, 1, axes))

    g, (X, Y, Z) = r(o["grid2d"], (1, 2)), r(o["grid3d"], (1, 2))
    return dict(
        o,
        flat=r(o["flat"]),
        warped=r(o["warped"]),
        mask=r(o["mask"]),
        grid2d=np.stack([g[1], -g[0]]),
        grid3d=np.stack([Y.max() - Y, X - X.min(), Z]),
    )
