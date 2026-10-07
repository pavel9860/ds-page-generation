"""Part 3: a photo of a layout page on a 3D page surface, with its flat image, UV map and 3D map.

All at the sheet's aspect: the photo has the flat page's size, the UV and 3D maps the page grid's (mesh).
flat   (h, w) uint8: the layout sheet (luminance)
warped (h, w) uint8: the photo (luminance)
uv     (mesh h, mesh w, 2) float32: page coordinates in [0, 1] (u across the width, v down the height) of the photo,
       NaN off the page
map3d  (mesh h, mesh w, 3) float32: X, Y, Z [mm] of the page grid (rows down the sheet height), table z = 0
xyz    (mesh h, mesh w, 3) float32: map3d at the page coordinates uv of each photo node (same grid as uv), NaN off
       the page
mask   (h, w) bool: photo pixels on the page
"""
import cv2
import numpy as np

from ..config import Preset
from . import photo
from .camera import make_camera, project, vertex_normals
from .raster import grid_faces, rasterize

BG_STEP = 4
SHADOW_STEP = 2                 # shadow maps from every 2nd node: the 384 px maps cannot resolve the full grid
MIP_LEVELS = (0.0, 0.5, 1.0, 2.0)   # px, page blurs blended by how much a photo pixel shrinks the page
MIP_STEP = 4                    # the shrink is measured on every 4th pixel (it changes slowly)


def _uv_map(uv, mask, size):
    m = mask.astype(np.float32)
    s = cv2.resize(np.nan_to_num(uv) * m[..., None], size, interpolation=cv2.INTER_AREA)
    k = cv2.resize(m, size, interpolation=cv2.INTER_AREA)
    out = s / np.maximum(k, 1e-6)[..., None]
    out[k < 0.5] = np.nan
    return out


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
    pos = cv2.resize(np.interp(sigma, MIP_LEVELS, np.arange(len(MIP_LEVELS))).astype(np.float32), mx.shape[::-1],
                     interpolation=cv2.INTER_LINEAR)
    out = np.zeros(mx.shape, np.float32)
    for k, s in enumerate(MIP_LEVELS):
        if pos.min() < k + 1 and pos.max() > k - 1:
            wk = np.clip(1 - np.abs(pos - k), 0, 1)
            src = lin if s == 0 else cv2.GaussianBlur(lin, (0, 0), s)
            out += wk * cv2.remap(src, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out


def make_sample(rng, page, surf, P: Preset):
    """page: layout luminance sheet; surf: geometry.make_surface output. -> dict of arrays and meta."""
    c = P.render
    ns = rng.uniform(*P.layout.noise_scale)          # factor of this photo's noise feature sizes
    px = page.shape[1::-1]                                   # photo (w, h) = the flat page's
    X, Y, Z = surf["X"], surf["Y"], surf["Z"]
    nv, nu = X.shape
    Pts = np.stack([X, Y, Z], -1).reshape(-1, 3)
    N = vertex_normals(X, Y, Z).reshape(-1, 3)
    faces = grid_faces(nv, nu)
    cam, view = make_camera(rng, Pts, N, px, c.camera)
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

    ph, pw = page.shape[:2]
    lin = photo.SRGB_TO_LINEAR[page]
    mx = (np.nan_to_num(uv[..., 0]) * (pw - 1)).astype(np.float32)
    my = (np.nan_to_num(uv[..., 1]) * (ph - 1)).astype(np.float32)
    img = _page_in_photo(lin, mx, my, mask) * lf
    if c.background == "texture" and rng.random() >= c.black_prob:
        ys, xs = np.nonzero(~mask)
        bg = np.zeros(mask.shape, np.float32)
        bg[ys, xs] = photo.table_texture(rng, photo.table_points(cam, px, xs=xs + 0.5, ys=ys + 0.5), photo.table_pixel(cam, px), ns) ** 2.2
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
    map3d = np.stack([X, Y, Z], -1).astype(np.float32)
    meta = dict(view=view, light=None if light is None else dict(sharp=light["sharp"], ambient=light["ambient"]),
                effects=effects, page_frac=float(mask.mean()))
    uvm = _uv_map(uv, mask, (nu, nv)).astype(np.float32)
    xyz = cv2.remap(map3d, uvm[..., 0] * (nu - 1), uvm[..., 1] * (nv - 1), cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE)
    xyz[np.isnan(uvm[..., 0])] = np.nan
    return dict(flat=page, warped=warped, uv=uvm, map3d=map3d, xyz=xyz, mask=mask, meta=meta)
