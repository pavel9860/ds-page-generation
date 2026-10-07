"""Part 3: a photo of a layout page on a 3D page surface, with its flat image, UV map and 3D map.

All at the sheet's aspect, turned portrait when it is landscape (an open book): the photo has the flat page's size, the UV and 3D maps the page grid's (mesh): node (i, j)
is the page point at evenly spaced page coordinates (rows down the sheet height), so neither map has holes.
flat   (h, w) uint8: the layout sheet (luminance)
warped (h, w) uint8: the photo (luminance)
grid2d (2, mesh h, mesh w) float32: x, y of each page node in the photo, in [-1, 1] with align_corners=True (pixel
       centres at -1 and 1): the grid_sample grid that flattens the photo
grid3d (3, mesh h, mesh w) float32: X, Y, Z [mm] of each page node on the curved surface, table z = 0
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
    cam, view = make_camera(rng, Pts.reshape(nv, nu, 3), N.reshape(nv, nu, 3), px, c.camera)
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
    meta = dict(view=view, light=None if light is None else dict(sharp=light["sharp"], ambient=light["ambient"]),
                effects=effects, page_frac=float(mask.mean()))
    grid2d = ((pix - 0.5) / (np.array(px) - 1) * 2 - 1).T.reshape(2, nv, nu).astype(np.float32)
    grid3d = np.stack([X, Y, Z]).astype(np.float32)
    o = dict(flat=page, warped=warped, grid2d=grid2d, grid3d=grid3d, mask=mask, meta=meta)
    return _portrait(o) if px[0] > px[1] else o


def _portrait(o):
    """A landscape sample (an open book) turned a quarter counterclockwise: images, mask and maps, grid2d's photo
    coordinates (x' = y, y' = -x) and grid3d's X, Y (X' = -Y, Y' = X, from 0) turned with them."""
    def r(a, axes=(0, 1)):
        return np.ascontiguousarray(np.rot90(a, 1, axes))
    g, (X, Y, Z) = r(o["grid2d"], (1, 2)), r(o["grid3d"], (1, 2))
    return dict(o, flat=r(o["flat"]), warped=r(o["warped"]), mask=r(o["mask"]),
                grid2d=np.stack([g[1], -g[0]]), grid3d=np.stack([Y.max() - Y, X - X.min(), Z]))
