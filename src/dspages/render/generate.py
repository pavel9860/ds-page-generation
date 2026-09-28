"""Part 3: a photo of a layout page on a 3D page surface, with its flat image, UV map and 3D map.

flat   (canvas h, w) uint8: the layout sheet (luminance) centred on a black canvas
warped (h, w) uint8: the photo (luminance)
uv     (uv h, uv w, 2) float16: page coordinates in [0, 1] (u across the width, v down the height) of each photo
       pixel, at the photo's aspect, NaN off the page
map3d  (map h, map w, 3) float16: X, Y, Z [mm] of the page grid (rows down the sheet height), table z = 0
mask   (h, w) bool: photo pixels on the page
"""
import cv2
import numpy as np

from ..config import Preset
from . import photo
from .camera import make_camera, project, vertex_normals
from .raster import grid_faces, rasterize

BG_STEP = 4


def _uv_map(uv, mask, size):
    m = mask.astype(np.float32)
    s = cv2.resize(np.nan_to_num(uv) * m[..., None], size, interpolation=cv2.INTER_AREA)
    k = cv2.resize(m, size, interpolation=cv2.INTER_AREA)
    out = s / np.maximum(k, 1e-6)[..., None]
    out[k < 0.5] = np.nan
    return out


def canvas(page, size):
    w, h = size
    out = np.zeros((h, w), np.uint8)
    y0, x0 = (h - page.shape[0]) // 2, (w - page.shape[1]) // 2
    out[y0:y0 + page.shape[0], x0:x0 + page.shape[1]] = page
    return out


def make_sample(rng, page, surf, P: Preset):
    """page: layout luminance sheet; surf: geometry.make_surface output. -> dict of arrays and meta."""
    c = P.render
    X, Y, Z = surf["X"], surf["Y"], surf["Z"]
    nv, nu = X.shape
    Pts = np.stack([X, Y, Z], -1).reshape(-1, 3)
    N = vertex_normals(X, Y, Z).reshape(-1, 3)
    faces = grid_faces(nv, nu)
    cam, view = make_camera(rng, Pts, N, c.px, c.camera)
    pix, depth = project(Pts, cam)
    uu, vv = np.meshgrid(np.linspace(0, 1, nu), np.linspace(0, 1, nv))
    shading = c.effects["shading"].prob > 0 and rng.random() < c.effects["shading"].prob
    light = photo.sample_light(rng, c.light, 0.5 * (Pts.min(0) + Pts.max(0))) if shading else None
    coarse = np.stack([X[::2, ::2], Y[::2, ::2], Z[::2, ::2]], -1)
    maps = photo.shadow_maps(light, coarse.reshape(-1, 3), grid_faces(*coarse.shape[:2]),
                             0.5 * (Pts.min(0) + Pts.max(0))) if shading else None
    lit = photo.shade(Pts, N, cam["eye"], light, maps) if shading else np.ones(len(Pts))
    uv, lf, zmap = rasterize(pix, depth, np.stack([uu.ravel(), vv.ravel(), lit], -1), faces, c.px)
    mask = np.isfinite(uv[..., 0])

    ph, pw = page.shape[:2]
    lin = photo.SRGB_TO_LINEAR[page]
    mx = (np.nan_to_num(uv[..., 0]) * (pw - 1)).astype(np.float32)
    my = (np.nan_to_num(uv[..., 1]) * (ph - 1)).astype(np.float32)
    img = cv2.remap(lin, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) * lf
    if c.background == "texture":
        Q = photo.table_points(cam, c.px)
        bg = cv2.resize(photo.table_texture(rng, Q[::2, ::2]), c.px, interpolation=cv2.INTER_LINEAR) ** 2.2
        if shading:
            q = Q[BG_STEP // 2::BG_STEP, BG_STEP // 2::BG_STEP]
            ok = np.isfinite(q[..., 0])
            t = np.ones(q.shape[:2], np.float32)
            t[ok] = photo.shade(q[ok], np.array([[0.0, 0.0, 1.0]]), cam["eye"], light, maps)
            bg = bg * cv2.resize(t, c.px, interpolation=cv2.INTER_LINEAR)
        img = np.where(mask, img, bg)
    else:
        img[~mask] = 0.0
    warped, effects = photo.camera_effects(rng, img.astype(np.float32), zmap, mask, c)
    uv_size = (round(c.map_px[1] * c.px[0] / c.px[1]), c.map_px[1])
    map3d = np.stack([cv2.resize(A.astype(np.float32), c.map_px, interpolation=cv2.INTER_LINEAR)
                     for A in (X, Y, Z)], -1)
    meta = dict(view=view, light=None if light is None else dict(sharp=light["sharp"], ambient=light["ambient"]),
                effects=effects, page_frac=float(mask.mean()))
    return dict(flat=canvas(page, P.layout.canvas_px), warped=warped, uv=_uv_map(uv, mask, uv_size).astype(np.float16),
                map3d=map3d.astype(np.float16), mask=mask, meta=meta)
