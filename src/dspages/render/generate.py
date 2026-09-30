"""Part 3: a photo of a layout page on a 3D page surface, with its flat image, UV map and 3D map.

All at the sheet's aspect: the photo has the flat page's size, the UV and 3D maps the page grid's (mesh).
flat   (h, w) uint8: the layout sheet (luminance)
warped (h, w) uint8: the photo (luminance)
uv     (mesh h, mesh w, 2) float32: page coordinates in [0, 1] (u across the width, v down the height) of the photo,
       NaN off the page
map3d  (mesh h, mesh w, 3) float32: X, Y, Z [mm] of the page grid (rows down the sheet height), table z = 0
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


def make_sample(rng, page, surf, P: Preset):
    """page: layout luminance sheet; surf: geometry.make_surface output. -> dict of arrays and meta."""
    c = P.render
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
    light = photo.sample_light(rng, c.light, 0.5 * (Pts.min(0) + Pts.max(0))) if shading else None
    maps = photo.shadow_maps(light, Pts, faces, 0.5 * (Pts.min(0) + Pts.max(0))) if shading else None
    occ = photo.sample_occluders(rng, light, 0.5 * (Pts.min(0) + Pts.max(0)), c.light.occluder_prob,
                                 float(np.ptp(Pts[:, :2], 0).max())) if shading else []
    lit = photo.shade(Pts, N, cam["eye"], light, maps) * photo.occlusion(Pts, light, occ) if shading else \
        np.ones(len(Pts))
    uv, lf, zmap = rasterize(pix, depth, np.stack([uu.ravel(), vv.ravel(), lit], -1), faces, px)
    mask = np.isfinite(uv[..., 0])

    ph, pw = page.shape[:2]
    lin = photo.SRGB_TO_LINEAR[page]
    mx = (np.nan_to_num(uv[..., 0]) * (pw - 1)).astype(np.float32)
    my = (np.nan_to_num(uv[..., 1]) * (ph - 1)).astype(np.float32)
    img = cv2.remap(lin, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) * lf
    if c.background == "texture" and rng.random() >= c.black_prob:
        ys, xs = np.nonzero(~mask)
        bg = np.zeros(mask.shape, np.float32)
        bg[ys, xs] = photo.table_texture(rng, photo.table_points(cam, px, xs=xs + 0.5, ys=ys + 0.5)) ** 2.2
        if shading:
            q = photo.table_points(cam, px, BG_STEP)
            ok = np.isfinite(q[..., 0])
            t = np.ones(q.shape[:2], np.float32)
            t[ok] = photo.shade(q[ok], np.tile([0.0, 0.0, 1.0], (int(ok.sum()), 1)), cam["eye"], light, maps) * \
                photo.occlusion(q[ok], light, occ)
            bg = bg * cv2.resize(t, px, interpolation=cv2.INTER_LINEAR)
        img = np.where(mask, img, bg)
    else:
        img[~mask] = 0.0
    warped, effects = photo.camera_effects(rng, img.astype(np.float32), zmap, mask, c)
    map3d = np.stack([X, Y, Z], -1).astype(np.float32)
    meta = dict(view=view, light=None if light is None else dict(sharp=light["sharp"], ambient=light["ambient"]),
                effects=effects, page_frac=float(mask.mean()))
    return dict(flat=page, warped=warped, uv=_uv_map(uv, mask, (nu, nv)).astype(np.float32), map3d=map3d, mask=mask,
                meta=meta)
