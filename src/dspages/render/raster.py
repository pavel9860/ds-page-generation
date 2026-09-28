"""Triangle rasterizer for a page grid seen by a pinhole camera: per pixel page coordinates, depth and one
scalar attribute, perspective-correct, nearest surface wins (z-buffer)."""
import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def _raster(px, py, iz, attr, faces, w, h, out_uv, out_a, zbuf):
    for f in range(faces.shape[0]):
        i0, i1, i2 = faces[f, 0], faces[f, 1], faces[f, 2]
        x0, y0, x1, y1, x2, y2 = px[i0], py[i0], px[i1], py[i1], px[i2], py[i2]
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if area == 0.0 or iz[i0] <= 0 or iz[i1] <= 0 or iz[i2] <= 0:
            continue
        xmin = max(int(np.floor(min(x0, min(x1, x2)))), 0)
        xmax = min(int(np.ceil(max(x0, max(x1, x2)))), w - 1)
        ymin = max(int(np.floor(min(y0, min(y1, y2)))), 0)
        ymax = min(int(np.ceil(max(y0, max(y1, y2)))), h - 1)
        for y in range(ymin, ymax + 1):
            cy = y + 0.5
            for x in range(xmin, xmax + 1):
                cx = x + 0.5
                b0 = ((x1 - cx) * (y2 - cy) - (x2 - cx) * (y1 - cy)) / area
                b1 = ((x2 - cx) * (y0 - cy) - (x0 - cx) * (y2 - cy)) / area
                b2 = 1.0 - b0 - b1
                if b0 < -1e-7 or b1 < -1e-7 or b2 < -1e-7:
                    continue
                q = b0 * iz[i0] + b1 * iz[i1] + b2 * iz[i2]
                if q <= zbuf[y, x]:
                    continue
                zbuf[y, x] = q
                w0, w1, w2 = b0 * iz[i0] / q, b1 * iz[i1] / q, b2 * iz[i2] / q
                out_uv[y, x, 0] = w0 * attr[i0, 0] + w1 * attr[i1, 0] + w2 * attr[i2, 0]
                out_uv[y, x, 1] = w0 * attr[i0, 1] + w1 * attr[i1, 1] + w2 * attr[i2, 1]
                out_a[y, x] = w0 * attr[i0, 2] + w1 * attr[i1, 2] + w2 * attr[i2, 2]


def grid_faces(nv, nu):
    i = np.arange(nv - 1)[:, None] * nu + np.arange(nu - 1)[None, :]
    i = i.ravel()
    return np.concatenate([np.stack([i, i + 1, i + nu], 1), np.stack([i + 1, i + nu + 1, i + nu], 1)]).astype(np.int64)


def rasterize(pix, depth, attrs, faces, size):
    """pix (N, 2) pixel coords, depth (N,) along the optical axis, attrs (N, 3): the first two are the page
    coordinates, the third any scalar. -> uv (h, w, 2) NaN off the page, attr (h, w), depth (h, w) inf off."""
    w, h = size
    uv = np.full((h, w, 2), np.nan, np.float32)
    a = np.zeros((h, w), np.float32)
    zbuf = np.zeros((h, w), np.float64)
    _raster(pix[:, 0].astype(np.float64), pix[:, 1].astype(np.float64), 1.0 / depth.astype(np.float64),
            attrs.astype(np.float64), faces, w, h, uv, a, zbuf)
    with np.errstate(divide="ignore"):
        return uv, a, np.where(zbuf > 0, 1.0 / zbuf, np.inf).astype(np.float32)
