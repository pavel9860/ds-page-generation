"""CPU text-page render: Newton per-pixel UV inversion (no mesh-locate),
reusing render.py's camera/photo pipeline (emulate_photo) with
text_texture's flat texture in place of the grid ink. Camera intrinsics
are fit so the page's projected corners stay inside the frame -- no
big-shift, no crop headroom."""
import cv2
import numpy as np

from . import config as cfg
from .geometry import isometric_mesh, make_surface, project, retry_camera, rotation_matrix
from .newton_cpu import newton_uv_invert as _newton_uv_invert_generic
from .render import CAP_DEG, emulate_photo
from .text_texture import render_flat_text, sample_snippet

PAGE_MM = cfg.TEXT_PAGE_MM
PAGE_PX = cfg.TEXT_PAGE_PX
PPMM = PAGE_PX / PAGE_MM


def newton_uv_invert(zf, cam, out_size, n_iters=cfg.TEXT_NEWTON_ITERS, downscale=cfg.TEXT_UV_DOWNSCALE):
    return _newton_uv_invert_generic(zf, cam, PAGE_MM, out_size, n_iters, downscale)


def _flat_grid(n):
    lin = np.linspace(0, PAGE_MM, n)
    return np.meshgrid(lin, lin)


def _fit_camera_to_fill(rng, zf, out_size, tilt_deg, n=cfg.CAMERA_RESAMPLE_N):
    """Project the full grid (not just flat-UV corners -- creases/folds
    can bulge past them) at a placeholder focal length, then rescale and
    recenter so the bounding box fits inside the frame."""
    mu, mv = _flat_grid(n)
    mz = zf(mu, mv)
    az = rng.uniform(0, 2 * np.pi)
    jitter_mag = rng.uniform(*cfg.TEXT_INPLANE_JITTER_DEG_RANGE)
    roll = np.deg2rad(rng.choice([-1.0, 1.0]) * jitter_mag)
    R = rotation_matrix(az, roll, tilt_deg)
    dist = rng.uniform(*cfg.TEXT_CAMERA_DIST_MM)
    centre = np.array([PAGE_MM / 2, PAGE_MM / 2, 0.0])

    cam0 = dict(R=R, f=PPMM * dist, dist=dist, centre=centre, cx=out_size / 2, cy=out_size / 2)
    px, py, _ = project(mu, mv, mz, cam0)
    bbox_w = px.max() - px.min()
    bbox_h = py.max() - py.min()
    fill_frac = rng.uniform(*cfg.TEXT_FILL_FRAC_RANGE)
    scale = fill_frac / max(bbox_w / out_size, bbox_h / out_size, 1e-6)
    f = cam0['f'] * scale

    cam1 = dict(cam0, f=f)
    px, py, _ = project(mu, mv, mz, cam1)
    cx = out_size / 2 - (px.max() + px.min()) / 2 + out_size / 2
    cy = out_size / 2 - (py.max() + py.min()) / 2 + out_size / 2
    return dict(cam1, cx=cx, cy=cy)


def _camera_for_surface(rng, zf, out_size, n=cfg.CAMERA_RESAMPLE_N, az_tries=cfg.CAMERA_AZIMUTH_TRIES):
    """Camera with incidence <=CAP_DEG to every point on the surface. Tilt
    is drawn once and kept fixed; only azimuth (and jitter/dist/fill_frac)
    is retried -- fit_camera_to_fill already guarantees in-frame corners
    for any pose, this only bounds foreshortening severity."""
    mu, mv = _flat_grid(n)
    tilt_deg = rng.uniform(0, cfg.TEXT_MAX_TILT_DEG)
    return retry_camera(lambda: _fit_camera_to_fill(rng, zf, out_size, tilt_deg, n),
                        zf, mu, mv, az_tries, CAP_DEG)


def _flat_to_photo_map(zf, cam, n):
    """Regular (n,n) flat-UV grid (mm) projected into photo-pixel space --
    used by rectify_backward (flat pixel -> photo pixel, gathered)."""
    mu, mv = _flat_grid(n)
    mz = zf(mu, mv)
    px, py, _ = project(mu, mv, mz, cam)
    return mu, mv, px, py


def _downsample_masked(arrs: list, mask: np.ndarray, out_n: int) -> tuple:
    """Mask-aware area-average downsample (unpremultiplied alpha: weight by
    mask, resize, divide back out) of same-shape arrays to (out_n, out_n)."""
    m = mask.astype(np.float32)
    m_ds = cv2.resize(m, (out_n, out_n), interpolation=cv2.INTER_AREA)
    out = []
    for a in arrs:
        a_ds = cv2.resize((a * mask).astype(np.float32), (out_n, out_n), interpolation=cv2.INTER_AREA)
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append(a_ds / m_ds)
    return out, m_ds > 0.999


def _build_maps(U: np.ndarray, V: np.ndarray, page: np.ndarray, zf, uv_size: int) -> tuple:
    """uv_map: (uv_size, uv_size, 2) float32, photo-pixel indexed, [0,1]
    flat-texture coords, NaN off-page (downsampled from the full-res Newton
    inverse). map3d: (uv_size, uv_size, 3) float32, the page's own isometric
    mesh (arc-length spacing, not photo-pixel indexed) -- ground truth for
    the physical 3D shape, independent of camera/photo pixels."""
    (Uds, Vds), valid = _downsample_masked([U, V], page, uv_size)
    uv_map = np.stack([Uds / PAGE_MM, Vds / PAGE_MM], axis=-1).astype(np.float32)
    uv_map[~valid] = np.nan
    X, Y, Z = isometric_mesh(zf, PAGE_MM, PAGE_MM, uv_size)
    map3d = np.stack([X, Y, Z], axis=-1).astype(np.float32)
    return uv_map, map3d


def rectify_backward(zf, cam, rgb_hr, out_size, page_px=PAGE_PX, bg_value=cfg.TEXT_BG_GRAY):
    """Backward UV-map application: for each FLAT-page pixel, gather the
    hi-res photo value at that pixel's forward-projected photo position --
    dewarps the photo back to flat layout, for visually checking the
    camera/surface/UV-warp chain is self-consistent."""
    _, _, px, py = _flat_to_photo_map(zf, cam, page_px)
    mapx = px.astype(np.float32)
    mapy = py.astype(np.float32)
    return cv2.remap(rgb_hr, mapx, mapy, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=(bg_value,) * 3)


def render_text_raw(seed, corpus_paths, font_files, out_size=cfg.TEXT_CANVAS, uv_size=cfg.TEXT_UV_SIZE,
                    flat_tex=None, add_creases=None):
    """One text-page sample: (img, depth, page, cam, zf, uv_map, flat_tex)
    at CPU resolution `out_size`. uv_map: (uv_size, uv_size, 2) float32 in
    [0,1] flat-texture coords, NaN where off-page. flat_tex: the source
    flat page (post rotate-aug, pre-warp), float32 (PAGE_PX, PAGE_PX).
    `flat_tex`: if given (float32, PAGE_PX x PAGE_PX, INK/PAPER-scaled
    reflectance), used as the flat page instead of synthesizing one with
    render_flat_text -- corpus_paths/font_files are then ignored.
    `add_creases`: forwarded to make_surface -- the 3D crease/fold/bend
    network on the page's own height field (None draws it internally at
    CREASE_CLUSTER_PROB, same as before; False disables it entirely)."""
    rng = np.random.default_rng(seed)
    severity = rng.uniform(*cfg.SEVERITY_RANGE)
    zf = make_surface(rng, span=PAGE_MM, ppmm=PPMM, severity=severity, add_creases=add_creases)
    cam = _camera_for_surface(rng, zf, out_size)

    if flat_tex is not None:
        tex = flat_tex
    else:
        snippet = sample_snippet(corpus_paths, rng)
        tex = render_flat_text(snippet, rng, PAGE_PX, PAGE_MM, font_files)
        if rng.random() < cfg.TEXT_ROTATE_AUG_PROB:
            tex = np.ascontiguousarray(tex[::-1, ::-1])

    U, V, depth, page = newton_uv_invert(zf, cam, out_size)
    mapx, mapy = (U * PPMM).astype(np.float32), (V * PPMM).astype(np.float32)
    img = cv2.remap(tex, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

    uv_map, map3d = _build_maps(U, V, page, zf, uv_size)
    return img, depth, page, cam, zf, uv_map, tex, map3d


def generate_text(seed, corpus_paths, font_files, out_size=cfg.TEXT_CANVAS, uv_size=cfg.TEXT_UV_SIZE,
                  lowres_px=cfg.TEXT_LOWRES_PX, shade_enabled=True, flat_tex=None,
                  blur_scale=None, add_creases=None, photometric_enabled=True):
    """render_text_raw + stage-4 photo emulation. Returns (photo_hr,
    photo_lowres, uv_map, flat_page, cam, zf, map3d); cam/zf are for
    diagnostics (rectify_backward, quiver visualization). map3d: (uv_size,
    uv_size, 3) float32 (X, Y, Z) mm surface coords, the page's own
    isometric mesh (see _build_maps). `flat_tex`: see render_text_raw.
    `add_creases`: forwarded to render_text_raw (3D crease network).
    `blur_scale`: defocus/smudge/shake strength (None: cfg.TEXT_BLUR_SCALE).
    `photometric_enabled`: see emulate_photo."""
    img, depth, page, cam, zf, uv_map, tex, map3d = render_text_raw(seed, corpus_paths, font_files,
                                                                      out_size, uv_size, flat_tex,
                                                                      add_creases=add_creases)
    rng = np.random.default_rng(seed * cfg.EXPORT_SEED_MULT + 3)
    blur_scale = cfg.TEXT_BLUR_SCALE if blur_scale is None else blur_scale
    rgb, _ = emulate_photo(img, depth, page, cam, out_size, rng, bg_value=cfg.TEXT_BG_GRAY,
                           blur_scale=blur_scale, bad_area_enabled=False,
                           border_jitter_enabled=False, shade_enabled=shade_enabled,
                           photometric_enabled=photometric_enabled)
    rgb_lowres = cv2.resize(rgb, (lowres_px, lowres_px), interpolation=cv2.INTER_AREA)
    flat_page = np.clip(tex * 255, 0, 255).astype(np.uint8)
    return rgb, rgb_lowres, uv_map, flat_page, cam, zf, map3d
