"""Four stages compose a rendered page photo + exact GT:
  1. print emulation      -- flat 2D print texture (texture.py)
  2. 3D map emulation      -- the paper's height field z(U,V) (geometry.py)
  3. UV mapping            -- project the print onto the 3D map, into image space
  4. photo emulation       -- defocus, shading, blur/shake, exposure/noise/JPEG

`render_raw()` does stages 1-3 at whatever resolution the caller asks for.
`generate()` calls it then runs stage 4 (`emulate_photo`) at that SAME
resolution -- correct when out_size IS the final image. Off-page pixels are
filled by emulate_photo's own inpainting.

export.py needs a different order: it renders stages 1-3 at a large CANVAS
(headroom for a fully-on-page crop), then must run stage 4 AFTER cropping,
not before -- every stage-4 effect sizes/positions itself in absolute
pixels or a fraction of `out_size`, only meaningful relative to the final
image.
"""
import cv2
import numpy as np

from . import config as cfg
from .geometry import make_camera, make_surface, project, retry_camera
from .newton_cpu import apply_uv_mapping_newton
from .texture import FINE_MM, lattice_points, make_old_creases, render_grid_texture

CAP_DEG = cfg.CAP_DEG


def _emulate_print(mm_w, mm_h, ppm, rng, margin_mm, add_creases=False):
    """Stage 1: flat print texture + its own print-quality blur defect.
    `add_creases`: also paint the fine fBm crumple-crease texture (paired
    1:1 with the 3D ridge network by render_raw's shared trigger draw)."""
    old_creases = make_old_creases(rng, mm_w)
    return render_grid_texture(mm_w, mm_h, ppm, rng, old_creases=old_creases, margin_mm=margin_mm,
                               add_fbm_creases_flag=add_creases)


def _emulate_3d_map(rng, span, ppmm, severity, add_creases=None):
    """Stage 2: the paper's height field z(U,V) -- bends, folds, creases,
    undulation. `add_creases`: force the crease-network decision (see
    render_raw); None draws it internally at CREASE_CLUSTER_PROB."""
    return make_surface(rng, span=span, ppmm=ppmm, severity=severity, add_creases=add_creases)


def _camera_for_surface(rng, zf, mm_w, mm_h, out_size, ppmm, tilt_deg,
                        n=cfg.CAMERA_RESAMPLE_N, az_tries=cfg.CAMERA_AZIMUTH_TRIES):
    """Camera with incidence <=CAP_DEG to every point on the surface. Tilt
    is drawn once and kept fixed; only azimuth is retried."""
    MU, MV = np.meshgrid(np.linspace(0, mm_w, n), np.linspace(0, mm_h, n))
    target_tilt = rng.uniform(0, cfg.MAX_CAMERA_TILT_DEG) if tilt_deg is None else tilt_deg
    return retry_camera(lambda: make_camera(rng, mm_w, mm_h, out_size, ppmm, target_tilt),
                        zf, MU, MV, az_tries, CAP_DEG)


def _apply_uv_mapping(tex, zf, cam, mm_w, mm_h, out_size, ppm):
    """Stage 3: project the flat print through the 3D surface into image
    space, via per-pixel Newton inversion (mm_w == mm_h always, square
    page)."""
    img, depth, page = apply_uv_mapping_newton(tex, zf, cam, mm_w, out_size, ppm,
                                               downscale=cfg.GRID_UV_DOWNSCALE)
    return img, depth.astype(np.float32), page


def emulate_photo(img, depth, page, cam, out_size, rng, bg_value=0, blur_scale=1.0,
                  bad_area_enabled=True, border_jitter_enabled=True, shade_enabled=True):
    """Stage 4: shading, defocus, smudge/shake blur, exposure/noise/JPEG.
    `out_size` must be the FINAL image size. `bg_value`: off-page fill
    color. `blur_scale`: multiplies defocus/smudge/shake strength.
    `bad_area_enabled`, `border_jitter_enabled`, `shade_enabled`: on for
    the grid pipeline; text turns off bad_area/border_jitter (no crop-
    style off-page edge to fake) and can turn off shade. Returns (rgb,
    bad_area) -- bad_area marks a damaged region whose GT must be dropped."""
    if shade_enabled:
        img = img * _local_shade(depth, out_size, rng)
    off_frac = (~page).mean()
    do_inpaint = 0 < off_frac <= cfg.OFF_PAGE_INPAINT_MAX_FRAC
    if do_inpaint:
        img8 = np.clip(img * 255, 0, 255).astype(np.uint8)
        img8 = cv2.inpaint(img8, (~page).astype(np.uint8), cfg.INPAINT_RADIUS, cv2.INPAINT_TELEA)
        img = img8.astype(np.float32) / 255.0
    img = _defocus_blur(img, depth, page, cam, rng, blur_scale)
    img = _local_blur(img, out_size, rng, blur_scale)
    img = _camera_shake(img, rng, blur_scale)
    rgb = _photometric(img, rng)
    bad_area = np.zeros((out_size, out_size), bool)
    if bad_area_enabled:
        rgb, bad_area = _bad_area(rgb, depth, out_size, rng)
    if not do_inpaint:
        if border_jitter_enabled:
            sdf = (cv2.distanceTransform(page.astype(np.uint8), cv2.DIST_L2, 3)
                  - cv2.distanceTransform((~page).astype(np.uint8), cv2.DIST_L2, 3))
            noise = rng.standard_normal(page.shape, dtype=np.float32)
            noise = cv2.GaussianBlur(noise, (0, 0), cfg.OFF_PAGE_EDGE_FEATHER_PX)
            noise *= cfg.OFF_PAGE_EDGE_JITTER_PX / (noise.std() + 1e-6)
            keep = (sdf + noise) > 0
        else:
            keep = page
        rgb[~keep] = bg_value
    return rgb, bad_area


def _diffuse_blob(out_size, rng, size_px, sharp=False, center=None):
    """A round blob mask (wobbly circle via low-freq polar radius profile,
    not thresholded noise -- avoids spiky/jagged shapes). `sharp`=False
    soft-blurs the edge; shape is round either way."""
    cx, cy = center if center is not None else rng.uniform(0, out_size, 2)
    theta = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    r_profile = np.ones_like(theta)
    for k in range(1, cfg.BLOB_HARMONICS + 1):
        amp = rng.uniform(*cfg.BLOB_HARMONIC_AMP) / k
        r_profile += amp * np.cos(k * theta + rng.uniform(0, 2 * np.pi))
    r_profile = np.clip(r_profile, *cfg.BLOB_R_CLIP)
    base_r = size_px / 2
    pts = np.stack([cx + base_r * r_profile * np.cos(theta),
                    cy + base_r * r_profile * np.sin(theta)], -1)

    canvas = np.zeros((out_size, out_size), np.uint8)
    cv2.fillPoly(canvas, [pts.astype(np.int32)], 255)
    canvas = canvas.astype(np.float32) / 255.0
    if not sharp:
        canvas = cv2.GaussianBlur(canvas, (0, 0), size_px * cfg.BLOB_EDGE_BLUR_FRAC)
    return canvas


def _local_blur(img, out_size, rng, blur_scale=1.0):
    """A local smudge (isotropic blob, e.g. thumb print) -- distinct from
    _camera_shake below, which is directional and whole-frame."""
    if rng.random() >= cfg.LOCAL_BLUR_PROB:
        return img
    region = _diffuse_blob(out_size, rng, rng.uniform(*cfg.LOCAL_BLUR_SIZE_PX))
    sigma = rng.uniform(*cfg.LOCAL_BLUR_SIGMA) * blur_scale
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    return img * (1 - region) + blurred * region


def _camera_shake(img, rng, blur_scale=1.0):
    """Whole-frame directional motion blur -- a real hand-shake PSF is a
    short streak hitting every pixel, unlike the local smudge above."""
    if rng.random() >= cfg.CAMERA_SHAKE_PROB:
        return img
    length = max(int(round(rng.uniform(*cfg.CAMERA_SHAKE_LENGTH_PX) * blur_scale)), 3)
    angle = rng.uniform(*cfg.CAMERA_SHAKE_ANGLE_DEG)
    kernel = np.zeros((length, length), np.float32)
    kernel[length // 2, :] = 1.0
    center = (length / 2 - 0.5, length / 2 - 0.5)
    rot = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (length, length))
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(img, -1, kernel)


_EDGE_SIDES = ('top', 'bottom', 'left', 'right', 'corner')


def edge_shade(out_size, rng):
    """Shadow OR glare falloff from one edge/corner, mutually exclusive,
    each independently SHADOW_PROB/GLARE_PROB likely. `out_size` must be
    the final exported size -- called from export.py post-crop, not from
    render.py's stage 4. Returns None if neither fires."""
    r = rng.random()
    if r >= cfg.SHADOW_PROB + cfg.GLARE_PROB:
        return None
    is_glare = r >= cfg.SHADOW_PROB

    side = rng.choice(_EDGE_SIDES)
    extent = rng.uniform(*cfg.SHADE_EDGE_FRAC) * out_size
    yy, xx = np.mgrid[0:out_size, 0:out_size].astype(np.float32)
    if side == 'top':
        d = yy
    elif side == 'bottom':
        d = out_size - 1 - yy
    elif side == 'left':
        d = xx
    elif side == 'right':
        d = out_size - 1 - xx
    else:
        cx = rng.choice([0, out_size - 1])
        cy = rng.choice([0, out_size - 1])
        d = np.hypot(xx - cx, yy - cy)
    t = np.clip(1.0 - d / max(extent, 1e-6), 0, 1)
    t = 0.5 - 0.5 * np.cos(np.pi * t)   # cosine ease, not a linear kink
    if is_glare:
        gain = rng.uniform(*cfg.GLARE_GAIN)
        return 1.0 + gain * t
    dim = rng.uniform(*cfg.SHADE_DIFFUSE_DIM)
    return 1.0 - dim * t


def _local_shade(depth, out_size, rng):
    """Directional shading (gradient projected onto a random light
    direction) times curvature-based ambient occlusion: concave regions
    (crease/fold valleys) are darkened regardless of light direction, same
    as a real crevice getting less ambient light from all directions."""
    d = np.nan_to_num(depth, nan=np.nanmedian(depth))
    gy, gx = np.gradient(d)
    light_ang = rng.uniform(0, 2 * np.pi)
    proj = gx * np.cos(light_ang) + gy * np.sin(light_ang)
    p99 = np.percentile(np.abs(proj), 99) + 1e-6
    grad_shade = 1.0 - cfg.SHADE_GRAD_STRENGTH * np.clip(proj / p99, 0, 1)

    d_smooth = cv2.GaussianBlur(d.astype(np.float32), (0, 0), cfg.AO_BLUR_SIGMA)
    curvature = cv2.Laplacian(d_smooth, cv2.CV_32F, ksize=cfg.AO_LAPLACIAN_KSIZE)
    concave = np.clip(curvature, 0, None)   # depth curving away from camera
    p99c = np.percentile(concave, 99) + 1e-6
    ao = 1.0 - cfg.AO_STRENGTH * np.clip(concave / p99c, 0, 1)

    return np.clip(grad_shade * ao, *cfg.SHADE_CLIP)


def _bad_area(rgb, depth, out_size, rng):
    """A rare damaged region, painted flat black with a sharp border. 60%
    of the time centred at the steepest local curvature (a bend/crease);
    40% a plain round spot elsewhere."""
    if rng.random() >= cfg.BAD_AREA_PROB:
        return rgb, np.zeros((out_size, out_size), bool)
    center = None
    if rng.random() < cfg.BAD_AREA_CURVATURE_BIAS_PROB:
        d = np.nan_to_num(depth, nan=np.nanmedian(depth))
        gy, gx = np.gradient(d)
        curvature = np.hypot(gx, gy)
        thresh = np.percentile(curvature, cfg.BAD_AREA_CURVATURE_PCTL)
        ys, xs = np.where(curvature >= thresh)
        if len(xs):
            i = rng.integers(0, len(xs))
            center = (float(xs[i]), float(ys[i]))
    region = _diffuse_blob(out_size, rng, rng.uniform(*cfg.BAD_AREA_SIZE_PX),
                           sharp=True, center=center) > 0.5
    rgb = rgb.copy()
    rgb[region] = 0
    return rgb, region


def render_raw(seed, out_size=512, base_pitch_px=None, severity=None, tilt_deg=None,
               supersample=cfg.DEFAULT_SUPERSAMPLE, margin=cfg.DEFAULT_MARGIN,
               print_margin_px=None, geom_override=None):
    """Stages 1-3: the geometric composite before any photo-capture effects,
    at `out_size` resolution. `seed` alone fully determines the sample.
    Returns the RNG continued (not reset) so a caller doing its own stage 4
    (export.py) draws from the same reproducible stream `generate()` would."""
    rng = np.random.default_rng(seed)
    if severity is None:
        severity = rng.uniform(*cfg.SEVERITY_RANGE)
    if base_pitch_px is None:
        base_pitch_px = rng.uniform(*cfg.BASE_PITCH_PX)
    if print_margin_px is None:
        print_margin_px = rng.uniform(*cfg.PRINT_MARGIN_PX)
    ppmm = base_pitch_px / FINE_MM
    mm_w = mm_h = margin * out_size / ppmm
    ppm = supersample * ppmm
    print_margin_mm = print_margin_px / ppmm

    add_creases = rng.random() < cfg.CREASE_CLUSTER_PROB
    tex = _emulate_print(mm_w, mm_h, ppm, rng, print_margin_mm, add_creases=add_creases)
    zf = geom_override if geom_override is not None else \
        _emulate_3d_map(rng, mm_w, ppmm, severity, add_creases=add_creases)
    cam = _camera_for_surface(rng, zf, mm_w, mm_h, out_size, ppmm, tilt_deg)

    img, depth, page = _apply_uv_mapping(tex, zf, cam, mm_w, mm_h, out_size, ppm)
    gt = _project_lattice(zf, cam, mm_w, mm_h, out_size, print_margin_mm)
    return dict(image=img, depth=depth, page=page, cam=cam, gt=gt, rng=rng,
               page_frac=float(page.mean()))


def generate(seed, out_size=512, base_pitch_px=None, severity=None, tilt_deg=None,
             supersample=cfg.DEFAULT_SUPERSAMPLE, margin=cfg.DEFAULT_MARGIN,
             print_margin_px=None, geom_override=None, skip_photo_emulation=False):
    """render_raw() then stage 4 at that same out_size -- correct for
    direct/standalone use (see module docstring for export.py's own flow).
    `skip_photo_emulation`: stop after stage 3, for inspecting the raw warp."""
    raw = render_raw(seed, out_size, base_pitch_px, severity, tilt_deg,
                     supersample, margin, print_margin_px, geom_override)
    img, depth, page, cam = raw['image'], raw['depth'], raw['page'], raw['cam']
    gt, rng = raw['gt'], raw['rng']

    bad_area = np.zeros((out_size, out_size), bool)
    if skip_photo_emulation:
        rgb = cv2.cvtColor(np.clip(img * 255, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        rgb, bad_area = emulate_photo(img, depth, page, cam, out_size, rng)
        if bad_area.any():
            ix = np.clip(gt[:, 0].astype(int), 0, out_size - 1)
            iy = np.clip(gt[:, 1].astype(int), 0, out_size - 1)
            gt = gt[~bad_area[iy, ix]]
    return dict(image=rgb, gt=gt, bad_area=bad_area, page_frac=raw['page_frac'])


def _defocus_blur(img, depth, page, cam, rng, blur_scale=1.0):
    """Depth-of-field: blur radius grows with |depth - focus plane|."""
    dvalid = depth[page]
    if dvalid.size == 0:
        dvalid = np.array([cam['dist']], np.float32)
    depth_range = dvalid.max() - dvalid.min()
    focus = np.median(dvalid) + rng.uniform(*cfg.DEFOCUS_FOCUS_JITTER) * depth_range / 2
    coc = np.abs(np.nan_to_num(depth, nan=focus) - focus) * rng.uniform(*cfg.DEFOCUS_COC_SCALE)
    coc = coc * blur_scale + rng.uniform(*cfg.DEFOCUS_COC_BASE) * blur_scale
    levels = np.array(cfg.DEFOCUS_LEVELS)
    idx = np.clip(np.searchsorted(levels, coc), 1, len(levels) - 1)
    lo, hi = levels[idx - 1], levels[idx]
    w = np.clip((coc - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
    out = np.zeros_like(img)
    stack = {}
    for i in range(1, len(levels)):
        m = idx == i
        if not m.any():
            continue
        for j in (i - 1, i):
            if j not in stack:
                stack[j] = img if levels[j] == 0 else cv2.GaussianBlur(img, (0, 0), levels[j])
        out[m] = stack[i - 1][m] * (1 - w[m]) + stack[i][m] * w[m]
    return out


def _photometric(img, rng):
    """Exposure/gamma/noise/JPEG, ranges matched to measured real-photo stats."""
    img = np.clip(img * rng.uniform(*cfg.EXPOSURE_RANGE), 0, 1) ** rng.uniform(*cfg.GAMMA_RANGE)
    cast = np.array([1.0, 1.0, 1.0]) + rng.normal(0, cfg.COLOR_CAST_STD, 3)
    rgb = np.clip(img[..., None] * cast[None, None, :], 0, 1)
    rgb += rng.uniform(*cfg.PHOTO_NOISE_STD) * rng.standard_normal(rgb.shape, dtype=np.float32)
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    q = int(rng.integers(*cfg.JPEG_QUALITY))
    return cv2.imdecode(cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, q])[1], 1)


def _project_lattice(zf, cam, mm_w, mm_h, out_size, margin_mm=0.0):
    """Exact analytic GT: lattice points landing in-frame."""
    LU, LV = lattice_points(mm_w, mm_h, margin_mm)
    LZ = zf(LU, LV)
    lx, ly, _ = project(LU, LV, LZ, cam)
    inb = (lx >= 0) & (lx < out_size) & (ly >= 0) & (ly < out_size)
    return np.stack([lx[inb], ly[inb]], -1)
