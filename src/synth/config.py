"""Tunable constants for the synthetic-data pipeline, grouped by stage."""

# ---- grid spec (texture.py) -----------------------------------------------
FINE_MM = 2.5           # fine grid pitch
HEAVY_MM = 10.0         # heavy grid pitch
FINE_PX_300 = 3.0       # fine line width @300dpi
HEAVY_PX_300 = 6.0      # heavy line width @300dpi
PX_PER_MM_300 = 300.0 / 25.4
INK = 0.10
PAPER = 0.93
NOISE_FIB_AMP = 0.06
NOISE_FIB_BLUR_SIGMA = 1.2
NOISE_GRAIN_AMP = 0.05
NOISE_EDGE_AMP = 0.25

# old (print-stage) crease/blur defect
OLD_CREASE_PROB = 0.2
OLD_CREASE_LENGTH_MM = (20, 50)
OLD_CREASE_CENTER_FRAC = (0.15, 0.85)
OLD_CREASE_S_FRAC = (0.015, 0.05)   # * span / 3.0
OLD_CREASE_BLUR_SIGMA_MIN = 2.0
OLD_CREASE_BLUR_SIGMA_SCALE = 0.6   # sigma = max(ppm*SCALE, MIN)

# fine crumple-crease texture (2D, print-stage) -- paired 1:1 with the 3D
# ridge network in geometry.py, same trigger, not an independent draw
FBM_CREASE_RIDGE_FRAC = 0.35
FBM_CREASE_SHARPEN = 1.0
FBM_CREASE_DARKEN = 0.97

# ---- 3D surface geometry (geometry.py) -------------------------------------
MAX_CAMERA_TILT_DEG = 30.0     # camera's own tilt range (combined cap = CAP_DEG below)

CREASE_R_MM = (1, 2)                  # bend radius at each end
CREASE_ANGLE_DEG = (15, 45)           # ramp angle between bends
CREASE_HEIGHT_MM = (1, 2)             # peak amplitude
CREASE_LENGTH_PX = (50, 150)          # along-mark length

FOLD_ANGLE_DEG = (15, 45)
FOLD_R_FRAC_OF_WIDTH = (0.05, 0.25)   # bottom radius, fraction of half-width
FOLD_HALF_WIDTH_PX = (100, 400)

BEND_PROB = 0.9
BEND_COUNT = (1, 3)
BEND_ANGLE_DEG = (10, 40)
BEND_S_FRAC = (0.08, 0.35)            # * span
FOLD_PROB = 0.6
FOLD_COUNT = (1, 3)
CREASE_CLUSTER_PROB = 0.1

UNDULATION_AMP_FRAC = (0.005, 0.03)   # * span

# ---- camera (geometry.py) --------------------------------------------------
CAMERA_ROLL_DEG = (-20, 20)
CAMERA_DIST_MM = (260, 350)
CAMERA_CENTER_JITTER_PX = (-40, 40)
CAMERA_BIG_SHIFT_PROB = 0.25
CAMERA_BIG_SHIFT_PX = (200, 600)

# ---- render pipeline (render.py) -------------------------------------------
CAP_DEG = 80.0                 # combined tilt+slope incidence cap
CAMERA_RESAMPLE_N = 40         # incidence-check mesh resolution
CAMERA_AZIMUTH_TRIES = 8       # re-roll azimuth this many times, tilt fixed
GRID_UV_DOWNSCALE = 5          # GT comes from a separate exact lattice projection, so the
                               # image warp itself tolerates coarser Newton downscale than text

SEVERITY_RANGE = (0.4, 1.4)
BASE_PITCH_PX = (25.5, 29.5)   # measured real: median 27.2, p5 24.5
PRINT_MARGIN_PX = (30, 60)     # measured real: 30-95px
DEFAULT_SUPERSAMPLE = 1.0     # texture's own coverage-based line AA makes oversampling
                               # redundant (verified: no visible moire/AA difference vs 1.2x)
DEFAULT_MARGIN = 1.2

# diffuse-blob shape (round, polar radius profile)
BLOB_HARMONICS = 3
BLOB_HARMONIC_AMP = (0.03, 0.10)
BLOB_R_CLIP = (0.7, 1.25)
BLOB_EDGE_BLUR_FRAC = 0.11     # * size_px

# local smudge (isotropic blob, e.g. thumb print)
LOCAL_BLUR_PROB = 0.25
LOCAL_BLUR_SIZE_PX = (100, 300)
LOCAL_BLUR_SIGMA = (1.5, 4.0)

# camera shake (directional, whole-frame)
CAMERA_SHAKE_PROB = 0.18
CAMERA_SHAKE_LENGTH_PX = (4, 16)
CAMERA_SHAKE_ANGLE_DEG = (0, 180)

# shadow/glare edge falloff, mutually exclusive
SHADOW_PROB = 0.10
GLARE_PROB = 0.10
SHADE_EDGE_FRAC = (0.10, 0.50)     # * out_size
SHADE_DIFFUSE_DIM = (0.30, 0.50)
GLARE_GAIN = (0.30, 0.60)
SHADE_CLIP = (0.4, 1.6)
SHADE_GRAD_STRENGTH = 0.5
AO_STRENGTH = 0.35
AO_BLUR_SIGMA = 3.0
AO_LAPLACIAN_KSIZE = 5

# bad-area damage (black, sharp-edged, round)
BAD_AREA_PROB = 0.05
BAD_AREA_CURVATURE_BIAS_PROB = 0.6
BAD_AREA_CURVATURE_PCTL = 95
BAD_AREA_SIZE_PX = (50, 150)

# depth-of-field defocus
DEFOCUS_FOCUS_JITTER = (-0.4, 0.4)
DEFOCUS_COC_SCALE = (0.008, 0.072)
DEFOCUS_COC_BASE = (0.3, 1.6)
DEFOCUS_LEVELS = (0.0, 0.6, 1.2, 2.0, 3.2, 5.0)

# exposure / color / noise / JPEG
EXPOSURE_RANGE = (0.72, 1.18)
GAMMA_RANGE = (0.85, 1.25)
COLOR_CAST_STD = 0.025
PHOTO_NOISE_STD = (0.004, 0.03)
JPEG_QUALITY = (45, 95)
INPAINT_RADIUS = 7
OFF_PAGE_INPAINT_MAX_FRAC = 0.02
OFF_PAGE_EDGE_FEATHER_PX = 4
OFF_PAGE_EDGE_JITTER_PX = 8.0

# ---- export (export.py) ----------------------------------------------------
PATCH = 1024
BORDER = 10              # bad-area collar dilation radius, px
ERASE_R = 12              # erase radius around a not-kept crossing
CANVAS = 1126             # crop headroom
EXPORT_SEED_MULT = 131071
EXPORT_ERASE_BLUR_SIGMA = 4

# ---- text pipeline (text_texture.py, text_render.py) -----------------------
TEXT_CANVAS = 1024         # output photo size, px
TEXT_PAGE_MM = 150.0       # flat page physical size (square)
TEXT_PAGE_PX = 1024        # flat page raster size at TEXT_PAGE_MM (square)
TEXT_UV_SIZE = 256         # exported UV-map resolution
TEXT_FONT_PT_RANGE = (8, 13)
TEXT_LINE_SPACING_RANGE = (1.15, 1.6)
TEXT_MARGIN_MM = 0.0
TEXT_CAMERA_DIST_MM = (200.0, 300.0)
TEXT_INPLANE_JITTER_DEG_RANGE = (3.0, 15.0)   # camera roll magnitude, random per sample
TEXT_ROTATE_AUG_PROB = 0.5         # 180deg flip of the flat text before warp
TEXT_BG_GRAY = 255   # uint8 scale -- rgb[] assignment in emulate_photo (white background)
TEXT_BLUR_SCALE = 0.5
TEXT_FILL_FRAC_RANGE = (0.97, 1.0)   # * out_size, corner-fit camera intrinsics
TEXT_MAX_TILT_DEG = 30.0
TEXT_LOWRES_PX = 256      # net-resolution photo, cv2.INTER_AREA downscale of the 1024 photo
TEXT_NEWTON_ITERS = 2
TEXT_UV_DOWNSCALE = 3
