"""All parameters of the pipeline. Units: geometry in metres, layout and camera in mm, images in px."""
from dataclasses import dataclass, field, replace

G = 9.81


@dataclass(frozen=True)
class Paths:
    layouts: str = "/run/media/me/D/ML_DS/UVTM/Layouts"
    manifest: str = "/run/media/me/D/ML_DS/UVTM/Layouts/manifest_pages.jsonl"
    books: str = "/run/media/me/D/ML_DS/UVTM/Layouts/books/Texts"
    out: str = "/run/media/me/D/ML_DS/UVTM/dspages"
    font_dirs: tuple = ("/usr/share/fonts", "/usr/local/share/fonts", "~/.fonts")


@dataclass(frozen=True)
class PaperCfg:
    gsm: tuple = (50, 60, 80, 100, 120)             # paper types, g/m2, drawn uniformly
    ref_gsm: float = 80.0
    ref_bending: float = 0.688e-3                   # N m, at ref_gsm; bending ~ thickness^3 ~ gsm^3
    ref_membrane: float = 4.0e5                     # N/m, at ref_gsm; membrane ~ gsm
    poisson: float = 0.3


def paper_props(p: PaperCfg, gsm):
    r = gsm / p.ref_gsm
    return dict(gsm=float(gsm), q=gsm * 1e-3 * G, bending=p.ref_bending * r ** 3,
                membrane=p.ref_membrane * r, poisson=p.poisson)


@dataclass(frozen=True)
class SceneCfg:
    combos: tuple = (                               # (clamp, n_supports, folds, crumple): weight
        ((False, 0, True, False), 0.16),
        ((False, 1, False, False), 0.09), ((False, 2, False, False), 0.09), ((False, 3, False, False), 0.05),
        ((False, 1, True, False), 0.09), ((False, 2, True, False), 0.09), ((False, 3, True, False), 0.04),
        ((True, 0, False, False), 0.09), ((True, 1, False, False), 0.05), ((True, 2, False, False), 0.03),
        ((True, 0, True, False), 0.06), ((True, 1, True, False), 0.04),
        ((False, 0, False, True), 0.0463))            # crumple: 5% of all
    clamp_deg: tuple = (5.0, 35.0)
    clamp_free_len: float = 0.03
    support_edge: float = 0.05
    support_min_gap: float = 0.04
    support_h: tuple = (0.005, 0.040)
    reach_slope: float = 0.5
    skew_deg: float = 60.0
    skew_rise: float = 0.0035
    side_b_jitter: float = 0.35
    height_jitter: float = 0.2
    fold_templates: tuple = (((0.5,), 0.3), ((1 / 3, 2 / 3), 0.3), ((0.25, 0.5, 0.75), 0.1), (None, 0.3))
    fold_max: int = 4
    fold_jitter: float = 0.006
    fold_min_gap: float = 0.03
    fold_valley_prob: float = 0.5
    fold_angle_deg_median: float = 25.0
    fold_angle_sigma: float = 0.6
    fold_angle_deg: tuple = (3.0, 45.0)
    fold_total_deg: float = 70.0
    fold_total_deg_supported: float = 45.0
    fold_k_median: float = 0.01
    fold_k_sigma: float = 0.7
    fold_k: tuple = (0.003, 0.05)
    fold_side_b_angle_jitter: float = 0.25
    near_flat_prob: float = 0.15
    flat_support_h: tuple = (0.001, 0.005)
    flat_fold_deg: tuple = (1.0, 8.0)
    flat_clamp_deg: tuple = (2.0, 10.0)


@dataclass(frozen=True)
class ShallowCreaseCfg:
    prob: float = 0.5
    mean_groups: float = 1.8
    p_isolated: float = 0.3
    mean_extra: float = 1.5
    radius_mm: tuple = (15.0, 45.0)
    spread_deg: float = 25.0
    singles_per_group: tuple = (3.0, 4.0)
    k: tuple = (0.15, 0.9)
    w_max_mm: float = 1.0
    pitch_mm: float = 0.5
    fold_clear_mm: float = 8.0


@dataclass(frozen=True)
class DeepCreaseCfg:
    levels: tuple = (("medium", 0.55), ("heavy", 0.45))   # crumpled pages (no bends or folds)
    on_bends_prob: float = 0.1                    # light creases on top of bends and folds
    coverage: dict = field(default_factory=lambda: {"light": (0.1, 0.4), "medium": (0.3, 0.8), "heavy": (0.6, 1.0)})
    max_scale_mm: dict = field(default_factory=lambda: {"light": (30.0, 90.0), "medium": (25.0, 80.0),
                                                        "heavy": (15.0, 70.0)})   # page's largest crease length
    scale_ratio: tuple = (1.0, 5.0)               # largest / smallest local crease length on the page
    region_mm: tuple = (40.0, 120.0)              # size of the regions of equal scale / coverage
    density: float = 1.2                          # ridges per local length^2 of covered area
    k: dict = field(default_factory=lambda: {"light": (0.2, 0.6), "medium": (0.5, 1.2), "heavy": (0.7, 1.4)})
    max_depth_mm: dict = field(default_factory=lambda: {"light": 3.0, "medium": 8.0, "heavy": 15.0})   # valley to peak
    broad_n: dict = field(default_factory=lambda: {"light": (0, 2), "medium": (1, 3), "heavy": (2, 5)})
    max_length_mm: float = 150.0                 # longest crease; the shortest is the 3D grid pitch
    broad_length_mm: tuple = (80.0, 150.0)       # large creases: long, wide, low angle
    broad_spread: tuple = (4.0, 12.0)            # width multiplier of the crease profile
    broad_depth_frac: tuple = (0.2, 1.0)         # depth as a fraction of max_depth_mm
    kink_deg: float = 25.0
    pitch_mm: float = 0.5


@dataclass(frozen=True)
class GeometryCfg:
    sheet_mm: tuple = (210.0, 297.0)                # (width, height)
    mesh: tuple = (212, 300)                        # page grid (u along width, v along height)
    strip_segments: int = 150
    two_profile_prob: float = 0.5
    n_slices: int = 9
    along_long_prob: float = 0.5
    bend_dir_deg: float = 45.0
    support_frac: tuple = (0.2, 1.0)
    shell_nodes: tuple = (21, 29)
    scene: SceneCfg = SceneCfg()
    shallow: ShallowCreaseCfg = ShallowCreaseCfg()
    deep: DeepCreaseCfg = DeepCreaseCfg()


@dataclass(frozen=True)
class EffectCfg:
    prob: float = 0.0
    params: dict = field(default_factory=dict)


def _fx(prob, **params):
    return EffectCfg(prob, params)


@dataclass(frozen=True)
class GridCfg:
    fine_mm: float = 2.5
    heavy_every: int = 4
    fine_width_mm: float = 0.254                    # 3 px at 300 dpi
    heavy_width_mm: float = 0.508


@dataclass(frozen=True)
class PictureCfg:
    model: str = "~/.cache/dspages/yolov11n-doclaynet.onnx"   # python -m dspages.prep.layout_model
    weights_url: str = "https://huggingface.co/hantian/yolo-doclaynet/resolve/main/yolov11n-doclaynet.pt"  # AGPL-3.0
    imgsz: int = 1024
    picture_class: int = 6                          # DocLayNet "Picture"
    conf: float = 0.3
    iou: float = 0.5


@dataclass(frozen=True)
class TextFormatCfg:
    columns: tuple = ((1, 0.85), (2, 0.12), (3, 0.03))
    gutter_mm: tuple = (4.0, 10.0)
    heading_prob: float = 0.3
    heading_scale: tuple = (1.3, 2.2)


@dataclass(frozen=True)
class LayoutCfg:
    sheet_mm: tuple = (210.0, 297.0)
    canvas_px: tuple = (1024, 1365)                 # flat image (width, height); the sheet is fitted, rest padded
    margin_frac: tuple = (0.0, 0.10)                # blank margin around the content, per side, of the sheet
    min_fill: float = 0.8                           # fraction of the content window's patches with ink
    fill_grid: int = 10
    max_image: float = 0.3                          # max share of the window covered by pictures
    pictures: PictureCfg = PictureCfg()
    max_zoom: float = 4.0                           # PDF re-render limit for a 1:1 window
    script_mix: tuple = (("latin", 0.5), ("unknown", 0.1), ("cyrillic", 0.15), ("cjk", 0.1), ("other", 0.15))
    scripts: dict = field(default_factory=lambda: {
        "cyrillic": ("cyr", "bg", "ru", "uk", "sr", "mk", "be", "kk"),
        "cjk": ("zh", "ja", "ko"),
        "other": ("el", "he", "ar", "fa", "hi", "bn", "th", "ka", "hy", "ta", "te", "ur"),
        "unknown": ("unknown", None)})
    use_all: tuple = ("unknown",)                   # groups taking all their raster pages, outside script_mix
    grid_prob: float = 0.0                          # printed grid page instead of the manifest entry
    grid: GridCfg = GridCfg()
    font_pt: tuple = (10.0, 16.0)
    book_page_chars: int = 12000                    # one book page: the text of a full A4 sheet at the smallest font
    text_format: TextFormatCfg = TextFormatCfg()
    line_spacing: tuple = (1.15, 1.6)
    paper_tone: tuple = (0.88, 0.97)                # paper reflectance
    ink_tone: tuple = (0.04, 0.18)
    effects: dict = field(default_factory=lambda: {
        "paper_texture": _fx(1.0, fiber_amp=(0.01, 0.04), fiber_sigma=(0.6, 1.6), grain_amp=(0.005, 0.02),
                             cloud_amp=(0.0, 0.03), cloud_scale_mm=(8.0, 40.0)),
        "yellowing": _fx(0.15, strength=(0.03, 0.12), edge_mm=(5.0, 40.0)),
        "print_spread": _fx(0.5, sigma_px=(0.3, 1.2), gain=(-0.25, 0.25)),
        "ink_fade": _fx(0.3, strength=(0.05, 0.35), scale_mm=(10.0, 60.0)),
        "toner_speckle": _fx(0.3, density=(1e-5, 2e-4), size_px=(1, 3)),
        "banding": _fx(0.15, period_mm=(2.0, 12.0), amp=(0.01, 0.05)),
        "show_through": _fx(0.15, strength=(0.02, 0.08), blur_px=(1.5, 4.0)),
        "stains": _fx(0.1, n=(1, 3), size_mm=(10.0, 50.0), strength=(0.03, 0.12)),
        "old_creases": _fx(0.25, darken=(0.02, 0.08), shade=(0.02, 0.10), ink_loss=(0.0, 0.3)),
        "local_blur": _fx(0.25, n=(1, 3), size_mm=(15.0, 70.0), sigma_px=(0.8, 3.5)),
    })


@dataclass(frozen=True)
class CameraCfg:
    dist_mm: tuple = (200.0, 600.0)                 # camera to page centre
    tilt_deg: tuple = (0.0, 50.0)                   # optical axis to the table normal
    roll_deg: tuple = (-8.0, 8.0)
    fill: tuple = (0.93, 1.0)                       # page bbox to frame, along the tighter side
    shift: float = 0.3                              # page centre offset, fraction of the free space
    incidence_deg: float = 50.0                     # max view angle to the surface normal, anywhere on the page


@dataclass(frozen=True)
class LightCfg:
    ambient: tuple = (0.25, 0.6)
    dist_mm: tuple = (300.0, 2000.0)
    elev_deg: tuple = (30.0, 90.0)
    sharp_prob: float = 0.4                         # point light with hard cast shadows, else area light
    area_mm: tuple = (100.0, 600.0)
    specular: tuple = (0.0, 0.15)
    shininess: tuple = (8.0, 40.0)


@dataclass(frozen=True)
class RenderCfg:
    px: tuple = (1024, 1365)                        # warped photo (width, height)
    map_px: tuple = (212, 300)                      # UV / 3D maps
    background: str = "texture"                     # "black" or "texture"
    black_prob: float = 0.2                         # black background share under "texture"
    camera: CameraCfg = CameraCfg()
    light: LightCfg = LightCfg()
    effects: dict = field(default_factory=lambda: {
        "shading": _fx(1.0),
        "vignetting": _fx(0.5, strength=(0.05, 0.35)),
        "defocus": _fx(0.5, coc_px=(0.3, 1.0)),
        "motion_blur": _fx(0.12, length_px=(2.0, 10.0)),
        "exposure": _fx(1.0, ev=(-0.6, 0.4), gamma=(0.85, 1.2)),
        "iso_noise": _fx(0.9, iso=(50.0, 3200.0)),
        "jpeg": _fx(0.8, quality=(45, 95)),
    })


@dataclass(frozen=True)
class Preset:
    name: str
    paths: Paths = Paths()
    paper: PaperCfg = PaperCfg()
    layout: LayoutCfg = LayoutCfg()
    geometry: GeometryCfg = GeometryCfg()
    render: RenderCfg = RenderCfg()


def _off(effects):
    return {k: replace(v, prob=0.0) for k, v in effects.items()}


def _demo():
    sheet, flat, maps = (150.0, 150.0), (1024, 1024), (256, 256)
    lay = LayoutCfg(sheet_mm=sheet, canvas_px=flat, margin_frac=(0.0, 0.0))
    lay = replace(lay, effects=_off(lay.effects))
    ren = RenderCfg(px=flat, map_px=maps, background="black")
    ren = replace(ren, effects=_off(ren.effects))
    geo = GeometryCfg(sheet_mm=sheet, mesh=(256, 256), shallow=replace(ShallowCreaseCfg(), prob=0.0))
    return Preset("demo", layout=lay, geometry=geo, render=ren)


PRESETS = {"demo": _demo(), "full": Preset("full")}


def page_px(sheet_mm, canvas_px):
    """Sheet raster size fitted into the canvas, aspect kept."""
    s = min(canvas_px[0] / sheet_mm[0], canvas_px[1] / sheet_mm[1])
    return round(sheet_mm[0] * s), round(sheet_mm[1] * s)


def get_preset(name: str, **overrides) -> Preset:
    return replace(PRESETS[name], **overrides)
