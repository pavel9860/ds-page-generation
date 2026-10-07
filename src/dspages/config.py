"""All parameters of the pipeline. Units: geometry in metres, layout and camera in mm, images in px."""
from dataclasses import dataclass, field, replace

G = 9.81


@dataclass(frozen=True)
class Paths:
    layouts: str = "/run/media/me/D/ML_DS/UVTM/Layouts"
    manifest: str = "/run/media/me/D/ML_DS/UVTM/Layouts/manifest_pages.jsonl"
    files: str = "/run/media/me/D/ML_DS/UVTM/Layouts/manifest_files.jsonl"   # listing of the source files
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
    combos: tuple = (                               # (clamp, n_supports, folds, crumple): weight; clamp "book":
        ((False, 0, True, False), 0.16),
        ((False, 1, False, False), 0.09), ((False, 2, False, False), 0.09), ((False, 3, False, False), 0.05),
        ((False, 1, True, False), 0.09), ((False, 2, True, False), 0.09), ((False, 3, True, False), 0.04),
        ((True, 0, False, False), 0.09), ((True, 1, False, False), 0.05), ((True, 2, False, False), 0.03),
        ((True, 0, True, False), 0.06), ((True, 1, True, False), 0.04),
        ((False, 0, False, True), 0.0463),            # crumple
        (("book", 0, False, False), 0.3288))          # book page: 25% of all
    book_deg: tuple = (10.0, 45.0)                  # book page: spine edge angle; the rest lies on a plane of
    book_h: tuple = (0.0, 0.040)                    # height book_h (<= book_reach of the rise to it) from
    book_plane: tuple = (0.4, 0.8)                  # book_plane of the length on
    book_reach: float = 0.8
    book_bend_deg: float = 15.0                     # book page: bend direction within +-book_bend_deg of a page axis
    book_along_long_prob: float = 1 / 3             # profile along the long axis (bend lines across it): 1/3,
                                                    # bend lines along the long axis (spine on the long side): 2/3
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
    near_flat_prob: float = 0.0
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
    w_max_mm: float = 2.0
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
    max_depth_mm: dict = field(default_factory=lambda: {"light": 5.0, "medium": 8.0, "heavy": 15.0})   # valley to peak
    broad_n: dict = field(default_factory=lambda: {"light": (0, 2), "medium": (1, 3), "heavy": (2, 5)})
    max_length_mm: float = 150.0                 # longest crease; the shortest is the 3D grid pitch
    broad_length_mm: tuple = (80.0, 150.0)       # large creases: long, wide, low angle
    broad_spread: tuple = (4.0, 12.0)            # width multiplier of the crease profile
    broad_depth_frac: tuple = (0.2, 1.0)         # depth as a fraction of max_depth_mm
    kink_deg: float = 25.0


@dataclass(frozen=True)
class GeometryCfg:
    sheet_mm: tuple = (210.0, 297.0)                # (width, height)
    mesh: tuple = (256, 360)                        # page grid nodes (u across, v down): layout page_px / 4; the
                                                    # 3D and UV maps are this grid
    strip_segments: int = 150
    two_profile_prob: float = 0.5
    n_slices: int = 9
    along_long_prob: float = 0.5
    bend_dir_deg: float = 45.0
    support_frac: tuple = (0.2, 1.0)
    shell_nodes: tuple = (21, 29)
    view_tries: int = 5                             # surfaces drawn until one has a possible camera view (CameraCfg)
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
    imgsz: int = 640
    picture_class: int = 6                          # DocLayNet "Picture"
    conf: float = 0.3
    iou: float = 0.5
    large_area: float = 0.3                         # boxes over this page share need large_conf (whole forms
    large_conf: float = 0.75                        # come out as one low-score picture)


@dataclass(frozen=True)
class RefontCfg:
    """Numbers of the page restyling (layout/refont.py): thresholds of the page analysis, spacing and limits."""
    sheet_pt_w: float = 210.0 / 25.4 * 72
    head_ratio: float = 1.18
    header_pt: int = 45
    footer_pt: int = 70
    max_pages: int = 6
    bezier_steps: int = 8
    font_step: int = 2
    row_bin: int = 10
    row_reach: int = 3
    frame_span: float = 0.6
    frame_thin: float = 0.08
    edge_zone: float = 0.1
    background_share: float = 0.25
    trim_max: float = 0.85
    tilt: float = 0.3
    rule_thin: float = 2.5
    rule_long: float = 8.0
    script_purity: float = 0.8
    private_share: float = 0.05
    gutter_cover: float = 0.2
    gutter_em: float = 0.5
    max_columns: int = 4
    edge_bin: int = 3
    edge_lines: int = 3
    split_em: float = 1.6
    fit_floor: float = 0.4
    max_stretch: float = 3.0
    ragged_lines: int = 5
    ragged_percentile: float = 0.9
    indent_lines: int = 4
    start_share: float = 0.6
    aa: int = 4
    ragged_fill: float = 0.45
    run_step: int = 4
    run_zone: float = 0.14
    short_line_em: float = 3.0
    cjk_share: float = 0.25
    flush_share: float = 0.6
    table_step: int = 6
    column_line: int = 6
    justify_edge: float = 0.8
    edge_slack: float = 0.06
    row_overlap: float = 0.15
    room: float = 1.5
    font_tries: int = 6
    tab_em: float = 0.9
    justify_fill: float = 0.85
    edge_tol: float = 0.03
    size_step: int = 4
    touch_pt: float = 1.5                           # pictures this close to each other are tiles of one picture
    tile_cover: float = 0.9                         # when together they cover this share of their bounding box
    tab_max_em: float = 4.0                         # widest gap after a list marker that still joins the marker to its text
    space_em: float = 0.2                           # a gap between two letters wider than this is a space
    gap_reach: float = 3.0                          # lines (in font sizes) above and below a wide gap that tell its kind:
    gap_cross: float = 0.5                          # this share with ink over it makes it a justified space, not a gutter
    letter_gray: int = 128                          # drawings as letters are dark fills (a white box in a line is a field)
    ink_level: int = 250                            # gray below which a picture's pixel shows something
    cell_share: float = 0.25                        # a box (filled or closed drawing) up to this share of the page holds cells,
                                                    # not flowing text
    field_em: float = 3.0                           # a box at most this many font sizes high on a line's level is its field
    marker_em: float = 1.5                          # a shape this small just before a line is its radio button or check box
    balance_px: float = 2.0                         # precision of the column balance of a section
    dot_pt: float = 3.0                             # a picture no larger is a dot (of a dotted frame): nothing cleared under it
    glyph_gap: float = 0.3
    min_noto_cmap: int = 40
    min_text_chars: int = 150
    max_image_share: float = 0.5
    rule_min_pt: int = 20
    shrink: float = 0.93
    superscript_ratio: float = 0.75
    gap_body: float = 1.5
    text_image_share: float = 0.4
    align_inset: float = 0.1
    right_share: float = 0.5                        # widest share of its column a right-flush one-line block covers
    leading_max: float = 1.3                        # most a short frame spreads its lines to fill itself
    grow_max: float = 2.0                           # most a size class grows to fill its frames
    fill_share: float = 0.97                        # frames filled less than this are grown
    letter_em: float = 1.5                          # a drawing no larger than this many line heights, inside a line, is a letter
    letter_cover: float = 0.3                       # share of a span the letter drawings cover that makes it bold
    contain_share: float = 0.5                      # share of a block inside another one that makes the other hold it
    min_lead: float = 0.9                           # lines are at least this share of the font's own height apart
    min_line_em: float = 3.0                        # shortest line (in line heights) set beside an obstacle
    frame_gap: float = 2.5                          # lines of gap that end a text frame
    frame_lines: int = 6                            # lines a run of body paragraphs needs to be a text frame
    paragraph_lines: int = 3                        # and a paragraph of this many lines
    frame_slack: float = 0.3                        # share of a line a frame may be overrun by
    overprint_pt: float = 16.0                      # a line is compared with the lines starting within this height of it
    over_picture_share: float = 0.5                 # text lines inside pictures: more means a scan with hidden text
    line_em: float = 1.2                            # line height of a font without metrics
    join_back_em: float = 0.1
    join_ahead_em: float = 1.6
    script_share: float = 0.15
    source_gap_em: float = 0.15
    ascent_share: float = 0.8
    column_hit: float = 0.2
    rtl_share: float = 0.3
    same_size_pt: float = 0.3
    inline_share: float = 0.3
    row_tol_em: float = 0.35
    fixed_cover: float = 0.6
    repeat_cover: float = 0.7
    indent_min_em: float = 0.5
    indent_max_em: float = 4.0
    edge_match_em: float = 0.6
    full_line: float = 0.8
    pitch_tol: float = 1.2
    min_width_em: float = 8.0
    duplicate_overlap: float = 0.7
    body_band: float = 1.15
    single_pitch_em: float = 2.2
    max_lead: float = 2.4
    para_bin: float = 4.0
    lookback: int = 8
    regular_weight: int = 400
    bold_weight: int = 700
    min_alpha: float = 0.02


@dataclass(frozen=True)
class LayoutCfg:
    sheet_mm: tuple = (210.0, 297.0)
    page_px: tuple = (1024, 1440)                   # flat page and photo (width, height)
    margin_frac: tuple = (0.0, 0.10)                # blank margin around the content, per side, of the sheet
    min_fill: float = 0.75                          # fraction of the content window's patches with ink
    fill_grid: int = 16
    max_image: float = 0.25                         # max share of the window covered by pictures
    pictures: PictureCfg = PictureCfg()
    max_zoom: float = 4.0                           # PDF re-render limit for a 1:1 window
    manifest_pages: int = 20000                     # pages the manifest collects, shared by script_mix
    pages_per_pdf: int = 10                         # pages of one PDF (books: doc_share of their group's pages)
    tries_per_page: int = 3                         # pages checked per page taken
    doc_share: float = 0.02
    min_content_area: float = 0.01                  # content box of a raster page: share of the page, at least
    max_blank_margin: float = 0.50                  # and blank margin share, at most
    check_px: int = 800                             # longest side of the render the raster checks run on
    script_mix: tuple = (("latin", 0.5), ("unknown", 0.1), ("cyrillic", 0.15), ("cjk", 0.1), ("other", 0.15))
    scripts: dict = field(default_factory=lambda: {
        "cyrillic": ("cyr", "bg", "ru", "uk", "sr", "mk", "be", "kk", "ky", "tg", "mn", "kz", "tj"),
        "cjk": ("zh", "ja", "ko"),
        "other": ("el", "he", "ar", "fa", "hi", "bn", "th", "ka", "hy", "ta", "te", "ur", "gu", "kn", "ml", "pa",
                  "or", "si", "km", "lo", "my", "am", "as", "mni", "mr", "sa", "bo", "ckb", "ku", "ps", "ne", "mai", "sd"),
        "unknown": ("unknown", None)})
    sources: tuple = ("books_pdf", "bureaucracy", "corpus_add/pdf", "exams", "other_docs", "scientific_paper", "certificates",
                      "corpus_add/Images", "maps_blueprints", "filled_forms", "sheet_music", "flyers", "handwritten",
                      "magazines")                  # folders of Paths.layouts the listing takes
    raster_only: tuple = ("certificates", "corpus_add/Images", "maps_blueprints", "filled_forms", "sheet_music", "flyers",
                          "handwritten", "magazines")   # folders whose PDFs are never restyled, only rasterized
    use_all: tuple = ("unknown",)                   # groups taking all their raster pages, outside script_mix
    grid_prob: float = 0.0                          # printed grid page instead of the manifest entry
    grid: GridCfg = GridCfg()
    font_pt: tuple = (10.0, 16.0)                   # body text of a restyled PDF page
    role_scale: tuple = (0.6, 1.5)                  # its other sizes (headings, notes) relative to the source ratio
    noise_scale: tuple = (0.6, 1.5)                 # factor of the paper and photo noise feature sizes, one per page
    paper_tone: tuple = (0.88, 0.97)                # paper reflectance
    ink_tone: tuple = (0.04, 0.18)
    effects: dict = field(default_factory=lambda: {
        "paper_texture": _fx(1.0, fiber_amp=(0.0, 0.02), fiber_mm=(0.3, 1.5), fiber_stretch=(1.0, 8.0),
                             cloud_amp=(0.0, 0.02), cloud_mm=(2.0, 80.0), fleck_prob=0.2, fleck_amp=(0.03, 0.15),
                             grain_amp=(0.0, 0.012)),
        "yellowing": _fx(0.15, strength=(0.02, 0.08), edge_mm=(5.0, 40.0)),
        "print_spread": _fx(0.5, sigma_px=(0.3, 1.0), gain=(-0.2, 0.2)),
        "ink_fade": _fx(0.3, strength=(0.03, 0.2), scale_mm=(5.0, 100.0)),
        "toner_speckle": _fx(0.3, density=(1e-5, 1e-4), size_px=(1, 3)),
        "banding": _fx(0.15, period_mm=(1.0, 20.0), amp=(0.005, 0.03)),
        "show_through": _fx(0.15, strength=(0.01, 0.05), blur_px=(1.0, 5.0)),
        "stains": _fx(0.1, n=(1, 3), size_mm=(5.0, 60.0), strength=(0.02, 0.08)),
        "old_creases": _fx(0.25, darken=(0.01, 0.05), shade=(0.01, 0.06), ink_loss=(0.0, 0.2)),
        "local_blur": _fx(0.25, n=(1, 3), size_mm=(15.0, 70.0), sigma_px=(0.8, 3.5)),
    })


@dataclass(frozen=True)
class CameraCfg:
    dist_mm: tuple = (200.0, 400.0)                 # camera to page centre (the tilt follows from the page's shape)
    roll_deg: tuple = (-4.8, 4.8)
    fill: tuple = (0.93, 1.0)                       # page bbox to frame, along the tighter side
    max_offset: float = 0.3                         # camera's horizontal offset from the page centre / page size
    side_offset: float = 0.5                        # the offset allowed when no view within max_offset sees all the page
    shift: float = 0.3                              # page centre offset, fraction of the free space
    incidence_deg: float = 65.0                     # max view angle to the surface normal, anywhere on the page
    view_dirs: int = 256                            # view directions spread evenly over the tilts the offsets allow
    view_step: int = 4                              # the views are worked out on every 4th node of the page grid


@dataclass(frozen=True)
class LightCfg:
    ambient: tuple = (0.05, 0.35)
    dist_mm: tuple = (300.0, 2000.0)
    elev_deg: tuple = (20.0, 80.0)
    sharp_prob: float = 0.4                         # point light with hard cast shadows, else area light
    area_mm: tuple = (100.0, 600.0)
    specular: tuple = (0.0, 0.15)
    shininess: tuple = (8.0, 40.0)


@dataclass(frozen=True)
class RenderCfg:
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
    sheet = (150.0, 150.0)
    lay = LayoutCfg(sheet_mm=sheet, page_px=(1024, 1024), margin_frac=(0.0, 0.0))
    lay = replace(lay, effects=_off(lay.effects))
    ren = RenderCfg(background="black")
    ren = replace(ren, effects=_off(ren.effects))
    geo = GeometryCfg(sheet_mm=sheet, mesh=(256, 256), shallow=replace(ShallowCreaseCfg(), prob=0.0))
    return Preset("demo", layout=lay, geometry=geo, render=ren)


PRESETS = {"demo": _demo(), "full": Preset("full")}


def get_preset(name: str, **overrides) -> Preset:
    return replace(PRESETS[name], **overrides)
