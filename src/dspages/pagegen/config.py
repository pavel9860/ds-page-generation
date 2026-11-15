"""Numbers of the page generator (pagegen)"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PageCfg:
    """The page of the detection set"""
    ratio: float = 768 / 1024
    long_side: int = 1024
    margin: float = 0.05
    tolerance: float = 0.2                                  # a source page is used only when its ratio is within this share of A4
    min_hw: float = 0.02                                    # handwriting is at least this share of the page height
    slack: float = 1.25
    min_jitter: float = 0.15                                # a line below min_hw is written at min_hw plus up to this share
    max_hw: float = 0.06                                    # no handwritten line is taller than this share of the page height
    pitch_share: tuple = (0.5, 0.8)                         # height of a handwritten line on a printed page (form, notes), as a share of the printed line pitch


@dataclass(frozen=True)
class GenCfg:
    long_side: int = 1024
    text_min_font: float = 3
    natural_persons: tuple = ("school_cursive", "round_print", "compact_print", "backhand", "angular_cursive", "wide_garland",
                              "narrow_tall", "print_script", "left_hooked", "ruled_neat", "rounded_neat", "soft_upright",
                              "looped_cursive", "elderly")
    natural_styles: tuple = ("casual", "formal", "accurate", "educated", "professional")
    box_ink: float = 0.15
    box_pad: float = 0.3
    box_row_tol: float = 0.5
    crop_ink: float = 0.05
    p_blue_ink: float = 0.8
    pen_width_sd: float = 0.5
    pen_width_scale: float = 0.7       # pen width against the generator's own (generated strokes were 1.5x the real width)
    min_leader_run: int = 4
    leader_chars: str = ".…_·‥"
    gap_factor: float = 2.0
    paper_value: int = 225
    paper_sat: int = 40
    rule_len: int = 60
    rule_max_h: int = 5
    vrule_len: int = 25
    solid: int = 6
    occ_dilate: int = 5
    p_scan_hw: float = 0.5       # paper/scan effects: the same share for pages with and without handwriting, so the look
    scan_bleed: float = 0.0      # no text of the other side showing through: a mirrored shadow of the writing is an artifact, not a feature of the detector's pages
    scan_dpi: int = 87
    scan_bitonal: float = 0.0
    scan_blur: float = 0.8
    p_blur: float = 0.3
    blur_sigma: tuple = (0.4, 1.0)


PAGE = PageCfg()
