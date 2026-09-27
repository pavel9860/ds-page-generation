from dataclasses import dataclass, field

G = 9.81


@dataclass
class PaperConfig:
    grammage_gsm: float = 80.0
    bending_stiffness: float = 0.688e-3
    membrane_stiffness: float = 4.0e5
    poisson: float = 0.3

    @property
    def q(self):
        return self.grammage_gsm * 1e-3 * G


@dataclass
class ProfileRanges:
    clamp_deg: tuple = (5.0, 60.0)
    center_h: tuple = (0.005, 0.040)
    two_p1_frac: tuple = (0.05, 0.30)
    two_p2_frac: tuple = (0.70, 0.95)
    two_h: tuple = (0.010, 0.040)
    # folded: template crease positions (fractions of length), else 1..max random
    fold_templates: dict = field(default_factory=lambda: {
        (0.5,): 0.3, (1 / 3, 2 / 3): 0.3, (0.25, 0.5, 0.75): 0.1, None: 0.3})
    fold_max: int = 4
    fold_jitter: float = 0.006          # m, crease position noise
    fold_min_gap: float = 0.03          # m, between creases and to edges
    fold_valley_prob: float = 0.5       # per crease: valley (+) vs mountain (-)
    fold_angle_deg_median: float = 25.0 # residual rest angle after unfolding, lognormal
    fold_angle_sigma: float = 0.6
    fold_angle_deg: tuple = (3.0, 110.0)
    fold_k_median: float = 0.01         # N*m/m per rad, crease hinge stiffness, lognormal
    fold_k_sigma: float = 0.7
    fold_side_b_angle_jitter: float = 0.25


@dataclass
class PageConfig:
    width: float = 0.210
    height: float = 0.297
    paper: PaperConfig = field(default_factory=PaperConfig)
    kind_probs: dict = field(default_factory=lambda: {"clamp": 0.35, "center_support": 0.2, "two_support": 0.15, "folded": 0.3})
    profile_along_long_prob: float = 0.5
    two_profile_prob: float = 0.5
    side_b_jitter: float = 0.35
    ranges: ProfileRanges = field(default_factory=ProfileRanges)
    support_frac: tuple = (0.0, 1.0)
    n_slices: int = 9
    nu: int = 149
    nv: int = 105
    shell_nodes: tuple = (31, 43)
