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


@dataclass
class PageConfig:
    width: float = 0.210
    height: float = 0.297
    paper: PaperConfig = field(default_factory=PaperConfig)
    kind_probs: dict = field(default_factory=lambda: {"clamp": 0.5, "center_support": 0.3, "two_support": 0.2})
    profile_along_long_prob: float = 0.5
    two_profile_prob: float = 0.5
    side_b_jitter: float = 0.35
    ranges: ProfileRanges = field(default_factory=ProfileRanges)
    support_frac: tuple = (0.0, 1.0)
    n_slices: int = 9
    nu: int = 149
    nv: int = 105
    shell_nodes: tuple = (31, 43)
