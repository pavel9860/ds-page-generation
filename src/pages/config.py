from dataclasses import dataclass, field

from .scenes import SceneConfig

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
class ShallowCreaseConfig:
    """Shallow creases (creases.shallow): a height map on the page, not applied to the surface."""
    prob: float = 0.5                   # per page; 0 keeps pages crease-free
    mean_groups: float = 1.8            # groups per page ~ 1 + Poisson(mean_groups - 1)
    p_isolated: float = 0.3             # group is one unit
    mean_extra: float = 1.5             # else 2 + Poisson(mean_extra) units
    radius_mm: tuple = (15.0, 45.0)     # group radius
    spread_deg: float = 25.0            # direction spread inside a group
    singles_per_group: tuple = (3.0, 4.0)   # stand-alone single creases ~ Poisson(U(.) * n_groups)
    pitch_mm: float = 0.5               # height-map grid
    fold_clear_mm: float = 8.0          # drop shallow creases this close to a fold line


@dataclass
class PageConfig:
    width: float = 0.210
    height: float = 0.297
    paper: PaperConfig = field(default_factory=PaperConfig)
    scenes: SceneConfig = field(default_factory=SceneConfig)
    profile_along_long_prob: float = 0.5
    bend_dir_deg: float = 45.0          # profile direction turned up to this off the base axis (with along_long: any)
    two_profile_prob: float = 0.5
    n_slices: int = 9
    nu: int = 149
    nv: int = 105
    shell_nodes: tuple = (31, 43)
    shallow_creases: ShallowCreaseConfig = field(default_factory=ShallowCreaseConfig)
