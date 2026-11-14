"""Every number of the fill / pic detector. Frozen dataclasses with the values the shipped model was trained with; `Config.load(path)` overlays a JSON file
(nested keys as in the dataclasses) and DSPAGES_FILLPIC_CONFIG points to one. A model is valid only for the geometry and map settings it was trained with."""
import json
import os
from dataclasses import dataclass, field, fields, replace


@dataclass(frozen=True)
class Geometry:
    grid: int = 16
    canvas_w: int = 768
    canvas_h: int = 1024
    mask_side: int = 512
    margin: float = 0.05
    min_ink_px: int = 3
    jpeg_quality: int = 90
    picture_cells: int = 128
    page_sized_picture: float = 0.6

    @property
    def canvas(self):
        return self.canvas_h, self.canvas_w

    @property
    def infer_margin(self):
        return self.margin / 2


@dataclass(frozen=True)
class Ink:
    line_ksize: int = 15
    line_thresh: int = 36
    eps: float = 0.01


@dataclass(frozen=True)
class Maps:
    hat_thresholds: tuple = (36, 12, 20, 28, 48)
    box_size: int = 5
    mid_grey: tuple = (70, 195)
    mid_flat: float = 18.0
    dark_level: int = 100
    hist_edges: tuple = (64, 128, 192, 224)
    stroke_ksize: int = 9
    stroke_thresh: int = 24
    blob_area: int = 400
    canny: tuple = ((50, 150), (20, 60))
    canny_min_threads: int = 2
    std_kernels: tuple = (3, 13)
    open_ksize: int = 7
    run_length: int = 31
    sobel_scale: float = 1020.0
    grad_scale: float = 1e6
    spread_scale: float = 128.0
    coherence_eps: float = 1e-3
    paper_percentile: int = 90
    paper_stride: int = 4
    noise_stride: int = 2


@dataclass(frozen=True)
class Features:
    neighbours: int = 3
    blur_sizes: tuple = (7, 5)
    base_derived: tuple = ("n3", "ctx7", "page")
    candidate_derived: tuple = ("ctx5", "pagemean", "pos")


@dataclass(frozen=True)
class Train:
    n_pages: int = 4000
    workers: int = 4
    sample_seconds: int = 60
    train_trees: int = 200
    train_depth: int = 6
    train_lr: float = 0.12
    study_trees: int = 150
    study_depth: int = 6
    study_lr: float = 0.15
    study_rows: int = 250000
    n_eval_docs: int = 300
    subsample: float = 0.8
    colsample: float = 0.7
    seed: int = 0
    keep_gain: float = 0.99
    keep_loss: float = 1.02
    chunk_pages: int = 100
    degrade_scale: tuple = (0.45, 1.0)
    degrade_blur_prob: float = 0.6
    degrade_blur_sigma: tuple = (0.4, 1.4)
    degrade_gain: tuple = (0.7, 1.0)
    degrade_offset: tuple = (-10, 20)
    degrade_shade: float = 0.18
    degrade_shade_xy: float = 0.1
    degrade_noise: tuple = (1.5, 9)
    degrade_dust_prob: float = 0.4
    degrade_dust: tuple = (0.0002, 0.002)
    degrade_jpeg: tuple = (35, 92)
    degrade_binarise_prob: float = 0.15


@dataclass(frozen=True)
class Kaggle:
    user: str = "pavel9860"
    detector_dataset: str = "scan-pct-detector"
    shard_dataset: str = "pages-768x1024-{}"
    shard_datasets: tuple = tuple(range(17))
    batch: int = 32
    hard_stop_hours: float = 10.0
    cpu_slots: int = 5
    poll_seconds: int = 300
    local_render_seconds: int = 120
    local_pdf_seconds: int = 100


@dataclass(frozen=True)
class Languages:
    cjk: tuple = ("ja", "zh", "ko")
    rare: tuple = ("am", "bo", "dz", "km", "lo", "my", "si", "ti", "ka", "hy", "ne", "hi", "bn", "ta", "te", "ml", "kn", "gu", "pa", "or", "th", "he", "ar", "fa", "sa",
                   "mn", "ug", "yi", "ps", "ks", "dv", "ckb")


@dataclass(frozen=True)
class Config:
    geometry: Geometry = field(default_factory=Geometry)
    ink: Ink = field(default_factory=Ink)
    maps: Maps = field(default_factory=Maps)
    features: Features = field(default_factory=Features)
    train: Train = field(default_factory=Train)
    kaggle: Kaggle = field(default_factory=Kaggle)
    languages: Languages = field(default_factory=Languages)

    @classmethod
    def load(cls, path=None):
        path = path or os.environ.get("DSPAGES_FILLPIC_CONFIG")
        if not path:
            return cls()
        over = json.load(open(path))
        parts = {}
        for f in fields(cls):
            cur = getattr(cls(), f.name)
            parts[f.name] = replace(cur, **{k: (tuple(v) if isinstance(v, list) else v) for k, v in over.get(f.name, {}).items()})
        return cls(**parts)


CONFIG = Config.load()
