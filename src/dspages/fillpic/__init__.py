"""Fill / pic detector of page images: window around the content, G x G tiles, fill = share of tiles with ink, pic = mean picture share.
config, maps, tiles, features, detector and shards need only numpy, OpenCV and XGBoost (they are what the Kaggle kernel runs); the rest builds the training data,
trains, renders pages, evaluates and drives Kaggle."""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from .config import CONFIG, Config  # noqa: E402
from .detector import Detector, Result  # noqa: E402

__all__ = ["CONFIG", "Config", "Detector", "Result"]
