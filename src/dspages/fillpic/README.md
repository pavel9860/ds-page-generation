# fillpic

Fill / pic of a page image: a window around the content (widened to the canvas aspect), cut into `grid` x `grid` tiles; fill = share of tiles with ink, pic = mean
picture share per tile. Pages are the 768 x 1024 grey canvas of the shards (stretched, JPEG q90). A page without ink is fill 0, pic 0, window (0, 0, 1, 1).

    python -m dspages.fillpic gen | study | train        # training data, feature search, final models -> info/scan_detector/model
    python -m dspages.fillpic export                      # model + detector modules -> info/scan_detector/kaggle_detector
    python -m dspages.fillpic upload-detector             # ... as the Kaggle dataset
    python -m dspages.fillpic push|wait|fetch|rerun|merge # CPU kernels over the shard datasets, results -> info/scan_pct.parquet + manifest.sqlite scan_pct
    python -m dspages.fillpic fill-missing | reasons | recheck
    python -m dspages.fillpic show folder|hw|hw-new|random <out> ; labels

Use: `Detector.load(MODEL).predict([gray, ...])` -> `Result(window, ink, pic)` with `.fill` and `.pic_mean`.

Every number is in `config.py` (JSON overlay: DSPAGES_FILLPIC_CONFIG); a model is valid only for the geometry and map settings it was trained with. config, maps, tiles,
features, detector and shards need only numpy, OpenCV and XGBoost: they are what the kernel runs. Paths are in `paths.py`; data stays outside the repo.

cv2.Canny must not run on one thread (`maps.canny`): it corrupts the heap on pages with a very dense edge map.
