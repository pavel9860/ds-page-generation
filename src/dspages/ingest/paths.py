"""Where the ingest of acquired documents lives (outside the repo): store, work dir, the corpus tables it extends."""
from pathlib import Path

from ..fillpic.paths import EMB, INFO, LAYOUTS, MANIFEST_DB, UVTM  # noqa: F401

CC_ROOT = Path("/run/media/p/D/BUISNESS/corpus_collecting")
WORK = INFO / "ingest"
DB = WORK / "ingest.sqlite"
ACQ = LAYOUTS / "acquired"                         # pdf/<sha12>.pdf (dspages.acquire), img/<sha12>.jpg (hard links of the Lens images), lens/
LENS = ACQ / "lens"
ACQ_DB = INFO / "acquire" / "acquire.sqlite"
PARQUET = INFO / "scan_pct.parquet"
HW_WEIGHTS = Path("/run/media/p/D/BUISNESS/handwriting_detection/models/yolo26s-ds3k3250v3.pt")
LANG_MODELS = CC_ROOT / "models" / "langmodels"
