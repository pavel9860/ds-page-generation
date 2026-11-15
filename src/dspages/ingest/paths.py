"""Where the ingest of acquired documents lives (outside the repo): store, work dir, the corpus tables it extends."""

from ..fillpic.paths import EMB, INFO, LAYOUTS, MANIFEST_DB, UVTM  # noqa: F401
from ..paths import CORPUS, HWDET_WEIGHTS

CC_ROOT = CORPUS
WORK = INFO / "ingest"
DB = WORK / "ingest.sqlite"
ACQ = LAYOUTS / "acquired"  # pdf/<sha12>.pdf (dspages.acquire), img/<sha12>.jpg (hard links of the Lens images), lens/
LENS = ACQ / "lens"
ACQ_DB = INFO / "acquire" / "acquire.sqlite"
PARQUET = INFO / "scan_pct.parquet"
HW_WEIGHTS = HWDET_WEIGHTS
LANG_MODELS = CC_ROOT / "models" / "langmodels"
