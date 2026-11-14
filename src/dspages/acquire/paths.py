"""Where the acquired documents and their bookkeeping live (outside the repo)."""
from pathlib import Path

from ..fillpic.paths import LAYOUTS, MANIFEST_DB, doc_hash  # noqa: F401

CC_ROOT = Path("/run/media/p/D/BUISNESS/corpus_collecting")
CC_INDEX = CC_ROOT / "cc_pdf_index"
TOOLS = Path(__file__).resolve().parents[3] / "tools"
WORK = LAYOUTS / "archive" / "info" / "acquire"
STORE = LAYOUTS / "acquired"                       # pdf/<sha12>.pdf
DB = WORK / "acquire.sqlite"
