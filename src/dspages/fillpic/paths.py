"""Where the data of the pipeline lives (outside the repo): the corpus, the manifest, the embedding table, the model and the results."""
import json
import re
from pathlib import Path

from ..hand.config import PATHS

LAYOUTS = Path(PATHS.layouts)
UVTM = Path(PATHS.uvtm)
INFO = LAYOUTS / "archive" / "info"
WORK = INFO / "scan_detector"
MODEL = WORK / "model"
PCT = INFO / "scan_pct"
PARQUET = INFO / "scan_pct.parquet"
MANIFEST_DB = LAYOUTS / "manifest.sqlite"
TESTS = UVTM / "tests"
EMB = UVTM / "Labels" / "embedding" / "emb_out"
HEX = re.compile(r"(?<![0-9a-f])[0-9a-f]{12}(?![0-9a-f])")


def doc_hash(path):
    return next(iter(HEX.findall(str(path).rsplit("/", 1)[-1])), None)


def doc_paths(want=None):
    """{document id of the embedding table: absolute path of its file}, only `want` if given."""
    out = {}
    for line in open(EMB / "manifest.jsonl"):
        r = json.loads(line)
        if want is None or r["id"] in want:
            out[r["id"]] = str(LAYOUTS / r["path"])
    return out
