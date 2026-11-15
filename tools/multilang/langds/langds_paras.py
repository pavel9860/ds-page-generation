import os

os.environ["OMP_NUM_THREADS"] = "1"
import sys
from concurrent.futures import ProcessPoolExecutor

from dspages.multilang import realpages as R
from dspages.multilang.buckets import BUCKETS
from dspages.multilang.paths import LANGDS, LAYOUTS  # noqa: F401

if __name__ == "__main__":
    import json
    import pathlib

    langs = [
        l
        for l in sorted({l for ls in BUCKETS.values() for l in ls.split()} - {"dv"})
        if len(json.loads(pathlib.Path(sys.argv[1], "paras_before_ocrfix", l + ".json").read_text()).get("p", [])) >= 30
        and not len(json.loads(pathlib.Path(sys.argv[1], "paras_before_ocrfix", l + ".json").read_text()).get("h", []))
        == 24
    ]
    jobs = [(str(LAYOUTS), l, sys.argv[1], int(sys.argv[2])) for l in langs]
    with ProcessPoolExecutor(10) as ex:
        res = dict(ex.map(R.paragraph_pool, jobs, chunksize=1))
    ok = [l for l, c in res.items() if c["p"] >= 30]
    print(len(res), "languages;", len(ok), "with >=30 real paragraphs; without:", sorted(set(res) - set(ok)))
