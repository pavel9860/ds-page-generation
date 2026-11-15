import os

os.environ["OMP_NUM_THREADS"]="1"
import collections
import sys

from dspages.multilang import realpages as R
from dspages.multilang.buckets import BUCKETS
from dspages.multilang.paths import LANGDS, LAYOUTS  # noqa: F401

if __name__=="__main__":
    langs=sorted({l for ls in BUCKETS.values() for l in ls.split()}-{"dv"})
    rows=R.build_real(str(LAYOUTS),langs,sys.argv[1],per=int(sys.argv[2]))
    print(len(rows),dict(collections.Counter(r["type"] for r in rows)))
    print(collections.Counter(r["lang_status"] for r in rows))
