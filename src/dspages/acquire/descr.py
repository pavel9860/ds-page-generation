"""Text descriptions of documents, the second input of the url model: the text of the first two pages. Corpus documents: from the file. Candidates of the
Common Crawl index: from the first HEAD bytes of the capture (one range request; about half of the pdfs show text there) -> WORK/descr/part_NNNN.parquet
(url, text), resumable."""
import re
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import httpx
import pandas as pd
import pymupdf

from .paths import CC_ROOT, LAYOUTS, WORK, doc_hash

HEAD = 200_000
CHARS = 600
pymupdf.TOOLS.mupdf_display_errors(False)


def clean(t):
    return re.sub(r"\s+", " ", t).strip()[:CHARS]


def pdf_text(data):
    d = pymupdf.open(stream=data, filetype="pdf")
    return clean(" ".join(d[i].get_text() for i in range(min(2, d.page_count))))


def corpus_one(rel):
    try:
        with pymupdf.open(LAYOUTS / rel) as d:
            return clean(" ".join(d[i].get_text() for i in range(min(2, d.page_count))))
    except Exception:
        return ""


def corpus(paths, workers=8):
    """{document hash: text} for the pdf documents of the in-scope pages (paths: (path, page) list)."""
    first = {}
    for p, _ in paths:
        if p.lower().endswith(".pdf"):
            first.setdefault(doc_hash(p), p)
    with ProcessPoolExecutor(workers) as ex:
        return dict(zip(first, ex.map(corpus_one, first.values(), chunksize=64)))


def _cc():
    sys.path.insert(0, str(CC_ROOT))
    from corpus import config as C
    return C.CC_DATA


def head_one(args):
    cl, base, r = args
    try:
        x = cl.get(f"{base}/{r['f']}", headers={"Range": f"bytes={r['o']}-{r['o'] + min(r['l'], HEAD) - 1}"})
        if x.status_code not in (200, 206):
            return r["url"], ""
        return r["url"], pdf_text(zlib.decompressobj(31).decompress(x.content).split(b"\r\n\r\n", 2)[-1])
    except Exception:
        return r["url"], ""


def heads(cands, out=None, part=20_000, threads=16):
    """Description of every candidate (rows url, f, o, l) in parts of `part` rows; a part on disk is not fetched again."""
    out = out or WORK / "descr"
    out.mkdir(parents=True, exist_ok=True)
    base = _cc()
    C = pd.read_parquet(cands)
    cl = httpx.Client(timeout=30, limits=httpx.Limits(max_connections=threads))
    for i, a in enumerate(range(0, len(C), part)):
        f = out / f"part_{i:04d}.parquet"
        if f.exists():
            continue
        rows = C.iloc[a:a + part].to_dict("records")
        with ThreadPoolExecutor(threads) as ex:
            R = list(ex.map(head_one, [(cl, base, r) for r in rows]))
        pd.DataFrame(R, columns=["url", "text"]).to_parquet(f)
        print(f"descriptions {a + len(rows)}/{len(C)}, with text {sum(bool(t) for _, t in R)} of {len(R)}", flush=True)
