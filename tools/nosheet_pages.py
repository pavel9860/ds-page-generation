"""Plain page renders for the fonts that have no table and no glyph sheet (no embedded font program to draw glyphs from): up to PAGES pages per font, from the corpus documents that
use it, as tests/legacy_pages/<font>_<document>_p<page>.png. Nothing else.
python tools/nosheet_pages.py [family ...]"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from dspages.layout import legacy as L, legacy_evidence as E, legacy_tables as T

OUT = os.path.join(os.path.dirname(E.LAYOUTS), "tests", "legacy_pages")
PAGES, DPI = 3, 110


def without_sheet():
    """{family: script} of the fonts with no table (learned families and audit) and no glyph sheet."""
    cov = json.load(open(os.path.join(T.TABLES, "COVERAGE.json"), encoding="utf-8"))
    audit = json.load(open(os.path.join(T.TABLES, "AUDIT.json"), encoding="utf-8"))
    known = json.load(open(os.path.join(T.TABLES, "UNICODE_OK.json"), encoding="utf-8"))["fonts"]
    want = {r["family"]: r["script"] for r in cov if r["table"] is None}
    want.update({f: v[0] for f, v in audit.items() if not v[0].startswith("CJK") and f not in want and f not in known and T.route(f) is None})
    return {f: s for f, s in want.items() if not os.path.exists(os.path.join(E.READS, f, "glyphs.json"))}


def main(args):
    want = {f: s for f, s in without_sheet().items() if f in args} if args else without_sheet()
    corpus = E.Corpus(want, 30)
    os.makedirs(OUT, exist_ok=True)
    for fam in want:
        if not corpus.docs[fam]:
            corpus.docs[fam] = corpus.scan(fam, want[fam])
        n = 0
        for d in corpus.docs[fam]:
            pdf = corpus.open(d)
            for pn, pg in enumerate(list(pdf)[:6] if pdf else []):
                if n < PAGES and any(L.family(tr["font"]) == fam for tr in pg.get_texttrace()):
                    pg.get_pixmap(dpi=DPI).save(os.path.join(OUT, f"{fam}_{d}_p{pn}.png"))
                    n += 1
        print(fam, n, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
