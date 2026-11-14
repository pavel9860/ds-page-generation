"""Trace of the cleaning of the corpus (warped / distorted pages and documents) in one file, and the moves of newly judged pages.

  python tools/cleaned.py move   100k_warp_check   pages of the check list that were deleted from its folder (warped) -> archive/rejected_distorted
                                 (jpg back from the Trash; a document that is warped whole: moved; a partly warped one: its warped pages as
                                 <name>_part.pdf, the original stays in place so the page numbers of the embeddings stay valid)
  python tools/cleaned.py trace                   all logs -> Layouts/archive/info/cleaned_trace.tsv (source, verdict, doc, page, path, when)
Verdicts: warped (page / doc out), clean (checked and kept), moved (file moved by a cleaning step: flyers / tables clean, distorted).
"""
import glob
import json
import re
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import diverse_ds as D  # noqa: E402
from dspages.hand.config import PATHS  # noqa: E402

LAY = Path(PATHS.layouts)
ARC = LAY / "archive"
REJ = ARC / "rejected_distorted"
TRACE = ARC / "info" / "cleaned_trace.tsv"
TRASH = Path("/run/media/p/D/.Trash-1000")
LOGS = Path("/run/media/p/D/BUISNESS/corpus_collecting/logs")


def cmd_move(a):
    ix = pd.read_csv(ARC / a.name / "index.csv")
    have = set(os.listdir(ARC / a.name))
    ix["warped"] = ~ix.file.isin(have)
    REJ.mkdir(exist_ok=True)
    rows, hashes = [], D.hash_index()
    D._init(hashes)
    for r in ix[ix.warped].itertuples():                                  # 1) the page images back from the Trash
        stem = Path(r.path).stem
        dst = REJ / f"{stem}_p{r.page}.jpg"
        src = TRASH / "files" / r.file
        if src.exists() and not dst.exists():
            shutil.move(src, dst)
            (TRASH / "info" / (r.file + ".trashinfo")).unlink(missing_ok=True)
        elif not dst.exists():                                            # deleted for good: rendered again
            D.render((r.file, r.path, r.page, dst))
    import pymupdf
    pymupdf.TOOLS.mupdf_display_errors(False)
    for path, g in ix[ix.warped].groupby("path"):                          # 2) the documents
        p = D.resolve(path)
        if p is None:
            rows.append((path, "", "missing", "", ""))
            continue
        rel = os.path.relpath(p, LAY)
        pages = sorted(set(int(x) for x in g.page))
        if str(p).lower().endswith(".pdf"):
            doc = pymupdf.open(p)
            n = len(doc)
            pages = [x for x in pages if x < n]
            if len(pages) >= n:
                kind = "whole"
            else:
                kind = "part"
                part = pymupdf.open()
                for x in pages:
                    part.insert_pdf(doc, from_page=x, to_page=x)
                part.save(REJ / f"{p.stem}_part.pdf")
            doc.close()
        else:
            kind, n = "whole", 1
        if kind == "whole":
            shutil.move(p, REJ / p.name)
        rows.append((rel, "archive/rejected_distorted/" + (p.name if kind == "whole" else p.stem + "_part.pdf"), kind, ",".join(map(str, pages)), n))
    log = LAY / f"distorted_moved_log_{pd.Timestamp.now():%Y%m%d_%H%M%S}.tsv"
    pd.DataFrame(rows, columns=["moved_from", "moved_to", "kind", "page_idx", "doc_pages"]).to_csv(log, sep="\t", index=False)
    print(len(ix), "checked,", int(ix.warped.sum()), "warped pages in", len(rows), "documents:", pd.Series([r[2] for r in rows]).value_counts().to_dict(), "->", log)


def cmd_trace(a):
    t = []

    def add(src, verdict, doc="", page="", path="", when="", extra=""):
        t.append((src, verdict, doc, page, path, when, extra))

    for l in open(LOGS / "warped_manifest.jsonl"):                        # judged pages trashed as warped (judge_images runs)
        r = json.loads(l)
        add("warped_manifest:" + r.get("run", ""), "warped", r["doc_hash"], r["page_index"], r["origin_path"], r["trashed_on"][:10])
    for f, v in (("warped_docs_dropped.jsonl", "warped"), ("distorted_docs_dropped.jsonl", "warped")):
        for l in open(LOGS / f):
            r = json.loads(l)
            add(f[:-6], v + "_doc", Path(r["path"]).stem.split("_")[0], "", r["path"], r["dropped"])
    for f in sorted(glob.glob(str(LOGS / "judge_*.tsv"))):                # verdicts of every judge_images run
        for l in open(f):
            c = l.rstrip("\n").split("\t")
            if len(c) > 1 and c[1]:
                m = re.match(r"(?:\d{5}_)?([0-9a-f]{12})(?:_[a-z]+)?_p(\d+)\.", c[0])
                add(Path(f).stem, {"deleted": "warped", "bad": "warped", "ok": "clean"}.get(c[1], c[1]), m.group(1) if m else c[0].split("_")[0], m.group(2) if m else "", c[0])
    for f in sorted(glob.glob(str(LAY / "*moved_log*.tsv")) + glob.glob(str(ARC / "info" / "superseded" / "*moved_log*.tsv"))):
        d = pd.read_csv(f, sep="\t")
        when = Path(f).stem[-15:-7]
        for r in d.itertuples():
            add(Path(f).stem[:-16], "moved", Path(r.moved_from).stem.split("_")[0], getattr(r, "page_idx", ""), f"{r.moved_from} -> {r.moved_to}", when, getattr(r, "kind", ""))
    for f in glob.glob(str(LAY / "scan_flyers_*")):
        for l in open(f):
            add(Path(f).stem, "moved", "", "", l.strip(), "20261021")
    for j in glob.glob(str(ARC / "*_warp_check")) + glob.glob(str(ARC / "info" / "warp_check_*")):                        # pages of the warp check lists: kept = clean
        ix = pd.read_csv(Path(j) / "index.csv")
        if "verdict" not in ix:
            jpgs = set(f for f in os.listdir(j) if f.endswith(".jpg"))
            if jpgs:                                                      # judging in progress: deleted jpgs are the warped ones
                ix["verdict"] = ["clean" if f in jpgs else "warped" for f in ix.file]
            else:                                                         # jpgs cleared after the move: the warped pages are in rejected_distorted
                ix["verdict"] = ["warped" if (REJ / f"{Path(p_).stem}_p{g}.jpg").exists() else "clean" for p_, g in zip(ix.path, ix.page)]
                ix.to_csv(Path(j) / "index.csv", index=False)
        live = any(f.endswith(".jpg") for f in os.listdir(j))                # still being judged: only the deleted ones count
        for r in ix.itertuples():
            if not (live and r.verdict == "clean"):
                add(Path(j).name, r.verdict, r.h, r.page, r.path, "20261026")
    df = pd.DataFrame(t, columns=["source", "verdict", "doc", "page", "path", "when", "extra"])
    df.to_csv(TRACE, sep="\t", index=False)
    print(len(df), "rows ->", TRACE, df.groupby(["source", "verdict"]).size().to_string(), sep="\n")


if __name__ == "__main__":
    {"move": cmd_move, "trace": cmd_trace}[sys.argv[1]](argparse_ns := type("A", (), {"name": sys.argv[2] if len(sys.argv) > 2 else ""})())
