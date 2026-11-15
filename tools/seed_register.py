"""The generated seed pages of a run (tools/seed_run.sh: hw/ = hand pages of hw and notes texts, filled/ = filled forms) into the corpus: each page as
Layouts/generated/<sha12>_<lang>.jpg (768 x 1024, JPEG 90, as the embedded pages), registered by dspages.ingest (ids above the largest) and merged into
emb_out, hw_det and scan_pct as ingest run RUN (no dspages.ingest place: the files stay in generated/).
  boxes   from the layout, in the detector's units (one box per line): hand pages dspages.multilang.handpages.line_boxes of the written words, no print;
          filled forms dspages.multilang.filledforms.fill_window (hand boxes, print lines of the form)
  fill / pic  dspages.fillpic detector;  embedding  the vectors of tools/shift_test.py embed (run/emb_*.npz), pages without one embedded here
Layouts/generated/generated.csv: file, cls (hw / notes / filled), lang, seed, page (the seed layout of seed_pick), src (language of a translated seed), hw, text, fill, pic.

  PYTHONPATH=src:../handwriting_detection/src python tools/seed_register.py RUN_DIR RUN"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import seed_pick as P  # noqa: E402
import shift_test as ST  # noqa: E402
from dspages.fillpic.detector import Detector  # noqa: E402
from dspages.fillpic.paths import LAYOUTS, MODEL  # noqa: E402
from dspages.ingest import merge, registry  # noqa: E402
from dspages.ingest.paths import WORK  # noqa: E402

GEN = LAYOUTS / "generated"


def hand_pages(run):
    """(image, cls, lang, layout, page, hw, text) of the hand pages (layout = the seed document); notes are the pages of the notes texts."""
    from dspages.multilang.handpages import line_boxes
    notes = {(t["doc"], t["page"], t["lang"]) for t in map(json.loads, open(P.OUT / "notes.jsonl"))}
    for m in map(json.loads, open(run / "hw" / "meta_0.jsonl")):
        f = run / "hw" / "pages" / f"{m['i']:05d}.jpg"
        if f.exists():
            cls = "notes" if (m["doc"], m["page"], m["lang"]) in notes else "hw"
            yield f, cls, m["lang"], m["doc"], m["page"], len(line_boxes([w[1] for w in m["words"]])) if m["words"] else 0, 0


def form_pages(run):
    """(image, 'filled', lang, layout, page, hw, text) of the filled forms; layout = the form page filled (its translation when the seed was translated)."""
    from dspages.multilang.filledforms import fill_window
    meta = json.loads((run / "work" / "forms_work" / "meta.json").read_text())
    answers = json.loads((run / "filled" / "answers.json").read_text())
    for fid, ans in answers.items():
        m = meta[fid]
        _, hand, prints, _, _ = fill_window(fid, m, ans, 0, None)
        yield run / "filled" / f"{m['lang']}_{fid}.jpg", "filled", m["lang"], m["path"], m["page"], len(hand), len(prints)


def seeds():
    """{(seed file stem, lang): (path, page, src)} of the seed csvs (a layout is a seed once: seed_pick's used set); src = language of a translated layout."""
    return {(Path(r.path).stem, r.lang): (r.path, int(r.page), r.src) for n in ("hw", "notes", "filled")
            for r in pd.read_csv(P.OUT / f"{n}.csv").fillna("").itertuples()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("run")
    a = ap.parse_args()
    run = Path(a.run_dir)
    have = {str(n): v for z in map(np.load, sorted(run.glob("emb_*.npz"))) for n, v in zip(z["names"], z["V"])}
    src = seeds()
    rows, jp, V = [], [], []
    pages = [*hand_pages(run), *(form_pages(run) if (run / "filled").exists() else ())]           # a run of hand pages only (a page made again)
    gen = GEN
    gen.mkdir(exist_ok=True)
    for f, cls, lang, layout, page, hw, text in pages:
        j = ST.jpeg(cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB))
        out = gen / f"{hashlib.sha256(j).hexdigest()[:12]}_{lang}.jpg"
        out.write_bytes(j)
        stem = Path(layout).stem
        stem = stem.split("_", 1)[1].rsplit("_", 1)[0] if stem.startswith(f"{lang}_") else stem      # a translated form: <lang>_<seed stem>_<page>.pdf
        seed, spage, sl = src[(stem, lang)]
        rows.append(dict(file=out.name, cls=cls, lang=lang, seed=seed, page=spage, src=sl, hw=hw, text=text))
        jp.append(j)
        V.append(have.get(str(f)))
    miss = [i for i, v in enumerate(V) if v is None]
    if miss:
        for i, v in zip(miss, ST.embed([jp[i] for i in miss])):
            V[i] = v
    gray = [cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_GRAYSCALE) for j in jp]
    res = Detector.load(MODEL).predict(gray)
    R = pd.DataFrame(rows).assign(fill=[r.fill for r in res], pic=[r.pic_mean for r in res])
    old = gen / "generated.csv"
    pd.concat([pd.read_csv(old), R] if old.exists() else [R]).drop_duplicates("file", keep="last").to_csv(old, index=False)
    print(len(R), "pages ->", gen, R.groupby("cls").size().to_dict(), f"{len(miss)} embedded here", flush=True)

    print("registered", registry.register([GEN], acquired=False), flush=True)
    c = registry.db()
    doc = {k: d for k, d in c.execute("select key, doc from docs where origin = 'generated'")}
    keys = R.file.str[:12]
    c.executemany("update docs set lang = ?, lang_src = 'generated', lang_p = 1, picked = 1 where key = ?", list(zip(R.lang, keys)))
    c.commit()
    ids = np.array([[doc[k], 0] for k in keys], np.int32)
    o = WORK / "out" / a.run
    (o / "hw").mkdir(parents=True, exist_ok=True)
    np.save(o / "embeddings.npy", np.stack(V).astype(np.float16))
    np.save(o / "ids.npy", ids)
    cls = np.concatenate([np.r_[np.zeros(h, np.uint8), np.ones(t, np.uint8)] for h, t in zip(R.hw, R.text)])
    np.savez(o / "hw" / "pages_layout.npz", ids=ids, offsets=np.r_[0, np.cumsum(R.hw + R.text)].astype(np.int64), cls=cls,
             conf=np.full(len(cls), 255, np.uint8), failed=np.zeros((0, 2), np.int32))
    np.savez(o / "scan_pct_0.npz", ids=ids, fill=R.fill.to_numpy(np.float32), pic=R.pic.to_numpy(np.float32),
             win=np.array([r.window for r in res], np.float32))
    print("merged", merge.merge(a.run), "; lang_res / src_urls", merge.finish(a.run), flush=True)


if __name__ == "__main__":
    main()
