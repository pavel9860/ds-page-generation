"""Text / handwriting boxes of the embedded pages: detection shards (UVTM/Labels/hw_detection) -> manifest.sqlite table hw_det.

  python tools/hw_det.py detect   # embedded pages no shard has: 768 x 1024 render (tools/form_pages.py), yolo26s -> the next d<N> shard
  python tools/hw_det.py merge    # -> table hw_det (doc = emb_out id, page, hash, text, hw, shard); prints the re-render check

Shard d<N>__pages_<NNNN>.npz: ids int32 (n, 2) [doc, page]; offsets int64 (n + 1); per box cls uint8 (0 handwriting, 1 print) and
conf uint8 (round(conf * 255), floor 0.25); failed int32 (k, 2). Boxes of page i: [offsets[i]:offsets[i + 1]]; no coordinates.

A page in several shards takes the highest group: d0..d13 rendered 14,746 pages from the wrong file of a hash, d14 (info/pages_fix)
has them from the manifest's file. CHECK (d16, emb_check: a re-render of 3000 random pages) counts only for pages no other shard
has and is compared with the counts that are used. Needs pymupdf < 1.28 (1.28 aborts with a double free on a broken structure tree)."""
import argparse
import collections
import io
import json
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from dspages.hand.config import PATHS

UV = Path(PATHS.uvtm)
LAYOUTS = Path(PATHS.layouts)
HW = UV / "Labels" / "hw_detection"
EMB = UV / "Labels" / "embedding" / "emb_out"
WEIGHTS = Path("/run/media/p/D/BUISNESS/handwriting_detection/models/yolo26s-ds3k3250v3.pt")
CONF, CHECK, STEP = 0.25, 16, 10                     # conf floor; check group; shard number step of a new group
HEX = re.compile(r"(?<![0-9a-f])[0-9a-f]{12}(?![0-9a-f])")


def keys(ids):
    return np.asarray(ids, np.int64).reshape(-1, 2) @ np.array([100000, 1])


def emb_paths():
    with open(EMB / "manifest.jsonl") as fh:
        return {r["id"]: r["path"] for r in map(json.loads, fh)}


def shards():
    return sorted(HW.glob("d*__pages_*.npz"), key=lambda f: int(f.name[1:f.name.index("__")]))


def counts(f):
    """(key, grp, shard, hw, text) per page of one shard."""
    z = np.load(f)
    o, c = z["offsets"], np.r_[0, np.cumsum(z["cls"] == 0)]
    hw = c[o[1:]] - c[o[:-1]]
    return pd.DataFrame(dict(key=keys(z["ids"]), grp=int(f.name[1:f.name.index("__")]), shard=f.name, hw=hw, text=np.diff(o) - hw))


def cmd_detect(_):
    import form_pages as F
    from PIL import Image
    from ultralytics import YOLO
    done = np.concatenate([keys(np.load(f)["ids"]) for f in shards()])
    miss = np.setdiff1d(keys(np.load(EMB / "ids.npy")), done)
    print(len(miss), "embedded pages without detection", flush=True)
    if not len(miss):
        return
    path, by = emb_paths(), collections.defaultdict(list)
    for k in miss.tolist():
        by[str(LAYOUTS / path[k // 100000])].append((k // 100000, k % 100000))
    jpg = {(d, p): j for d, p, j in F.run_jobs(list(by.items()))}
    model = YOLO(str(WEIGHTS))
    ids, off, cls, conf, failed = [], [0], [], [], []
    for k in miss.tolist():
        key = (k // 100000, k % 100000)
        if key not in jpg:
            failed.append(key)
            continue
        b = model.predict(np.asarray(Image.open(io.BytesIO(jpg[key])).convert("RGB"))[:, :, ::-1], imgsz=1024, conf=CONF, verbose=False)[0].boxes
        ids.append(key)
        cls += b.cls.int().tolist()
        conf += (b.conf * 255).round().int().tolist()
        off.append(len(cls))
    g = int(shards()[-1].name[1:shards()[-1].name.index("__")]) + 1
    out = HW / f"d{g}__pages_{(g + 15) * STEP:04d}.npz"
    np.savez(out, ids=np.array(ids, np.int32).reshape(-1, 2), offsets=np.array(off, np.int64), cls=np.array(cls, np.uint8),
             conf=np.array(conf, np.uint8), failed=np.array(failed, np.int32).reshape(-1, 2))
    idx = json.loads((HW / "index.json").read_text())
    idx["pages"][out.name] = len(ids)
    (HW / "index.json").write_text(json.dumps(idx, indent=1))
    print(out.name, len(ids), "pages", len(cls), "boxes", len(failed), "failed")


def cmd_merge(_):
    D = pd.concat(map(counts, shards()))
    chk = D[D.grp == CHECK]
    D = D.assign(rank=np.where(D.grp == CHECK, -1, D.grp)).sort_values("rank").drop_duplicates("key", keep="last")
    c = chk.merge(D[D.grp != CHECK], on="key", suffixes=("_chk", ""))
    print(f"check: {len(c)} pages rendered twice, hw equal {(c.hw_chk == c.hw).mean():.4f}, text equal {(c.text_chk == c.text).mean():.4f},"
          f" text |diff| <= 2 {((c.text_chk - c.text).abs() <= 2).mean():.4f}")
    E = keys(np.load(EMB / "ids.npy"))
    D = D[D.key.isin(E)].assign(doc=lambda d: d.key // 100000, page=lambda d: d.key % 100000)
    path = emb_paths()
    D["hash"] = [m.group(0) if (m := HEX.search(Path(path.get(d, "")).name)) else None for d in D.doc]
    with sqlite3.connect(LAYOUTS / "manifest.sqlite") as con:
        con.execute("drop table if exists hw_det")
        con.execute("create table hw_det (doc integer, page integer, hash text, text integer, hw integer, shard text, primary key (doc, page))")
        con.executemany("insert into hw_det values (?,?,?,?,?,?)", D[["doc", "page", "hash", "text", "hw", "shard"]].astype(object).itertuples(False, None))
    print(f"hw_det: {len(D)} of {len(E)} embedded pages; pages by group", D.grp.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["detect", "merge"])
    a = ap.parse_args()
    {"detect": cmd_detect, "merge": cmd_merge}[a.cmd](a)
