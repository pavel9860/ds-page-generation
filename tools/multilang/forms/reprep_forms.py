"""Prepares again the forms of langds/forms3k with the corrected field finder (pale rules, no box over printed text, labels of empty table cells from the column header): old forms_work -> forms_work_old, new forms_work with meta.json, img/ and in_NNNN.json (PER_FILE forms each) for the agents.
python reprep_forms3k.py"""
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

from dspages.multilang.paths import FILLING as NEW

OLD = NEW.parent / "forms_old"
PER_FILE = 4


def prep(m):
    from dspages.multilang import filledforms as F
    return m, F.prep_one((m["path"], int(m["page"]), m["lang"], str(NEW), 0))


def main():
    if not OLD.exists():
        shutil.copytree(NEW, OLD)
        shutil.copy(NEW / "INSTRUCTIONS.md", OLD / "INSTRUCTIONS.md") if (NEW / "INSTRUCTIONS.md").exists() else None
        for f in list(NEW.glob("in_*.json")) + list(NEW.glob("out_*.json")) + [NEW / "meta.json"]:
            f.unlink()
    old = json.loads((OLD / "meta.json").read_text())
    meta, items = {}, []
    with ProcessPoolExecutor(6, mp_context=get_context("spawn")) as ex:
        for m, res in ex.map(prep, list(old.values()), chunksize=4):
            if res is None:
                continue
            item, nm = res
            nm["lang_cand"] = m.get("lang_cand")
            meta[nm["fid"]] = nm
            items.append(item)
    (NEW / "meta.json").write_text(json.dumps(meta))
    for n in range(0, len(items), PER_FILE):
        (NEW / f"in_{n // PER_FILE:04d}.json").write_text(json.dumps(items[n:n + PER_FILE], ensure_ascii=False))
    print(len(old), "old forms,", len(meta), "prepared,", -(-len(items) // PER_FILE), "batches")


if __name__ == "__main__":
    main()
