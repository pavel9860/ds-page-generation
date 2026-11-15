"""Fills prepared forms (dspages.multilang.formfill) and renders them
python scripts/forms/fill_cpu.py SET_DIR OUT_DIR [N] [--random SEED] [--no-render]"""
import argparse
import json
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
from dspages.multilang import filledforms
from dspages.multilang import formfill as FF


def render(job):
    fid, meta, answers, path = job
    cv2.imwrite(path, filledforms.fill_window(fid, meta, answers, 0, None)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("n", nargs="?", type=int, default=100)
    ap.add_argument("--random", type=int)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((src / "meta.json").read_text())
    items = [it for f in sorted(src.glob("in_*.json")) for it in json.loads(f.read_text())]
    items = random.Random(a.random).sample(items, min(a.n, len(items))) if a.random is not None else items[:a.n]
    answers, prov = FF.fill_forms(items, meta, out / "used_lines.json")
    (out / "answers.json").write_text(json.dumps(answers, ensure_ascii=False))
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False))
    print(len(items), "forms,", sum(len(it["boxes"]) for it in items), "fields;", FF.repeats(answers), FF.closeness(items, answers, prov))
    if not a.no_render:
        jobs = [(f, meta[f], ans, str(out / f"{meta[f]['lang']}_{f}.jpg")) for f, ans in answers.items()]
        with ProcessPoolExecutor(a.workers) as ex:
            list(ex.map(render, jobs, chunksize=2))
        print(len(jobs), "forms rendered ->", out)


if __name__ == "__main__":
    main()
