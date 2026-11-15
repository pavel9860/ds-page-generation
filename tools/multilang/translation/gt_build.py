"""Forms translated by Google Translate (browser; translation/google/gt_<lang>.json {lang, items: [{key, tr}]}, typed source lines in translation/source/src_typed.json) set again in the target language: each translated line replaces the printed line it came from (typed lines are matched to the lines of the source page in order), lines that were not translated are erased, the page is rendered with a font of the language (translate.render) into translation/pages/<lang>/<key>.pdf, and its fields are prepared (filledforms.prep_one) into handwriting/forms as new batches in_NNNN.json (answers by the agents, then forms3k_status.py). python gt_build.py [--lang xx]
python gt_build.py [--lang xx]"""

import difflib
import glob
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import pymupdf

from dspages.multilang.paths import FILLING as W
from dspages.multilang.paths import TRANSLATION

OUT = TRANSLATION / "pages"
DL = TRANSLATION / "google"
norm = lambda s: re.sub(r"[\W_]+", "", s.lower())


def align(typed, orig):
    """index of the original line of each typed line (in order), None when none is near enough."""
    res, j = [], 0
    for t in typed:
        best, bi = 0.0, None
        for i in range(j, len(orig)):
            r = (
                difflib.SequenceMatcher(None, norm(t), norm(orig[i])).ratio()
                if norm(t) and norm(orig[i])
                else float(t.strip() == orig[i].strip())
            )
            if r > best:
                best, bi = r, i
            if r > 0.97:
                break
        if best >= 0.6:
            res.append(bi)
            j = bi + 1
        else:
            res.append(None)
    return res


def build(args):
    lang, item, meta, typed, font, tl = args
    from dspages.multilang import translate as T

    fid = item["key"]
    m = meta.get(fid)
    if m is None:
        return lang, fid, None
    out = OUT / lang / f"{fid}.pdf"
    if out.exists():
        return lang, fid, str(out)
    with pymupdf.open(m["path"]) as doc:
        orig = T.lines(doc[int(m["page"])])
    ix = align(typed, [o["text"] for o in orig])
    lines, used = [], set()
    for tr, i in zip(item["tr"], ix):
        if i is not None and tr is not None and i not in used:
            used.add(i)
            lines.append((orig[i]["bbox"], orig[i]["size"], tr))
    lines += [(o["bbox"], o["size"], "") for k, o in enumerate(orig) if k not in used]
    out.parent.mkdir(parents=True, exist_ok=True)
    T.render(m["path"], int(m["page"]), lines, font, lang, str(out))
    return lang, fid, str(out)


def main():
    from hwsynth.writer import _index

    from dspages.multilang import translate as T

    only = sys.argv[sys.argv.index("--lang") + 1] if "--lang" in sys.argv else None
    meta = json.loads((W / "meta.json").read_text())
    typed = {s[0]: s[2] for s in json.loads((TRANSLATION / "source/src_typed.json").read_text())}
    index = _index()
    jobs = []
    for f in sorted(glob.glob(str(DL / "gt_*.json"))):
        d = json.loads(Path(f).read_text())
        if "items" not in d or (only and d["lang"] != only):
            continue
        lang = d["lang"]
        sample = " ".join(x for it in d["items"] for x in (it["tr"] or []) if x)[:400]
        try:
            font = T.fonts_for(sample, index, 1)[0]
        except IndexError:
            miss = [
                c
                for c in sorted(set(sample))
                if not index.covers(next(iter(index.cmap)), c) and not any(index.covers(f, c) for f in index.cmap)
            ]
            print(lang, "no font for", "".join(miss)[:40], flush=True)
            continue
        for it in d["items"]:
            if it["tr"] and sum(v is None for v in it["tr"]) <= 0.2 * len(it["tr"]):
                jobs.append((lang, it, meta, [t.strip() or "." for t in typed[it["key"]]], font, d["tl"]))
    done = []
    with ProcessPoolExecutor(6, mp_context=get_context("spawn")) as ex:
        for lang, fid, pdf in ex.map(build, jobs, chunksize=2):
            if pdf:
                done.append((pdf, 0, lang))
    from dspages.multilang import filledforms as F

    with ProcessPoolExecutor(6, mp_context=get_context("spawn")) as ex:
        res = [r for r in ex.map(F.prep_one, [(p, g, l, str(W), 0) for p, g, l in done]) if r]
    first = len(list(W.glob("in_*.json")))
    items = [it for it, m in res if m["fid"] not in meta]
    for it, m in res:
        m["lang_cand"] = dict(path=m["path"], page=m["page"])
        meta[m["fid"]] = m
    (W / "meta.json").write_text(json.dumps(meta))
    for n in range(0, len(items), 4):
        (W / f"in_{first + n // 4:04d}.json").write_text(json.dumps(items[n : n + 4], ensure_ascii=False))
    print(len(done), "forms rendered,", len(items), "prepared in", -(-len(items) // 4), "batches from in_%04d" % first)


if __name__ == "__main__":
    main()
