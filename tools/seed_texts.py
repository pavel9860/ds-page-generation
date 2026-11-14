"""Texts and forms of the seeds of tools/seed_pick.py (ds100k/seeds/{hw,filled,notes}.csv): where src is set the layout is in another language and its text is
translated into the language of the row (Google Translate over HTTP, hand.rare.translated: structure and boxes kept; forms: hwdet.translate.render sets the
translated lines back into the page). Google answers 429 above about one request a second: few workers, and a start only when it answers.
  texts  -> SEEDS/{hw,notes}.jsonl (rows done are kept)        forms  -> OUT/forms_work (hwdet.filledforms.prepare)
tools/seed_run.sh runs both, the page generators and the embedding.

  PYTHONPATH=src:../handwriting_detection/src python tools/seed_texts.py texts|forms OUT [--workers 1]"""
import argparse
from collections import Counter
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import seed_pick as P  # noqa: E402
from dspages.hand import corpus, plan, rare  # noqa: E402
from dspages.hand.structure import to_items  # noqa: E402
from dspages.hand.util import walk_text  # noqa: E402
from hwsynth.scripts import script  # noqa: E402


HWDET = Path("/run/media/p/D/BUISNESS/handwriting_detection")
sys.path.insert(0, str(HWDET / "scripts" / "translation"))
CHUNK = 4800                                                                          # characters translated at once (Google web input: 5000, 5200 measured cut)
GOOGLE = {"ku": "ckb", "zh": "zh-CN", "mni": "mni-Mtei"}                                                # Google: ku is Kurmanji in Latin letters (the corpus's ku is Sorani); Chinese is zh-CN / zh-TW, a bare zh falls back to the profile's last language


def tagged_lines(texts, tl, sl="auto"):
    """hwdet.gtranslate.lines as one tagged text per chunk of at most CHUNK characters (whole(): translate.googleapis.com, Google Translate as the
    web page); None for a line whose tag Google dropped."""
    tl, sl = GOOGLE.get(tl, tl), GOOGLE.get(sl, sl)
    out, lines = [], [" ".join(t.split()) or "." for t in texts]
    i = 0
    while i < len(lines):                                                              # chunks of at most CHUNK characters, each one call
        j, size = i + 1, len(lines[i])
        while j < len(lines) and size + len(lines[j]) + 1 <= CHUNK:
            size += len(lines[j]) + 1
            j += 1
        got = whole(lines[i:j], tl, sl)
        if not got:                                                                    # no answer (Google refuses): the run stops, nothing is gated
            raise RuntimeError(f"Google Translate gave no answer ({sl} -> {tl}, {j - i} lines): blocked or down")
        out += got
        i = j
    return out


KEY = re.compile(r'<\s*p\s*id\s*=\s*"?(\d+)"?\s*>')                                    # a block's opening tag as Google gives it back (<pid="9"> too)


def whole(lines, tl, sl):
    """The lines translated as one text, each in its tag <p id="n">..</p> (no line-by-line call) -> keyed(answer, n). None: no answer."""
    o = _google("\n".join(f'<p id="{i}">{x}</p>' for i, x in enumerate(lines)), tl, sl)
    return keyed(o, len(lines)) if o else None


def keyed(answer, n):
    """The n block texts of a tagged answer: the text after tag i up to the next tag, its </p> dropped; None for a block whose tag Google dropped."""
    parts = KEY.split(answer)
    got = {}
    for k, seg in zip(parts[1::2], parts[2::2]):
        if int(k) < n and int(k) not in got:
            got[int(k)] = " ".join(re.sub(r"<\s*/\s*p\s*>", " ", seg).split())
    return [got.get(i) for i in range(n)]


import hwdet.gtranslate as GT  # noqa: E402

_google = GT.google
GT.lines = tagged_lines                                                               # rare.translated and form() translate whole tagged texts


def text(job):
    """(path, page, lang, src) -> the passage of a seed layout, translated into lang when src is set; None when it cannot be read or translated."""
    path, page, lang, src = job
    if not src:
        t = P.passage((path, page, lang, 4))
        if not t:
            print(lang, path, page, "dropped: native passage fails the text filter", flush=True)
        return t
    blocks, _ = rare.page_blocks(str(corpus.resolve(path)), int(page), src)
    tr = rare.translated(blocks, lang, src) if blocks else None
    if not tr:
        print(lang, path, page, "dropped:", "no text" if not blocks else "no translation", flush=True)
        return None
    items = to_items(tr)
    t = dict(path=path, kind="article", doc=path, lang=lang, page=int(page), items=items, text="\n".join(walk_text(items)),
             verified=True)                                                         # Google's text is no broken extraction: only its script is checked
    if not plan.usable(t, repeats=True):
        print(lang, path, page, "dropped: not usable (script / garbled)", Counter(script(c) for c in t["text"] if c.isalpha()).most_common(3), repr(t["text"][:120]), flush=True)
        return None
    return t


def form(job):
    """(path, page, lang, src, out) -> (pdf, page index, lang) of the form to fill: the page itself, or its lines translated into lang and set again."""
    path, page, lang, src, out = job
    path = str(corpus.resolve(path))
    if not src:
        return path, int(page), lang
    from dspages.layout import legacy
    pdf = Path(out) / f"{lang}_{Path(path).stem}_{page}.pdf"
    if pdf.exists():
        if not Path(f"{pdf}.paths.json").exists():                                      # made before the outlines were marked
            legacy.mark_no_text(str(pdf))
        return str(pdf), 0, lang
    import pymupdf
    from hwdet import translate as TR
    from hwsynth.writer import _index
    t0 = time.time()
    read = Path(out) / f"src_{Path(path).stem}_{page}_{src}.json"                    # the source page read once for all its target languages
    if not read.exists():
        with pymupdf.open(path) as doc:
            got = dict(lines=TR.lines(doc[int(page)], src),                           # the page as the one reader reads it: legacy fonts decoded, outline text read
                       drawn=[list(l["bbox"]) for l in legacy.path_lines(doc[int(page)])])   # text drawn as glyph outlines: its drawings go too
        read.with_suffix(".tmp").write_text(json.dumps(got, ensure_ascii=False))
        read.with_suffix(".tmp").replace(read)
    got = json.loads(read.read_text())
    ls, drawn = got["lines"], [tuple(b) for b in got["drawn"]]
    tr = GT.lines([x["text"] for x in ls], lang, src) if ls else []
    if len(tr) != len(ls) or sum(t is None for t in tr) > rare.LOST * len(tr):          # garbled: more than LOST of the lines' tags lost
        print("garbled (tags lost):", lang, path, page, flush=True)
        return None
    print(lang, len(ls), "lines translated in", round(time.time() - t0), "s", flush=True)
    tr = [t or x["text"] for x, t in zip(ls, tr)]
    TR.render(path, int(page), [(x["bbox"], x["size"], t) for x, t in zip(ls, tr)], TR.fonts_for(" ".join(tr)[:400], _index(), 1)[0], lang, str(pdf),
              drawn=drawn)
    legacy.mark_no_text(str(pdf))                                                       # the outlines left (pictures the reading has as no text)
    return str(pdf), 0, lang


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["texts", "forms"])
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = lambda n: list(pd.read_csv(P.OUT / f"{n}.csv").fillna("").itertuples(index=False))
    with ProcessPoolExecutor(a.workers) as ex:
        if a.what == "texts":
            for n in ("hw", "notes"):
                f = P.OUT / f"{n}.jsonl"
                done = {(t["path"], t["page"], t["lang"]): t for t in map(json.loads, f.read_text().splitlines())} if f.exists() else {}
                S = rows(n)
                todo = [r for r in S if (r.path, int(r.page), r.lang) not in done]
                for i, (r, t) in enumerate(zip(todo, ex.map(text, [(r.path, r.page, r.lang, r.src) for r in todo])), 1):
                    if t:
                        done[(r.path, int(r.page), r.lang)] = t
                        f.write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in done.values()))
                    print(n, i, "of", len(todo), "done", len(done), flush=True)
        else:
            S = sorted(rows("filled"), key=lambda r: r.lang)                              # a worker takes a run of rows of the same languages
            F = [f for f in ex.map(form, [(r.path, r.page, r.lang, r.src, str(out)) for r in S], chunksize=-(-len(S) // a.workers)) if f]
            from hwdet import filledforms as FF
            print("filled", len(F), "of", len(S), "forms;", FF.prepare(F, str(out), workers=2), "with fields", flush=True)


if __name__ == "__main__":
    main()
