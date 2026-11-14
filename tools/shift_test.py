"""How far an edit moves a page in the 1536-d page space. Each page is rendered as the embedded pages were (768 x 1024, JPEG 90:
tools/form_pages.py); images are embedded locally (corpus_collecting pagetype/embed_kaggle.embed_pages, CPU fp32), unit rows as the pool.
Data: ds100k/shift (OUT).

  embed    OUT.npz DIR..: the images (jpg / png) and the first pages of the pdfs under the directories, embedded as above -> names, V (1536-d), Z (16-d head space)
  pairs    N documents balanced over the families of look (FAMILY), TL_PER layouts each (a third forms, the rest text pages translated by block
           into languages of other families) as images with the factors of each edit: orig, reflow (seed, margin), hand (writer, pen, paper, format, density) / filled,
           tr_print (the translation set in print: before writing), tr_hand / tr_filled (after; the same writer) -> pairs/pages_0000.npz (shard for
           tools/kaggle_embed.py) + pairs/pairs.csv
  fit      the pairs embedded (pairs/emb_*: embeddings.npy, ids.npy of each Kaggle run): ridge model of the shift z1 - z0 in the 16-d head space on the edit
           kind, kind x z0, source / target language (the script pair overfits: left out), writer, pen, paper, density, reflow margin, filled fields; 10-fold CV by layout
           against 'stays put' and 'mean shift per kind', the error increase of each factor group left out, errors per edit and script pair
           -> shift/transition.pkl (predict) + transition.csv


python tools/shift_test.py pairs [--n 400]  |  fit  |  embed OUT.npz DIR.."""
import argparse
import io
import json
import os
import sys

from pathlib import Path

import collections

import numpy as np
import pandas as pd

import diverse_ds as DD
import ds100k as D

sys.path.insert(0, "/run/media/p/D/BUISNESS/corpus_collecting/pagetype")
OUT = D.OUT / "shift"
W_, H_, Q90 = 768, 1024, 90


def jpeg(rgb):
    import cv2
    from PIL import Image
    if rgb.shape[:2] != (H_, W_):
        rgb = cv2.resize(rgb, (W_, H_), interpolation=cv2.INTER_AREA if rgb.shape[0] * rgb.shape[1] > W_ * H_ else cv2.INTER_CUBIC)
    b = io.BytesIO()
    Image.fromarray(rgb).save(b, "JPEG", quality=Q90)
    return b.getvalue()


def original(path, page):
    import pymupdf
    with pymupdf.open(path) as doc:
        pg = doc[page]
        pm = pg.get_pixmap(matrix=pymupdf.Matrix(W_ / pg.rect.width, H_ / pg.rect.height), colorspace=pymupdf.csRGB, alpha=False)
        return np.frombuffer(pm.samples, np.uint8).reshape(pm.h, pm.w, 3)


def embed(jpegs):
    import torch
    import embed_kaggle as EK
    torch.set_num_threads(int(os.environ.get("THREADS", 6)))
    model = EK.load_model("cpu")
    out = []
    with torch.no_grad():
        for a in range(0, len(jpegs), 4):
            raw = [torch.frombuffer(bytearray(j), dtype=torch.uint8) for j in jpegs[a:a + 4]]
            out.append(EK.embed_pages(model, raw, "cpu")[0])
    return DD.unit_rows(np.concatenate(out).astype(np.float32))


def head(V):
    import torch
    import torch.nn as nn
    import ds_head as H
    net = nn.Sequential(nn.Linear(1536, 512), nn.GELU(), nn.Linear(512, H.DIM))
    net.load_state_dict(torch.load(H.OUT / "head.pt", map_location="cpu"))
    with torch.no_grad():
        return torch.nn.functional.normalize(net(torch.from_numpy(V).float()), dim=1).numpy()


def cmd_embed(a):
    import cv2
    files = sorted(str(p) for d in a.dirs for p in Path(d).rglob("*") if p.suffix.lower() in (".jpg", ".png", ".pdf"))
    jp, names = [], []
    for f in files:
        try:
            jp.append(jpeg(original(f, 0) if f.endswith(".pdf") else cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)))
            names.append(f)
        except Exception as e:
            print("skip", f, repr(e)[:60], flush=True)
    V = embed(jp)
    np.savez(a.out, names=np.array(names), V=V, Z=head(V))
    print(len(names), "pages embedded ->", a.out)


PAIRS = Path(os.environ.get("PAIRS_DIR", OUT / "pairs"))                     # PAIRS_DIR: a sample build of its own
MARGINS = (0.03, 0.1)                                                          # reflow margin range
TL_PER = 5                                                                     # target languages per new layout (a row each: its own translation and writing)
FORMS = 1 / 3                                                                  # share of forms among the layouts (those with a translated pdf already: no Google)
LATIN = ["en", "es", "fr", "de", "pl", "tr", "vi"]                                # the Latin target languages (Latin: one script of the targets)
FAMILY = {"Latin": ("Latin",), "Cyrillic": ("Cyrillic",), "Greek": ("Greek",), "Han": ("Han",), "Arabic": ("Arabic",), "Hangul": ("Hangul",),
          "Hebrew": ("Hebrew",), "Tibetan": ("Tibetan",), "Telugu-Kannada": ("Telugu", "Kannada"), "Armenian-Ethiopic": ("Armenian", "Ethiopic"),
          "Georgian-Myanmar": ("Georgian", "Myanmar"), "Thai-Lao-Khmer-Gujarati": ("Thai", "Lao", "Khmer", "Gujarati"),
          "Tamil-Malayalam-Oriya-Sinhala": ("Tamil", "Malayalam", "Oriya", "Sinhala"), "Devanagari-Bengali-Gurmukhi-Meetei": ("Devanagari", "Bengali", "Gurmukhi", "Meetei_Mayek")}
# the families of look: the visual clusters of Meyer & Roman 2026 (A Quantitative Visual Taxonomy of Worldwide Writing Systems: 34 Ward clusters
# of 125 scripts, zenodo 24011343 data/info/scripts_clusters.xlsx), Latin, Cyrillic and Greek (one cluster there) a family each. Each family one
# share, its scripts one share of it each; a layout's languages of other families
FAMILY_OF = {s: f for f, ss in FAMILY.items() for s in ss}


def script(lang):
    from dspages.hand.langs import EXPECT
    return EXPECT.get(lang, "Latin")


def layouts(n, used, L0, workers, dropped):
    """n new documents, one page each (none of a document in used), TL_PER layouts each. Each next document is of the family (FAMILY) the layouts L0
    and the picks so far have fewest layouts of, in it the script of fewest (a script one share however many languages it has), in it the language
    of fewest (each language a small share: diversity), a form while the
    script's forms are under the FORMS share (forms / forms_clean pdf pages: their translation is made by translate_forms; the rest born-digital
    text pages: text / books / magazines); of its pages the one of largest variance x' Ai x on the start point (1, z0) given the layouts so far
    (greedy D-optimal: the pages whose shifts the z0 terms of the fit learn the most from), passing the gate (clean() in the workers: _gate). A script with no page left takes a
    form L0 or the dropped layouts have translated into it (original -> lang1 -> lang2: its translated pdf the source). Targets by targets(), writer factors by balanced(),
    the reflow of each copy at its own margin and font family. Every candidate left out is counted in the log."""
    import ds_targets as T
    import seed_pick as P
    from dspages.fillpic.paths import LAYOUTS
    from dspages.hand.langs import NO_TEXT_LAYER
    R = T.pages()
    Z = np.load(T.OUT / "aux" / "Z.npz")["Z"].astype(np.float32)                     # the rows of T.pages()
    kinds = {t: "form" for t in ("forms", "forms_clean")} | {t: "text" for t in ("text", "books_pdf", "magazines")}
    R = R.assign(full=R.path.map(lambda p: str(LAYOUTS / p)), script=R.lang.map(script), kind=R.type.map(kinds))
    A = np.c_[np.ones(int(R.full.isin(used).sum())), Z[R.index[R.full.isin(used)]]]
    Ai = np.linalg.inv(A.T @ A + np.eye(A.shape[1]))
    pool = languages()
    print("languages without a handwriting font (not written):", sorted(set(P.LANGS) - set(writable(P.LANGS))), flush=True)
    base = ~R.scan & R.lang.isin(pool) & ~R.lang.isin(NO_TEXT_LAYER) & R.path.str.lower().str.endswith(".pdf") & ~R.full.isin(used) & R.kind.notna()
    print("candidates left out: scan", int(R.scan.sum()), "| language not in the set or no text layer", int((~R.lang.isin(P.LANGS) | R.lang.isin(NO_TEXT_LAYER)).sum()),
          "| not pdf", int((~R.path.str.lower().str.endswith(".pdf")).sum()), "| document used already", int(R.full.isin(used).sum()), flush=True)
    C = R[base]
    Xc, cs, ck, cl = np.c_[np.ones(len(C)), Z[C.index]], C.script.to_numpy(), C.kind.to_numpy(), C.lang.to_numpy()
    free = np.ones(len(C), bool)
    F = pd.concat([L0, dropped])
    F = F[F.kind == "form"].drop_duplicates("tr")
    chain = [dict(kind="form", path=t, page=0, lang=tl) for t, tl in zip(F.tr, F.tl) if isinstance(t, str) and t not in used and Path(t).exists()]
    rows = L0[["kind", "lang", "tl"]]
    have, forms = collections.Counter(rows.lang.map(script)), collections.Counter(rows[rows.kind == "form"].lang.map(script))
    scripts = sorted({script(l) for l in pool})
    why, taken, X = collections.Counter(), set(), []
    while len(X) < n * TL_PER:
        open_ = [s for s in scripts if (free & (cs == s)).any() or any(script(c["lang"]) == s for c in chain)]
        if not open_:
            print("no page left in any script:", len(X) // TL_PER, "of", n, "documents", flush=True)
            break
        fam = collections.Counter({f: sum(have[x] for x in FAMILY[f]) for f in FAMILY})
        s = min(open_, key=lambda s: (fam[FAMILY_OF[s]], have[s], s))
        lg = collections.Counter(rows.lang)
        m = free & (cs == s)
        if m.any():
            g = min(set(cl[m]), key=lambda l: (lg[l], l))                                   # its language of fewest layouts: each a small share
            m &= cl == g
            kind = "form" if (m & (ck == "form")).any() and (forms[s] < FORMS * have[s] or not (m & (ck == "text")).any()) else "text"
            m &= ck == kind
            j = int(np.flatnonzero(m)[np.einsum("ij,jk,ik->i", Xc[m], Ai, Xc[m]).argmax()])
            free[j] = False
            if C.h.iat[j] in taken:                                                      # one page per document: the first of it picked
                why["document taken"] += 1
                continue
            o = dict(kind=kind, path=C.full.iat[j], page=int(C.page.iat[j]), lang=C.lang.iat[j])
        else:
            o = min((c for c in chain if script(c["lang"]) == s), key=lambda c: lg[c["lang"]])
            chain.remove(o)
        ok, how = dict(pooled(_gate, {0: (o["path"], o["page"], o["lang"], o["kind"])}, workers))[0]   # read in a worker: the main process keeps no reader caches
        if not ok:
            why[how] += 1
            gate_out({(o["path"], o["page"])}, how)                                      # failed: never read again
            continue
        if m.any():
            taken.add(C.h.iat[j])
            u = Ai @ Xc[j]
            Ai -= np.outer(u, u) / (1 + Xc[j] @ u)
        gate_out({(o["path"], o["page"])}, "text layer")                                  # passed clean() here: regate does not read them again
        k0 = len(L0) + len(X)
        new = [dict(o, tl=tl, margin=round(MARGINS[0] + (MARGINS[1] - MARGINS[0]) * k / (TL_PER - 1), 4), font=k0 + k,
                    tr=str(PAIRS / "forms_tr" / f"{tl}_{Path(o['path']).stem}_{o['page']}.pdf") if o["kind"] == "form" else "")
               for k, tl in enumerate(targets(s, rows, pool))]
        X += new
        rows = pd.concat([rows, pd.DataFrame(new)[["kind", "lang", "tl"]]], ignore_index=True)
        have[s] += len(new)
        forms[s] += len(new) * (o["kind"] == "form")
    print(len(X) // TL_PER, "documents picked; passed over:", dict(why), flush=True)
    X = pd.DataFrame(X).reset_index(drop=True)
    return pd.concat([X, balanced(len(X), seed=len(L0))], axis=1)


def languages():
    """The languages of the layouts: of the scripts of FAMILY, with a handwriting font and a text layer."""
    import seed_pick as P
    from dspages.hand.langs import NO_TEXT_LAYER
    return [l for l in writable(P.LANGS) + writable(LATIN) if l not in NO_TEXT_LAYER and script(l) in FAMILY_OF]


def writable(langs):
    """The languages whose script has a realistic handwriting font (hwdet.pagegen.handfonts.Restricted: the fonts text pages and forms are written in)."""
    from hwsynth.fonts import FontIndex
    from hwdet.pagegen.handfonts import Restricted
    from dspages.hand.langs import EXPECT
    F = Restricted(FontIndex())
    return [l for l in langs if F.candidates(EXPECT.get(l, "Latin"))]


def clean(path, page, lang, why, kind):
    """True for a page the transition fit can use: at least the language's least text for its kind (hand.langs.min_chars; a short text tells
    nothing more), its text layer read clean (page_text) and, for a form, a field to fill (prepped; a text page needs none: its
    handwriting is notes). why counts the others by reason."""
    from dspages.hand.langs import min_chars
    try:
        bl, how = page_text(path, page, lang)
    except Exception as e:
        bl, how = None, f"unreadable {type(e).__name__}"
    if bl and sum(len(b[1]) for b in bl) < min_chars(lang, kind):
        how = "too little text"
    if how == "text layer" and kind == "form" and not prepped(path, page, lang)[0]["boxes"]:
        how = "no fields to fill"
    if how != "text layer":
        why[how] += 1
        print("passed over:", how, path, page, flush=True)
    return how == "text layer"


def _cached(f, make):
    """The json file f, made by make() once."""
    if f.exists():
        return json.loads(f.read_text())
    v = json.loads(json.dumps(make(), ensure_ascii=False))                      # made or read: the same values
    _write(f, v)
    return v


def _write(f, v):
    """v as json in f, whole or not at all (a crash leaves no file)."""
    f.parent.mkdir(parents=True, exist_ok=True)
    f.with_suffix(".tmp").write_text(json.dumps(v, ensure_ascii=False))
    f.with_suffix(".tmp").replace(f)


def page_text(path, page, lang):
    """(blocks, how) of hand.rare.page_blocks, read once per (page, language): PAIRS/text/ (the gate and the translation share it)."""
    from dspages.hand import rare
    bl, how = _cached(PAIRS / "text" / f"{Path(path).stem}_{int(page)}_{lang}.json", lambda: rare.page_blocks(path, int(page), lang))
    return (None if bl is None else [tuple(b) for b in bl]), how                    # the blocks as the reader gives them (tuples)


def _prepfile(path, page, lang):
    return PAIRS / "forms_prep" / f"{Path(path).stem}_{int(page)}_{lang}.json"


def prepped(path, page, lang):
    """(batch item, meta) of hwdet.filledforms.prep_one (the fields of a form page and what each holds), made once per form pdf: PAIRS/forms_prep/
    (the gate and the filling share it; made again when the pdf is newer: a translation made again)."""
    from hwdet import filledforms as FF
    f = _prepfile(path, page, lang)
    if f.exists() and f.stat().st_mtime < Path(path).stat().st_mtime:
        f.unlink()
    return tuple(_cached(f, lambda: FF.prep_one((str(path), int(page), lang, str(PAIRS / "forms_prep"), 0))))


def balanced(n, seed=0):
    """Writer factors of n hand pages (persona, style, pen, paper, fmt, dense): each level of each factor equally often (plain paper as often as all
    the ruled kinds together), the factors in independent
    random orders (near orthogonal): the D-optimal main-effects design for the writer / pen / paper / density terms of the fit."""
    from dspages.hand import labelled as FL
    from dspages.hand.paper import FORMATS, KINDS
    rng = np.random.default_rng(seed)
    ruled = [k for k in KINDS if k not in ("blank", "dotted", "margin")]                          # dotted (no rules), margin (lined + margin rule, as lined at times): left out
    lv = dict(persona=FL.PERSONS, style=["casual", "accurate", "educated", "professional"], pen=FL.PENS, paper=["blank"] * len(ruled) + ruled,   # plain paper half the pages
              fmt=list(FORMATS), dense=[False, True])
    return pd.DataFrame({f: rng.permutation(np.resize(np.array(v, object), n)) for f, v in lv.items()})


def targets(src, L, pool):
    """TL_PER target languages of a page in script src, each of another family than src's (FAMILY): the families the layouts L have fewest of from
    src's family, then fewest of in all, in each the script L has fewest of, in it the language L has fewest of."""
    f, s, t = FAMILY_OF.get, L.lang.map(script), L.tl.map(script)
    cell, tot = collections.Counter(t[s.map(f) == f(src)].map(f)), collections.Counter(t.map(f))
    ts, lang = collections.Counter(t), collections.Counter(L.tl)
    fams = sorted((x for x in FAMILY if x != f(src) and any(f(script(l)) == x for l in pool)), key=lambda x: (cell[x], tot[x], x))[:TL_PER]
    pick = lambda x: min((l for l in pool if f(script(l)) == x), key=lambda l: (ts[script(l)], lang[l], l))
    return [pick(x) for x in fams]


def translate_forms(L, workers):
    """The forms of L whose translated pdf (tr) is not made yet: their lines Google-translated and set again (tools/seed_texts.py form), workers at once;
    a form whose translation fails the tag gate has no translated pdf (garbled() gates it, another form replaces it)."""
    import seed_texts
    F = L[(L.kind == "form") & ~L.tr.fillna("").map(lambda t: bool(t) and Path(f"{t}.paths.json").exists())]   # not made, or its outlines not marked
    if not len(F):
        return
    (PAIRS / "forms_tr").mkdir(parents=True, exist_ok=True)
    for i, got in pooled(seed_texts.form, {i: (r.path, r.page, r.tl, r.lang, str(PAIRS / "forms_tr")) for i, r in F.iterrows()}, workers):
        print("form", i, "translated" if got else "NOT translated", flush=True)


GATE = "2026-11-14b"                                                                       # the page gate's code (clean, prepped): a new one gates every layout again
BUILD = "2026-11-12"                                                                      # the image code (reader, writer, fonts, filling): images of another build are made again
EDITS = ("orig", "reflow", "tr_print", "hand", "tr_hand", "filled", "tr_filled")       # the image key of a layout: (i, EDITS.index(edit))
IMG = PAIRS / "img"                                                                       # <i>_<edit>.jpg + .json (factors): made once
MAX_PAGES = 10                                                                            # a page whose text is short takes up to this many pages of its document


def want(r):
    return EDITS[1:5] if r["kind"] == "text" else EDITS[1:3] + EDITS[5:]


def source(r, edit):
    """The file the edit is made from: the form's translation (tr), the text page's blocks and their translation (trfile), else the page's pdf."""
    if edit in ("tr_print", "tr_filled") and r["kind"] == "form":
        return Path(r["tr"])
    if edit in ("hand", "tr_print", "tr_hand"):
        return trfile(r)
    return Path(r["path"])


def have(i, edit, r):
    """The edit's image is made by this BUILD and newer than the file it is made from (a source made again leaves it stale); a filled form with no
    field written is not made (the forms are picked for their fields: none written is a failed fill)."""
    f, src = IMG / f"{i}_{edit}.jpg", source(r, edit)
    if not (f.exists() and (not src.exists() or src.stat().st_mtime < f.stat().st_mtime)):
        return False
    meta = json.loads(f.with_suffix(".json").read_text())
    return meta.get("build") == BUILD and (edit not in ("filled", "tr_filled") or meta.get("fields", 0) > 0)


def keep(i, edit, factors, j):
    (IMG / f"{i}_{edit}.json").write_text(json.dumps(dict(factors, build=BUILD), ensure_ascii=False))
    (IMG / f"{i}_{edit}.jpg").write_bytes(j)


def trfile(r, page=None):
    return PAIRS / "tr" / f"{Path(r['path']).stem}_{r['page'] if page is None else page}_{r['tl']}.json"


def translate_texts(jobs, workers):
    """(i, path, page, lang, tl) of text pages -> their blocks translated into tl (hand.rare.translated, whole tagged texts: tools/seed_texts.py)
    -> PAIRS/tr/<stem>_<page>_<tl>.json; a page Google left mostly untranslated is tried again on the next call."""
    import json
    (PAIRS / "tr").mkdir(parents=True, exist_ok=True)
    def stale(j):                                                          # not made, or not translated
        f = trfile(dict(path=j[1], page=j[2], tl=j[4]))
        return not f.exists() or json.loads(f.read_text())["tr"] is None
    todo = [j for j in jobs if stale(j)]
    for _, (i, ok) in pooled(_translate, {(j[0], j[2]): j for j in todo}, workers):
        print("translated", i, ok, flush=True)


def _translate(job):
    import seed_texts  # noqa: F401  (sets hwdet.gtranslate to whole tagged texts)
    from dspages.hand import rare
    i, path, page, lang, tl = job
    blocks, _ = page_text(path, page, lang)
    tr = rare.translated(blocks, tl, lang) if blocks else None
    _write(trfile(dict(path=path, page=page, tl=tl)), dict(blocks=blocks, tr=tr))
    return i, tr is not None


def _plan(i, r):
    """The hand page plan of text layout i (hand.rare.plan_of with the layout's writer factors): hand and tr_hand are written with it."""
    import random
    from dspages.hand import rare
    fix = {f: (bool(r[f]) if f == "dense" else r[f]) for f in ("persona", "style", "pen", "paper", "fmt", "dense") if pd.notna(r.get(f, np.nan))}
    return rare.plan_of(r["lang"], i, random.Random(i), fix)


def _pages(job):
    """(i, layout) -> (i, pages): the pages of the document from the layout's page on whose source text fills the hand page (hand.page.fills: the
    writer's estimate, nothing written), at most MAX_PAGES and the document's end; a next page whose text cannot be read ends it."""
    import pymupdf
    from dspages.hand import page
    from dspages.hand.structure import to_items
    i, r = job
    plan = _plan(i, r)
    with pymupdf.open(r["path"]) as d:
        last = min(MAX_PAGES, d.page_count - int(r["page"]))
    n, bl = 1, page_text(r["path"], r["page"], r["lang"])[0] or []
    while n < last and not page.fills(plan, to_items(bl)):
        try:
            nxt = page_text(r["path"], int(r["page"]) + n, r["lang"])[0]
        except LookupError:                                                    # a next page whose text drawn as paths is not known: the text ends before it
            break
        if not nxt:
            break
        bl = stack(bl, nxt)
        n += 1
    return i, n


def stack(bl, nxt):
    """The blocks of a next page below the blocks so far (its boxes moved down)."""
    dy = max(b[2][3] for b in bl) + 20 if bl else 0.0
    return list(bl) + [[b[0], b[1], [b[2][0], b[2][1] + dy, b[2][2], b[2][3] + dy], *b[3:]] for b in nxt]


def blocks_of(r):
    """(source blocks, translated blocks, translated blocks of the layout's page itself) of the layout's r.pages pages, each page below the one before."""
    bl, tr, t1 = [], [], None
    for n in range(int(r.get("pages", 1))):
        t = json.loads(trfile(r, int(r["page"]) + n).read_text())
        bl, tr = stack(bl, t["blocks"]), stack(tr, t["tr"]) if t["tr"] and tr is not None else None
        t1 = tr if t1 is None else t1
    return bl, tr, t1


def render(job):
    """(i, layout, fonts) -> (i, [(edit, factors, jpeg)] of the edits not made yet): reflow, tr_print, hand / tr_hand (text: the same writer for both,
    on the layout's pages (_pages), written once); forms: their translation as printed (the filling is fill())."""
    import random
    i, r, fonts = job
    rng = random.Random(i)
    out = []
    seed, margin = rng.randrange(10 ** 6), rng.uniform(*MARGINS)
    if pd.notna(r.get("margin", np.nan)):                                      # the copies of a new layout: margins spread over the range (the reflow factor of the fit)
        margin = float(r["margin"])
    if not have(i, "reflow", r):
        from dspages.layout.template import sheet
        k = r.get("font", np.nan)
        pick = None if pd.isna(k) else (int(k) + 1, int(k))                           # the copies of a new layout: font families in turn (head, body)
        g, info = sheet(r["path"], r["page"], seed, fonts, margin, (W_, H_), pick=pick)
        out.append(("reflow", dict(seed=seed, margin=round(margin, 3), font=info["body"]), jpeg(np.repeat(g[..., None], 3, 2))))
    if r["kind"] == "form":
        if not have(i, "tr_print", r) and Path(r["tr"]).exists():
            out.append(("tr_print", dict(tl=r["tl"]), jpeg(original(r["tr"], 0))))
        return i, out
    from dspages.hand import rare
    blocks, tr, t1 = blocks_of(r)
    if t1 and not have(i, "tr_print", r):
        out.append(("tr_print",) + printed(r, t1))
    for edit, b, lg in (("hand", blocks, r["lang"]), ("tr_hand", tr, r["tl"])):
        if have(i, edit, r) or not b:
            continue
        img, meta, plan, _, why = rare.write(b, lg, i, random.Random(i), _plan(i, r))        # the same writer, pen and paper for both
        if img is None:
            raise RuntimeError(f"layout {i} {edit}: {why}")
        f = {k: plan[k] for k in ("persona", "style", "pen", "paper", "fmt", "dense")} | dict(lines=meta["lines"], tl=lg, pages=int(r.get("pages", 1)))
        out.append((edit, f, jpeg(np.repeat(np.asarray(img)[..., None], 3, 2) if np.asarray(img).ndim == 2 else np.asarray(img))))
    return i, out


def printed(r, t1):
    """(factors, jpeg) of the translated blocks of the layout's page set in print in place of the original text (the page before writing)."""
    from hwdet import translate as TR
    from hwsynth.writer import _index
    font = TR.fonts_for(" ".join(b[1] for b in t1)[:400], _index(), 1)[0]
    pm = TR.render(r["path"], r["page"], [(b[2], b[3], b[1]) for b in t1], font, r["tl"], erase="all", dpi=100)
    pm = np.frombuffer(pm.samples, np.uint8).reshape(pm.h, pm.w, pm.n)[..., :3]
    return dict(tl=r["tl"], font=Path(font).stem), jpeg(np.ascontiguousarray(pm))


def _fill_one(job):
    """(fields, jpeg) of one form filled with its answers (hwdet.filledforms.fill_window)."""
    import cv2
    from hwdet import filledforms as FF
    fid, m, ans = job
    g, hand, _, _, _ = FF.fill_window(fid, m, ans, 0, None)
    return len(hand), jpeg(cv2.cvtColor(g, cv2.COLOR_GRAY2RGB) if g.ndim == 2 else g)


def _prep(job):
    return prepped(*job)


def _labels(job):
    """The form's batch item with its labels in English (formfill.labels_en, one call per form), kept in its prepped file:
    translated once."""
    item, meta = prepped(*job)
    if any("label_en" not in b for b in item["boxes"]):
        import seed_texts  # noqa: F401  (hwdet.gtranslate: whole tagged texts)
        from hwdet import formfill as FO
        FO.labels_en([item])
        _write(_prepfile(*job), [item, meta])
    return item


def fill(L, workers, google):
    """The forms and their translations filled (prepped, formfill.fill_forms, fill_window); layouts of one source form share its
    filled page -> keep (i, filled / tr_filled) -> how many forms are gated 'no fields to fill' (no field found in the form or its translation)."""
    from hwdet import formfill as FO
    F = L[L.kind == "form"]
    if all(have(i, "filled", r) and have(i, "tr_filled", r) for i, r in F.iterrows()):
        return 0
    want = {(r.path, int(r.page)): r.lang for i, r in F.iterrows() if not have(i, "filled", r)} | {(r.tr, 0): r.tl for i, r in F.iterrows() if not have(i, "tr_filled", r)}
    got = dict(pooled(_prep, {k: (*k, lg) for k, lg in want.items()}, workers))           # the fields of each form (the gate's for the originals)
    items, meta = [got[k][0] for k in want], {got[k][1]["fid"]: got[k][1] for k in want}
    empty = {p for (p, _), (it, _) in got.items() if not it["boxes"]}                  # a form (or its translation) where no field is found
    bad = {(r.path, int(r.page)) for _, r in F.iterrows() if r.path in empty or r.tr in empty}
    if bad:
        return gate_out(bad, "no fields to fill")
    got = dict(pooled(_labels, {k: (*k, lg) for k, lg in want.items()}, google))       # the labels in English, once per form
    items = [got[k] for k in want]
    answers, _ = FO.fill_forms(items, meta)
    at = {}
    for i, r in F.iterrows():
        at.setdefault((r.path, r.page), []).append((i, "filled"))
        at.setdefault((r.tr, 0), []).append((i, "tr_filled"))
    for fid, (fields, j) in pooled(_fill_one, {fid: (fid, meta[fid], ans) for fid, ans in answers.items()}, workers):
        m = meta[fid]
        for i, edit in at.get((m["path"], int(m["page"])), []):
            keep(i, edit, dict(tl=m["lang"], fields=fields, form=fid), j)
    return 0


def _job(fn, k, job):
    """fn(job) in a worker: its start is logged (a worker that dies natively leaves its job named in the log); an error comes back as a RuntimeError
    with the job and the worker's traceback (a MuPDF exception cannot be sent to the parent)."""
    import traceback
    print("job start", k, flush=True)
    try:
        return fn(job)
    except Exception as e:
        raise RuntimeError(f"job {k} failed: {e!r}\n{traceback.format_exc()}") from None


_pools = {}                                                                   # one pool per size for the whole run: workers (and their reference
# tables) start once; the workers=google pool makes every Google call (Google limits the rate)


def pooled(fn, jobs, workers):
    """fn over {key: job} in a process pool, (key, result) as they come. Any error stops the run at once and names the job: nothing is skipped (the
    results already given are kept by the caller, a rerun goes on from there). A worker that dies natively (MuPDF) names the jobs it may have run."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from concurrent.futures.process import BrokenProcessPool
    ex = _pools.get(workers) or _pools.setdefault(workers, ProcessPoolExecutor(workers))
    fut = {ex.submit(_job, fn, k, j): k for k, j in jobs.items()}
    done = set()
    try:
        for f in as_completed(fut):
            yield fut[f], f.result()
            done.add(fut[f])
    except BaseException as e:
        for f in fut:
            f.cancel()
        if isinstance(e, BrokenProcessPool):
            _pools.pop(workers).shutdown(cancel_futures=True)
            raise RuntimeError(f"a worker died natively (MuPDF crash); its job is among the last 'job start' lines of the log not done: "
                               f"{[k for k in jobs if k not in done][:workers * 2]}") from None
        raise


def garbled(L):
    """The documents with a translation that failed the tag gate (rare.LOST) (text: tr None; form: no translated pdf) -> gate verdict 'garbled
    layer' in PAIRS/GATED (regate drops them, others of their kind replace them) -> how many."""
    return gate_out({(r.path, int(r.page)) for _, r in L.iterrows() if (r.kind == "form" and not Path(r.tr).exists()) or
                     (r.kind == "text" and blocks_of(r)[1] is None)}, "garbled layer")


def gate_out(bad, how):
    """{(path, page)} -> gate verdict `how` in PAIRS/GATED (a failing one: regate drops them, others of their kind replace them) -> how many."""
    if how != "text layer":
        print(how + ":", len(bad), "documents' pages -> dropped and replaced", flush=True)
    if bad:
        f = PAIRS / GATED
        G = pd.read_parquet(f) if f.exists() else pd.DataFrame(columns=["path", "page", "gate", "how"])
        G = pd.concat([G[[k not in bad for k in zip(G.path, G.page.astype(int))]], pd.DataFrame([dict(path=p, page=n, gate=GATE, how=how) for p, n in bad])])
        G.to_parquet(f)
    return len(bad)


GATED = "gate.parquet"                                                                    # PAIRS/: (path, page, gate, how) of every layout page gated


def _gate(job):
    import collections
    path, page, lang, kind = job
    why = collections.Counter()
    return clean(path, page, lang, why, kind), (next(iter(why)) if why else "text layer")


def regate(L, workers):
    """The documents whose layout page fails the current gate (clean, GATE): every page gated once per GATE, the verdicts kept in PAIRS/GATED."""
    f = PAIRS / GATED
    G = pd.read_parquet(f) if f.exists() else pd.DataFrame(columns=["path", "page", "gate", "how"])
    G = G[G.gate == GATE]
    seen = set(zip(G.path, G.page.astype(int)))
    todo = {(p, int(n)): (p, int(n), lg, k) for p, n, lg, k in zip(L.path, L.page, L.lang, L.kind) if (p, int(n)) not in seen}
    for n, (k, (ok, how)) in enumerate(pooled(_gate, todo, workers)):                  # kept as they come: a stopped run keeps its verdicts
        G = pd.concat([G, pd.DataFrame([dict(path=k[0], page=k[1], gate=GATE, how=how)])])
        G.to_parquet(f)
    print("gate:", len(todo), "pages gated now,", int((G.how != "text layer").sum()), "fail", G[G.how != "text layer"].how.value_counts().to_dict(), flush=True)
    return set(G[G.how != "text layer"].path) & set(L.path)


def _replace(L, used, dropped, workers):
    """The documents whose page fails the gate (regate, the garbled verdicts included) out, replaced by documents of their kind -> (L, used, dropped)."""
    bad = regate(L, workers)
    if bad:
        out = L[L.path.isin(bad)]
        dropped = pd.concat([dropped, out])
        dropped.to_csv(PAIRS / "layouts_dropped.csv")
        L = L[~L.path.isin(bad)]
        nd = out.path.nunique()
        print("gate: drop", len(out), "layouts of", nd, "documents -> replaced", flush=True)
        N = layouts(nd, used, L, workers, dropped)                                              # by the balance of the layouts kept, none of a document used already
        L = pd.concat([L, N.set_axis(range(L.index.max() + 1, L.index.max() + 1 + len(N)))])
        used |= set(N.path)
    L.to_csv(PAIRS / "layouts.csv")                                            # kept at once: a rerun goes on from here
    print(len(L), "layouts", L.kind.value_counts().to_dict(), L.lang.nunique(), "languages", flush=True)
    return L, used, dropped


def cmd_pairs(a):
    """layouts -> translations -> images of every (layout, edit) not made yet -> PAIRS/pages_<n>.npz with only the
    images no embedding has (ids [layout, EDITS index]; the kaggle_embed shard format) + PAIRS/pairs.csv (every image)."""
    from dspages.config import Paths
    from dspages.layout import refont as R
    IMG.mkdir(parents=True, exist_ok=True)
    E = pd.DataFrame(columns=["kind", "path", "page", "lang", "tl", "tr"])
    L = pd.read_csv(PAIRS / "layouts.csv", index_col=0) if (PAIRS / "layouts.csv").exists() else layouts(a.n, set(), E, a.workers, E)
    dropped = pd.read_csv(PAIRS / "layouts_dropped.csv", index_col=0) if (PAIRS / "layouts_dropped.csv").exists() else L.iloc[:0]
    used = set(L.path) | set(dropped.path)                                     # a dropped document is never picked again
    if a.more:                                                                 # more layouts asked, none of a document used already
        N = layouts(a.more, used, L, a.workers, dropped)
        L = pd.concat([L, N.set_axis(range(L.index.max() + 1, L.index.max() + 1 + len(N)))])
        used |= set(N.path)
    fonts = R.font_set(Paths().font_dirs)
    while True:                                                                # a document that fails on the way (garbled translation, a translated
        L, used, dropped = _replace(L, used, dropped, a.workers)               # form without fields): gated, replaced, the stages again (what is made stays)
        translate_forms(L, a.google)
        if "pages" not in L:
            L["pages"] = np.nan
        T = L[L.kind == "text"]
        for _, (i, n) in pooled(_pages, {i: (i, r.to_dict()) for i, r in T.iterrows() if pd.isna(r.pages)}, a.workers):   # each text layout's pages,
            L.loc[i, "pages"] = n                                                                                     # from its source text
        L.to_csv(PAIRS / "layouts.csv")
        T = L[L.kind == "text"]
        translate_texts([(i, r.path, int(r.page) + k, r.lang, r.tl) for i, r in T.iterrows() for k in range(int(r.pages))], a.google)
        if garbled(L):
            continue
        for _, (i, out) in pooled(render, {i: (i, r.to_dict(), fonts) for i, r in L.iterrows() if any(not have(i, e, r) for e in want(r) if e in EDITS[1:5])}, a.workers):
            for edit, f, j in out:
                keep(i, edit, f, j)
        if not fill(L, a.workers, a.google):
            break
    L.to_csv(PAIRS / "layouts.csv")
    rows = [dict(i=int(f.stem.split("_", 1)[0]), edit=f.stem.split("_", 1)[1], factors=f.with_suffix(".json").read_text()) for f in sorted(IMG.glob("*.jpg"))
            if int(f.stem.split("_", 1)[0]) in L.index]                     # a dropped layout's images stay on disk, out of the pairs
    P_ = pd.DataFrame(rows).assign(k=lambda d: d.edit.map(EDITS.index))
    P_.to_csv(PAIRS / "pairs.csv", index=False)
    lack = [(i, e) for i, r in L.iterrows() for e in want(r.to_dict()) if not have(i, e, r)]
    done = {}                                                                  # (layout, edit) -> when it was last embedded or packed
    for f in [*PAIRS.glob("emb_*/ids.npy"), *PAIRS.glob("pages_*.npz")]:     # embedded, or packed in an earlier shard: not again
        for x in (np.load(f) if f.suffix == ".npy" else np.load(f)["ids"]).tolist():
            done[tuple(x)] = max(done.get(tuple(x), 0), f.stat().st_mtime)
    new = P_[[e != "orig" and (IMG / f"{i}_{e}.jpg").stat().st_mtime > done.get((i, k), 0) for i, k, e in zip(P_.i, P_.k, P_.edit)]]    # not embedded,
    # or made again after its embedding; originals: the corpus embedding (stored())
    jp = [(IMG / f"{i}_{e}.jpg").read_bytes() for i, e in zip(new.i, new.edit)]
    n = len(list(PAIRS.glob("pages_*.npz")))
    if jp:
        np.savez(PAIRS / f"pages_{n:04d}.npz", ids=new[["i", "k"]].to_numpy(np.int32), offsets=np.r_[0, np.cumsum([len(j) for j in jp])].astype(np.int64),
                 jpeg=np.frombuffer(b"".join(jp), np.uint8))
    print(len(P_), "images", P_.edit.value_counts().to_dict(), f"| {len(new)} not embedded -> pages_{n:04d}.npz | still missing {len(lack)}: {lack[:40]}", flush=True)


GROUPS = {"kind": ["edit"], "start": ["z0"], "src": ["src", "src_script"], "tl": ["tl", "tl_script"], "writer": ["persona", "style"], "pen": ["pen"],
          "paper": ["paper", "fmt"], "density": ["dense", "lines"], "reflow": ["margin", "font"], "fill": ["fields"]}
# src_script / tl_script: the script of the language (its visual group: languages of one script share most of the shift); font: the reflow's body family


def design(T, vocab=None):
    """Features of edits T (edit, z0 (16), src, tl, pair, writer / reflow / fill factors): one-hots (vocab = their levels), kind x z0, numbers."""
    from dspages.hand.langs import EXPECT
    T = T.assign(pair=[f"{EXPECT.get(a, '?')}>{EXPECT.get(b, '?')}" for a, b in zip(T.src, T.tl)],
                 src_script=[EXPECT.get(a) for a in T.src], tl_script=[EXPECT.get(b) for b in T.tl])
    if "font" not in T:
        T = T.assign(font=np.nan)
    cat = [c for g in GROUPS.values() for c in g if c not in ("z0", "lines", "margin", "fields", "dense")]
    vocab = vocab or {c: sorted(T[c].dropna().astype(str).unique()) for c in cat}
    cols, names = [np.ones((len(T), 1))], [("bias", "")]
    Z0 = np.stack(T.z0.to_numpy())
    for c in (c for c in cat if vocab[c]):                                  # a factor with no level yet (no edit embedded) has no column
        v = T[c].astype(str).to_numpy()
        cols.append(np.stack([(v == x) for x in vocab[c]], 1).astype(float))
        names += [(c, x) for x in vocab[c]]
    for x in vocab["edit"]:                                                  # the shift depends on where the page starts, per kind of edit
        cols.append(Z0 * (T.edit.to_numpy() == x)[:, None])
        names += [("z0", f"{x}:{j}") for j in range(Z0.shape[1])]
    for c in ("dense", "lines", "margin", "fields"):
        cols.append(pd.to_numeric(T[c], errors="coerce").fillna(0).to_numpy(float)[:, None])
        names.append((c, ""))
    return np.concatenate(cols, 1), names, vocab


def ridge(X, Y, lam=1.0):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)


def predict(z0, edit, factors, model=None):
    """Where a page at z0 (16-d head space) lands after an edit ('hand', 'tr_print', 'tr_hand', 'filled', 'tr_filled', 'reflow') with its factors
    (src, tl, persona, style, pen, paper, fmt, dense, lines, margin, fields): the fitted transition (shift/transition.pkl, fit)."""
    import pickle
    m = model or pickle.loads((OUT / "transition.pkl").read_bytes())
    T = pd.DataFrame([dict(factors, edit=edit, z0=np.asarray(z, np.float32)) for z in np.atleast_2d(z0)])
    for c in m["cols"]:
        T[c] = T.get(c, np.nan)
    X, names, _ = design(T, m["vocab"])
    return np.atleast_2d(z0) + X[:, m["keep"]] @ m["W"]


def landings(Z0, edit, src, tl, model=None):
    """Predicted landings (n, 16) of the pages Z0 (n, 16) of source languages src (n,) after `edit` into the target language tl (one for all): the fitted
    transition with the factors not chosen yet (writer, pen, paper, density, fields) at their mean for the edit. Vectorised form of predict."""
    import pickle
    from dspages.hand.langs import EXPECT
    m = model or pickle.loads((OUT / "transition.pkl").read_bytes())
    W, names = m["W"], m["names"]
    x = m["Xbar"][edit].copy()
    for j, (g, v) in enumerate(names):
        if g in ("edit", "src", "tl", "z0", "src_script", "tl_script"):
            x[j] = float(g == "edit" and v == edit or g == "tl" and v == tl or g == "tl_script" and v == EXPECT.get(tl))
    zcols = [j for j, (g, v) in enumerate(names) if g == "z0" and v.startswith(f"{edit}:")]
    scol = {(g, v): j for j, (g, v) in enumerate(names) if g in ("src", "src_script")}
    S = np.stack([sum((W[scol[k]] for k in (("src", l), ("src_script", EXPECT.get(l))) if k in scol), np.zeros(W.shape[1])) for l in src])
    return (Z0 + x @ W + Z0 @ W[zcols] + S).astype(np.float32)


def stored(L):
    """{layout: z0} the 16-d head point of each layout's page as the corpus has it embedded already (emb_out: the page is not rendered and embedded
    again); a page missing there stops the run."""
    import json
    E = np.load(D.EMB / "embeddings.npy", mmap_mode="r")
    I = np.load(D.EMB / "ids.npy")
    row = {(int(d), int(p)): k for k, (d, p) in enumerate(I.tolist())}
    doc = {Path(m["path"]).stem.split("_")[0]: m["id"] for m in map(json.loads, open(D.EMB / "manifest.jsonl"))}   # file hash -> the embedding's document id
    at = {i: row.get((doc.get(Path(p).stem.split("_")[0], -1), int(pg))) for i, p, pg in zip(L.index, L.path, L.page)}
    miss = [i for i, k in at.items() if k is None]
    if miss:
        raise RuntimeError(f"{len(miss)} layouts have no stored embedding: {miss[:20]}")
    ks = [at[i] for i in L.index]
    return dict(zip(L.index, head(DD.unit_rows(np.asarray(E[np.sort(ks)], np.float32)))[np.argsort(np.argsort(ks))]))


def table():
    """The embedded edits: (i, edit, factors, z0 = the layout's original, z1, src, tl, kind), one row per edit (head space, 16-d)."""
    import json
    P_ = pd.read_csv(PAIRS / "pairs.csv")
    L = pd.read_csv(PAIRS / "layouts.csv", index_col=0)
    em = sorted(PAIRS.glob("emb_*"))
    E = np.concatenate([np.load(d / "embeddings.npy").astype(np.float32) for d in em])
    I = np.concatenate([np.load(d / "ids.npy") for d in em])
    Z = head(DD.unit_rows(E))
    at = {(int(i), int(k)): j for j, (i, k) in enumerate(I.tolist())}
    P_["z"] = [Z[at[(i, k)]] if (i, k) in at else None for i, k in zip(P_.i, P_.k)]
    print(f"{P_.z.notna().sum()} of {len(P_)} images embedded", flush=True)
    P_ = P_[P_.z.notna()]
    orig = stored(L)
    T = pd.DataFrame([dict(json.loads(f), i=i, edit=e, z0=orig[i], z1=z, src=L.lang[i], kind=L.kind[i])
                      for i, e, f, z in zip(P_.i, P_.edit, P_.factors, P_.z) if e != "orig" and i in orig])
    T["tl"] = np.where(T.edit.str.startswith("tr_"), T.tl.fillna(T.src), T.src)
    for c in sorted({c for g in GROUPS.values() for c in g} - {"z0", "edit", "pair"}):
        T[c] = T.get(c, np.nan)
    return T


def cmd_fit(a):
    """pairs.csv + the embeddings of the pairs (PAIRS/emb_*: embeddings.npy, ids.npy [layout, EDITS index], one per Kaggle run) -> the transition
    model and its report."""
    import pickle
    T = table()
    cols = sorted({c for g in GROUPS.values() for c in g} - {"z0", "edit", "pair"})
    Y = np.stack(T.z1.to_numpy()) - np.stack(T.z0.to_numpy())
    X, names, vocab = design(T)
    groups = np.array([next((g for g, cs in GROUPS.items() if n in cs), "bias") for n, _ in names])
    fold = T.i.to_numpy() % 10

    def cv(keep):
        P = np.zeros_like(Y)
        for f in range(10):
            tr, te = fold != f, fold == f
            P[te] = X[te][:, keep] @ ridge(X[tr][:, keep], Y[tr], a.lam)
        return np.linalg.norm(Y - P, axis=1)

    full = np.ones(X.shape[1], bool)
    err = cv(full)
    kind_mean = T.groupby("edit").apply(lambda d: pd.Series(list(Y[d.index].mean(0)))).to_dict("index")
    base = np.linalg.norm(Y - np.stack([list(kind_mean[e].values()) for e in T.edit]), axis=1)
    T = T.assign(shift=np.linalg.norm(Y, axis=1), err_fit=err, err_kind_mean=base)
    from dspages.hand.langs import EXPECT
    T["pair"] = [f"{EXPECT.get(a, '?')}>{EXPECT.get(b, '?')}" for a, b in zip(T.src, T.tl)]
    pd.set_option("display.width", 220)
    print(f"16-d head space, {len(T)} edits of {T.i.nunique()} layouts; |shift| = error of 'stays put'; 10-fold CV by layout, ridge lambda {a.lam}")
    print(T.groupby("edit")[["shift", "err_kind_mean", "err_fit"]].median().round(3).assign(n=T.groupby("edit").size()).to_string())
    print("\nerror increase (median, all edits) when a factor group is left out:")
    for g in [g for g in GROUPS if g != "kind"]:
        e = cv(groups != g)
        print(f"  {g:8s} +{np.median(e) - np.median(err):.4f}   ({(groups == g).sum()} features)")
    print("\nper edit and script pair (median):")
    print(T.groupby(["edit", "pair"])[["shift", "err_kind_mean", "err_fit"]].median().round(3).assign(n=T.groupby(["edit", "pair"]).size()).query("n >= 3").to_string())
    W = ridge(X, Y, a.lam)
    Xbar = {e: X[(T.edit == e).to_numpy()].mean(0) for e in T.edit.unique()}               # the factors not chosen yet at selection (writer, pen, paper, ...): their mean
    (OUT / "transition.pkl").write_bytes(pickle.dumps(dict(W=W, keep=full, vocab=vocab, cols=cols, lam=a.lam, names=names, Xbar=Xbar)))
    T.drop(columns=["z0", "z1"]).to_csv(OUT / "transition.csv", index=False)
    print("->", OUT / "transition.pkl")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    pp = sp.add_parser("pairs")
    pp.add_argument("--n", type=int, default=400)
    pp.add_argument("--more", type=int, default=0, help="add this many new documents (not used yet), TL_PER layouts each")
    pp.add_argument("--workers", type=int, default=6)
    pp.add_argument("--google", type=int, default=2)
    e = sp.add_parser("embed")
    e.add_argument("out")
    e.add_argument("dirs", nargs="+")
    ft = sp.add_parser("fit")
    ft.add_argument("--lam", type=float, default=3.0)                              # 10-fold CV best of 0.3 1 3 10 30 100
    a = ap.parse_args()
    {"pairs": cmd_pairs, "embed": cmd_embed, "fit": cmd_fit}[a.cmd](a)
