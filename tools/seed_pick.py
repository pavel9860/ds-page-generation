"""Seed layouts for the Lens search of hw pages, filled forms and list-like notes in non-Latin scripts: digital-born pages whose layout, written by hand / filled
in, fills the gaps and extends the border of the hw / filled class in the 16-d head space; every non-Latin language gets the same quota.

Landing: the fitted transition of tools/shift_test.py (fit, landings): where a page lands in the head space after it is written by hand / filled in, as it
is (native edit) or translated into another language first (translated edit), from its own place, its source and target language.
Targets: the gap points of the ds_targets run (n200k) + the sparsest 25% of the existing pages of the class (border).
Round by round every language takes the layout whose predicted landing is nearest to a target not yet covered (a pick covers the targets within COVER of
its landing): a layout of the language itself with the native edit, a layout of any other language translated into it. A layout counts only when the hand
pipeline reads a passage from it that passes the text filter (decoded legacy fonts, no gibberish); a layout is used once, by one class.

  SEED_SET=v2 python tools/seed_pick.py [--hw 200 --filled 200 --notes 20]  ->  ds100k/seeds/<SEED_SET>/{hw,filled,notes}.csv (path, page, h, type,
  lang, dist (predicted), src: the language of a layout that is translated, else empty, edit)
Texts and forms of the seeds (translated where src is set): tools/seed_texts.py, then the existing generators (see there)"""

import argparse
import os
import pickle
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pymupdf
import torch

sys.path.insert(0, str(Path(__file__).parent))
import ds_targets as T  # noqa: E402
import shift_test as ST  # noqa: E402

from dspages.hand import corpus, plan, texts  # noqa: E402
from dspages.hand.langs import EXPECT, NO_TEXT_LAYER  # noqa: E402
from dspages.ingest.lang import garbled  # noqa: E402
from dspages.layout import legacy  # noqa: E402

OUT = T.D.OUT / "seeds" / os.environ.get("SEED_SET", "v2")  # the seed set (SEED_SET= : the first set, ds100k/seeds)
RUN = T.OUT / "n200k"
COVER, PLAIN = 0.15, 0.8  # targets a pick covers; share of Unicode letters of a plain page
LANGS = sorted(l for l, s in EXPECT.items() if s != "Latin")


def targets(Z, R, cls):
    """Gap points of the class + the sparsest 25% of its existing pages (kNN distance), as vectors."""
    X = np.load(RUN / "out" / "targets.npz")["Xt"].astype(np.float32)
    g = pd.read_csv(RUN / "targets.csv")
    gap = X[np.flatnonzero(g.hw.to_numpy() == cls)]
    E = torch.from_numpy(Z[np.flatnonzero(T.hw_class(R.hw, R.text) == cls)][:6000])
    d = torch.cdist(E, E).topk(6, largest=False).values[:, -1].numpy()
    return np.concatenate([gap, E[d >= np.quantile(d, 0.75)].numpy()])


def passage(job):
    """(path, page, lang, follow): the passage the hand pipeline reads from the page (its first page is 1) and the next follow - 1 pages, None when it fails the
    text filter."""
    path, page, lang, follow = job
    t = texts.passage(
        (
            dict(
                path=str(corpus.resolve(path)),
                kind="article",
                doc=path,
                lang=lang,
                page=int(page) + 1,
                follow=follow,
                native=True,
                layout=True,
            ),
            1,
        )
    )
    return t if t and not garbled(t["text"]) and plan.usable(t, repeats=True) else None


def gate(job):
    """(path, page, lang): the page reads cleanly. A page set in Unicode fonts of the script (most letters of the script, no stray glyphs) is judged on its
    plain text layer (a few ms); a page that needs the font decoding goes through the whole reader (passage)."""
    path, page, lang = job
    script = legacy.LANG_SCRIPT[lang]
    with pymupdf.open(corpus.resolve(path)) as doc:
        pg = doc[int(page)]
        chars = [chr(u) for t in pg.get_texttrace() for u, g, _, _ in t["chars"]]
        text = pg.get_text()
    letters = [c for c in chars if c.isalpha()]
    if letters and legacy.native_share(letters, script) >= PLAIN and not any(legacy.stray(c, script) for c in chars):
        return (
            len(text) >= 120
            and not garbled(text)
            and plan.usable(dict(text=text, lang=lang, kind="article"), repeats=True)
        )
    return passage((path, page, lang, 1)) is not None


def pick(Z, R, cand, tg, n, edits, used):
    """n layouts (rows of R among cand) for the targets tg: in rounds, every language takes the readable layout whose predicted landing (the fitted
    transition, tools/shift_test.py landings) is nearest to an uncovered target: a layout of the language itself with the native edit, a layout of any
    other language with the translated edit (edits = (native, translated)). A layout is read only when a language wants it, all wanted ones of a wave at once."""
    lang, docs, Z0 = R.lang.iloc[cand].to_numpy(), R.h.iloc[cand].to_numpy(), Z[cand]
    model = pickle.loads((ST.OUT / "transition.pkl").read_bytes())
    tg = torch.from_numpy(tg)
    alive = torch.ones(len(tg), dtype=torch.bool)
    quota, per, out, read = -(-n // len(LANGS)), dict.fromkeys(LANGS, 0), [], {}
    order = random.Random(0).sample(LANGS, len(LANGS))
    with ProcessPoolExecutor(6) as ex:
        while len(out) < n and any(v < quota for v in per.values()):
            todo = [l for l in sorted(order, key=lambda x: per[x]) if per[l] < quota][: n - len(out)]
            land, dist, rank = {}, {}, {}
            for l in todo:
                mine = lang == l
                land[l] = np.where(
                    mine[:, None], ST.landings(Z0, edits[0], lang, l, model), ST.landings(Z0, edits[1], lang, l, model)
                )
                dist[l] = torch.cdist(torch.from_numpy(land[l]), tg[alive]).min(1).values.numpy()
                key = dist[l].copy()
                key[np.isin(docs, list(used))] = np.inf
                rank[l] = [c for c in np.argsort(key)[:400] if np.isfinite(key[c])]
            ptr, got = dict.fromkeys(todo, 0), []
            pending = list(todo)
            while pending:
                want = sorted({rank[l][ptr[l]] for l in pending if ptr[l] < len(rank[l])} - read.keys())
                read.update(
                    zip(
                        want,
                        ex.map(
                            gate, [(R.path.iloc[cand[c]], R.page.iloc[cand[c]], lang[c]) for c in want], chunksize=1
                        ),
                    )
                )
                nxt = []
                for l in pending:
                    if ptr[l] >= len(rank[l]):
                        per[l] = quota  # no readable layout is left for the language
                        continue
                    c = rank[l][ptr[l]]
                    if not read[c] or docs[c] in used:
                        ptr[l] += 1
                        nxt.append(l)
                        continue
                    used.add(docs[c])
                    mine = lang[c] == l
                    got.append(land[l][c])
                    out.append((cand[c], l, float(dist[l][c]), "" if mine else lang[c], edits[0] if mine else edits[1]))
                    per[l] += 1
                pending = nxt
            if got:
                alive &= torch.cdist(tg, torch.from_numpy(np.stack(got))).min(1).values >= COVER
                if not alive.any():
                    alive[:] = True
    return pd.DataFrame(out[:n], columns=["row", "lang", "dist", "src", "edit"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hw", type=int, default=200)
    ap.add_argument("--filled", type=int, default=200)
    ap.add_argument("--notes", type=int, default=20)
    a = ap.parse_args()
    R = T.pages()
    Z = np.load(T.OUT / "aux" / "Z.npz")["Z"].astype(np.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    base = (
        ~R.scan.to_numpy()
        & (R.text.to_numpy() >= 3)
        & R.lang.isin(EXPECT).to_numpy()
        & ~R.lang.isin(NO_TEXT_LAYER).to_numpy()
    )
    used, tg = (
        set(),
        {c: targets(Z, R, c) for c in ("hw", "filled")},
    )  # a layout serves one class once; the targets a class covers are done
    for name, cls, types, n, edits in (
        ("hw", "hw", ("text", "books_pdf"), a.hw, ("hand", "tr_hand")),
        ("filled", "filled", ("forms", "forms_clean"), a.filled, ("filled", "tr_filled")),
        ("notes", "hw", ("text", "books_pdf", "magazines"), a.notes, ("hand", "tr_hand")),
    ):
        cand = np.flatnonzero(
            base & R.type.isin(types).to_numpy() & (R.text.to_numpy() <= (20 if name == "notes" else 10**6))
        )
        S = pick(Z, R, cand, tg[cls], n, edits, used)
        land = np.stack([ST.landings(Z[[r]], e, [R.lang.iloc[r]], l)[0] for r, e, l in zip(S.row, S.edit, S.lang)])
        tg[cls] = tg[cls][torch.cdist(torch.from_numpy(tg[cls]), torch.from_numpy(land)).min(1).values.numpy() >= COVER]
        S = pd.concat(
            [R.iloc[S.row].reset_index(drop=True)[["path", "page", "h", "type"]], S[["lang", "dist", "src", "edit"]]],
            axis=1,
        )
        S.to_csv(OUT / f"{name}.csv", index=False)
        print(
            name,
            len(S),
            S.lang.nunique(),
            "languages, translated",
            int((S.src.fillna("") != "").sum()),
            "predicted dist median",
            round(S.dist.median(), 3),
            "within COVER",
            round((S.dist < COVER).mean(), 3),
            flush=True,
        )


if __name__ == "__main__":
    main()
