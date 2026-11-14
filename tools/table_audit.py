"""Every legacy font of the corpus, read as the reader reads it, checked by measure: no font left out, no table trusted unchecked.

1. scan:   the documents of the non-Latin languages (manifest plan), first PAGES pages: the fonts whose letters are not of the language's script (legacy.native_share < NATIVE), whose script text breaks its syllable rules (garbled), or that a name routes to a native-coded table (Zawgyi)
           -> AUDIT_DIR/scan.parquet (doc, page, font, script, lang, letters)
2. decode: up to PER_FAMILY documents per font family (all for anonymous names), each font on its page with the most letters, through the reader (texts.page_lines: the table legacy.table_for picks, the shape fit, the repaired text maps), per (doc, font): the decision (legacy table_for key | shape), the decoded text,
           its clusters that break the script's syllable rules (legacy_fit.invalid) -> AUDIT_DIR/decode.parquet
3. report: per table: documents, characters read, invalid share, the decoded sample -> AUDIT_DIR/tables.parquet, worst first; the fonts no table decodes
           (shape fit or nothing) -> tables/AUDIT.json {family: [script, characters, invalid share, document]} (tools/ocr_tables.py builds tables for them)

python tools/table_audit.py scan|decode|report [--limit N] [--workers 6]"""
import argparse
import collections
import json
import os
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from dspages.layout import legacy as L, legacy_fit as F, legacy_tables as T, refont as R  # noqa: E402

LAYOUTS = os.environ.get("DSPAGES_LAYOUTS", "/run/media/p/D/ML_DS/UVTM/Layouts")
AUDIT_DIR = os.path.join(LAYOUTS, "archive", "info", "legacy", "audit")
PAGES = 3
MIN_LETTERS = 20                                                        # a font with fewer letters on the page tells nothing
PER_FAMILY = 20                                                         # documents decoded per font family (every document for an anonymous name)
ANONYMOUS = r"(?i)(CIDFont\+)?F\d+|TT[0-9A-F]+t00|T\d+|C\d+_\d+|[A-Z]{6}\+F\d+"     # names a producer gives any embedded font


def docs():
    """[(doc, path, lang, script)] of the pdf documents whose language has a script of its own (not Latin)."""
    db = sqlite3.connect(f"file:{LAYOUTS}/manifest.sqlite?mode=ro", uri=True)
    path = {h: p for h, p in db.execute("select hash, path from files where ext = '.pdf'")}
    return [(d, path[d], lg, L.LANG_SCRIPT[lg]) for d, lg in db.execute("select doc, lang from plan")
            if d in path and L.LANG_SCRIPT.get(lg) not in (None, "Latin")]


def latin_text(text):
    """Real Latin-script text (English, or Latin words of any language): words read in English, or nearly no codes above ASCII and words of normal shape."""
    return L.word_share(text, "en") >= 0.5


def garbled(text, script, share=0.02, least=3):
    """Text in the script's code points that breaks its syllable rules (legacy_fit.invalid): a font whose text map gives visual order or glyph pieces."""
    bad, n = F.invalid(text, script)
    return bad >= least and bad >= share * n


def scan_one(job):
    import pymupdf
    d, path, lg, script = job
    out = []
    try:
        with pymupdf.open(os.path.join(LAYOUTS, path)) as pdf:
            for no in range(min(PAGES, pdf.page_count)):
                chars = collections.defaultdict(list)
                for tr in pdf[no].get_texttrace():
                    chars[tr["font"]] += [chr(u) for u, g, _, _ in tr["chars"] if g > 0]
                for font, cs in chars.items():
                    letters = sum(c.isalpha() for c in cs)
                    if letters >= MIN_LETTERS and (T.route(font) or ("",))[0] in T.NATIVE_CODED:
                        out.append((d, no, font, script, lg, letters))          # the script's own code points in a legacy order (Zawgyi): known by the font's name
                    elif letters >= MIN_LETTERS and not script.startswith("CJK") and L.native_share(cs, script) >= L.NATIVE and garbled("".join(cs), script):
                        out.append((d, no, font, script, lg, letters))          # the script's code points, broken (a wrong text map: visual order, pieces)
                    elif letters >= MIN_LETTERS and not script.startswith("CJK") and L.native_share(cs, script) < L.NATIVE and not latin_text("".join(cs)):
                        out.append((d, no, font, script, lg, letters))
                    elif letters >= MIN_LETTERS and script.startswith("CJK") and sum(ord(c) > 0x2E80 for c in cs) < 0.1 * letters and sum(0x80 <= ord(c) < 0x100 for c in cs) > 0.3 * letters:
                        out.append((d, no, font, script, lg, letters))          # CJK typed as Latin-1 bytes (a legacy double-byte font)
    except Exception as e:
        print("unreadable:", d, path, repr(e)[:80], flush=True)
    return out


def decode_one(job):
    """(doc, lang, pages, fonts) -> rows of the decode of each legacy font on those pages, as the reader reads them."""
    import pymupdf
    d, path, lg, script, pages, fonts = job
    rows = []
    try:
        with pymupdf.open(os.path.join(LAYOUTS, path)) as pdf:
            text = collections.defaultdict(list)
            try:
                with L.language(lg):                                   # the reader's text decisions (tables, shape fit, text maps): its segments, no layout
                    read = [ln for no in pages for ln in R.extract_segments(pdf[no])]
            except LookupError as e:                                   # text drawn as glyph outlines with no reading yet (legacy.path_lines): a finding, not a crash
                return [dict(doc=d, font=f, family=L.family(f), script=script, lang=lg, how="paths not read", chars=0, clusters=0, invalid=0,
                             sample=str(e)[:300]) for f in fonts]
            for _ in (0,):
                for ln in read:
                    for sp in ln["spans"]:
                        if sp["font"] in fonts:
                            text[sp["font"]].append(sp["text"])
            keys = L._doc_cache(pdf)[1]["table_for"]
            shaped = {f for (f, _sc), v in L._doc_cache(pdf)[1]["tables"].items() if v and v[0] is not None}
            for font in fonts:
                t = " ".join(text[font])
                bad, n = F.invalid(t, script)
                key = (keys.get(font) or (None,))[0] if isinstance(keys.get(font), tuple) else keys.get(font)
                rows.append(dict(doc=d, font=font, family=L.family(font), script=script, lang=lg, how=key if key else "shape" if font in shaped else "none",
                                 chars=len(t), clusters=n, invalid=bad, sample=t[:300]))
    except Exception as e:
        print("failed:", d, path, repr(e)[:120], flush=True)
        raise
    return rows


def pooled(fn, jobs, workers):
    t0, out = time.time(), []
    with ProcessPoolExecutor(workers) as ex:
        for k, r in enumerate(ex.map(fn, jobs, chunksize=8)):
            out += r
            if (k + 1) % 2000 == 0:
                print(f"{k + 1}/{len(jobs)} {int(time.time() - t0)} s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=("scan", "decode", "report"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(AUDIT_DIR, exist_ok=True)
    if a.step == "scan":
        jobs = docs()[:a.limit or None]
        t0 = time.time()
        S = pd.DataFrame(pooled(scan_one, jobs, a.workers), columns=["doc", "page", "font", "script", "lang", "letters"])
        S.to_parquet(os.path.join(AUDIT_DIR, "scan.parquet" if not a.limit else "scan_try.parquet"))
        print(f"{len(jobs)} documents, {S.doc.nunique()} with legacy fonts, {len(S)} (page, font), {int(time.time() - t0)} s")
    elif a.step == "decode":
        S = pd.read_parquet(os.path.join(AUDIT_DIR, "scan.parquet"))
        S["fam"] = S.font.map(L.family)
        anon = S.font.str.fullmatch(ANONYMOUS)                         # one name for unrelated fonts: every document
        keep = set(S[~anon].drop_duplicates(["fam", "doc"]).groupby("fam").head(PER_FAMILY).doc) | set(S[anon].doc)
        S = S[S.doc.isin(keep)].sort_values("letters", ascending=False).drop_duplicates(["doc", "font"])     # each font on its page with the most letters
        path = {d: p for d, p, _, _ in docs()}
        jobs = [(d, path[d], g.lang.iat[0], g.script.iat[0], sorted(set(g.page)), set(g.font)) for d, g in S.groupby("doc")][:a.limit or None]
        D = pd.DataFrame(pooled(decode_one, jobs, a.workers))
        D.to_parquet(os.path.join(AUDIT_DIR, "decode.parquet" if not a.limit else "decode_try.parquet"))
        print(f"{len(jobs)} documents, {len(D)} (doc, font)")
    else:
        D = pd.read_parquet(os.path.join(AUDIT_DIR, "decode.parquet"))
        g = D.groupby(["how", "script"]).agg(docs=("doc", "nunique"), fonts=("family", "nunique"), clusters=("clusters", "sum"), invalid=("invalid", "sum"),
                                              lang=("lang", lambda s: s.mode().iat[0]), sample=("sample", "first"))
        g["invalid_share"] = g.invalid / g.clusters.clip(lower=1)
        g = g.sort_values("invalid_share", ascending=False)
        g.to_parquet(os.path.join(AUDIT_DIR, "tables.parquet"))
        no = D[D.how.isin(["shape", "none"])].groupby("family").agg(script=("script", "first"), letters=("letters", "sum"), invalid=("invalid", "sum"),
                                                                   clusters=("clusters", "sum"), chars=("chars", "sum"), doc=("doc", "first"))
        audit = {f: [r.script, int(r.chars), round(r.invalid / max(r.clusters, 1), 3), r.doc] for f, r in no.iterrows()}
        json.dump(audit, open(os.path.join(T.TABLES, "AUDIT.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
        print(len(audit), "font families no table decodes -> AUDIT.json")
        pd.set_option("display.width", 250, "display.max_rows", 400, "display.max_colwidth", 40)
        print(g.drop(columns="sample").to_string())


if __name__ == "__main__":
    main()
