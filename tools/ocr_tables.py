"""Fills the tables of the corpus fonts that have none (tables/COVERAGE.json: no table, or only hand labels; tables/AUDIT.json: legacy fonts of the whole corpus), font by font:
 1. cheap and exact first: the typed text read under every published table (layout.legacy_fit.explain), no pictures, no agents;
 2. the evidence: Tesseract reads the spans of the font, agents read pictures (`--dump`: lines, `--sheet`/`--sheet2`: the glyphs drawn from the font program, read twice, only the
    glyphs both readings agree on count) - layout.legacy_evidence;
 3. the verdict (layout.legacy_fit): a font of real Unicode / a symbol font / a published table that says the same / its own table, fitted, gated, deciphered.
Writes tables/ocr.json {"_scripts", "_done", family: {typed char: text}}, tables/aliases.json (published table found), tables/UNICODE_OK.json (no table needed). Hand labels are
compared with the table found (reported) and never overwritten.
python tools/ocr_tables.py [--dump | --sheet | --sheet2] [--docs N --pages N --lines N] [family ...]"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from dspages.layout import legacy as L, legacy_evidence as E, legacy_fit as F, legacy_tables as T

OCR, ALIASES, UNICODE_OK, COVERAGE, AUDIT, LABELLED = (os.path.join(T.TABLES, f) for f in ("ocr.json", "aliases.json", "UNICODE_OK.json", "COVERAGE.json", "AUDIT.json", "labelled.json"))
VERSION = "3"                                                         # bump when the verdict logic changes: finished fonts are then settled again
Outcome = collections.namedtuple("Outcome", "kind value how", defaults=("",))           # kind: alias [kind, name] | unicode agreement | symbol 0 | table {code: text}; how: lines | sheet


def read(path, default):
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else default


def write(path, data, indent=1):
    json.dump(data, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=indent)


def fill(fam, script, corpus, decoders, lang, args):
    """The Outcome for one font, or None when the evidence does not settle it."""
    lines = E.typed_codes(corpus, fam) if lang else []
    if script.startswith("CJK") and lines and not F.has_high_codes(lines):        # ASCII, punctuation or real CJK text: a legacy CJK font needs bytes above 127
        return Outcome("unicode", 1)
    if F.is_english(lines) or (lang and F.is_native(lines, F.char_lm(lang))):                            # its typed text already reads as the language: real Unicode, a table would only damage it
        return Outcome("unicode", 1)
    glyphs = E.agent_glyphs(fam)
    pairs = F.clean(E.ocr_pairs(corpus, fam, script, args.lines, args.pages) + E.agent_pairs(fam))
    if not pairs and not glyphs and lang:
        m = F.explain(lines, script, decoders, F.char_lm(lang))
        if m:
            return Outcome("alias", m[1].split(":", 1))
    if F.is_unicode_font(pairs):
        return Outcome("unicode", 1)
    if F.is_symbol_font(pairs):
        return Outcome("symbol", 0)
    if len(pairs) < 10 and not glyphs:
        return None
    script = F.script_of([e for _, e in pairs] + list(glyphs.values()), script)
    tab = {**glyphs, **(F.table(pairs, script) if len(pairs) >= 10 else {})}      # the fit saw the context: it wins where the two differ
    m = F.published(tab, decoders)
    if m:
        return Outcome("alias", m[1].split(":", 1))
    if tab and lang and len(pairs) >= 10:
        tab = F.decipher(tab, pairs, lines, script, F.char_lm(lang))
    return Outcome("table", tab, "lines" if len(pairs) >= 10 else "sheet") if tab else None


class Store:
    """The three files the verdicts go to; saved after every font, so a failure keeps the finished ones."""
    def __init__(self):
        raw = read(OCR, {})
        self.ocr = {k: v if k.startswith("_") else {c: t if isinstance(t, str) else t[0] for c, t in v.items()} for k, v in raw.items()}
        for k in ("_scripts", "_done", "_source"):
            self.ocr.setdefault(k, {})
        self.aliases, self.unicode = read(ALIASES, {}), read(UNICODE_OK, {"fonts": {}})
        self.unicode.setdefault("symbol", {})

    def settle(self, fam, script, out):
        self.ocr.pop(fam, None)
        self.ocr["_scripts"].pop(fam, None)
        self.ocr["_source"].pop(fam, None)
        if out is None:
            return
        self.aliases.pop(fam, None)                                             # the verdict on this evidence replaces an earlier one
        if out.kind == "alias":
            self.aliases[fam] = out.value
        elif out.kind in ("unicode", "symbol"):
            self.unicode["fonts" if out.kind == "unicode" else "symbol"][fam] = out.value
        else:
            self.ocr[fam] = out.value
            self.ocr["_source"][fam] = out.how                                  # lines: fitted to the text read from pictures (98-100 % right on known fonts); sheet: two agreeing readings of the glyphs (about 80 %)
            self.ocr["_scripts"][fam] = F.script_of(out.value.values(), script)

    def save(self):
        write(OCR, self.ocr, 0)
        write(ALIASES, self.aliases)
        write(UNICODE_OK, self.unicode)


def hand_agreement(fam, out, hand):
    """'n/m agree' between the table found and the codes answered by hand (never changed), '' when there is nothing to compare."""
    mine = {c: t for c, t in hand.get(fam, {}).items() if T.label_source(fam, c) == "label"}
    both = [c for c in mine if out and out.kind == "table" and c in out.value]
    return f"vs hand labels: {sum(F.nfc(mine[c]) == F.nfc(out.value[c]) for c in both)}/{len(both)} agree" if both else ""


def families():
    """{family: script} to work on: no table, read tables, hand-labelled ones, and the legacy fonts of the audit."""
    want = {r["family"]: r["script"] for r in read(COVERAGE, []) if not r["table"] or r["table"] in ("ocr", "labelled")}
    scripts = set(L.LANG_SCRIPT.values())
    want.update({f: v[0] for f, v in read(AUDIT, {}).items() if v[0] in scripts and f not in want})
    return want


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("families", nargs="*")
    ap.add_argument("--dump", action="store_true", help="pictures of text lines for an agent")
    ap.add_argument("--more", action="store_true", help="with --dump: other lines than the ones dumped before (second round)")
    ap.add_argument("--sheet", action="store_true", help="glyph sheet for an agent")
    ap.add_argument("--sheet2", action="store_true", help="second, independent glyph sheet")
    ap.add_argument("--docs", type=int, default=12, help="documents read per font")
    ap.add_argument("--pages", type=int, default=3, help="pages read per document")
    ap.add_argument("--lines", type=int, default=1500, help="spans Tesseract reads per font")
    ap.add_argument("--script", help="script of named families the coverage and the audit do not list (fonts found later)")
    args = ap.parse_args()
    known = families()
    want = {f: known.get(f, args.script) for f in args.families if f in known or args.script} if args.families else known
    corpus = E.Corpus(want, args.docs)
    if args.dump or args.sheet or args.sheet2:
        for fam in want:
            try:
                if not corpus.docs[fam]:
                    corpus.docs[fam] = corpus.scan(fam, want[fam])
                n = E.dump_lines(corpus, fam, pages=args.pages, more=args.more) if args.dump else E.dump_sheet(corpus, fam, second=args.sheet2)
                print(fam, want[fam], n, flush=True)
            except Exception as e:
                print(fam, "failed:", repr(e)[:80], flush=True)
        return
    store, decoders, hand = Store(), T.decoders(), read(LABELLED, {})
    for fam, script in want.items():
        sig = f"v{VERSION}/{args.docs}/{args.pages}/{args.lines}/{E.signature(fam)}"
        if store.ocr["_done"].get(fam) == sig:                                  # settled on this evidence already
            continue
        try:
            out = fill(fam, script, corpus, decoders, corpus.language(fam, script), args)
            store.settle(fam, script, out)
            store.ocr["_done"][fam] = sig
            print(fam, script, out.kind if out else "-", out.value if out and out.kind != "table" else len(out.value) if out else 0, hand_agreement(fam, out, hand), flush=True)
        except Exception as e:
            print(fam, "failed:", repr(e)[:80], flush=True)
        store.save()


if __name__ == "__main__":
    main()
