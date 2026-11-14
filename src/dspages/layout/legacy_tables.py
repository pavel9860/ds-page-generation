"""Legacy-font text to Unicode by the published tables of each font family: exact where a table exists.

  Padma (101 fonts: Kruti Dev, Chanakya, DV-TT*, Shivaji, Gopika, TAB/TAM ... ; Devanagari, Bengali, Gujarati, Gurmukhi, Kannada, Malayalam, Tamil, Telugu)
        its own JavaScript tables and parser run in V8 (layout.padma)                                      GPL v2 or later
  Preeti, Kantipur, Sagarmatha, PCS Nepali, Fontasy Himali (Nepali)    npttf2utf map.json: character map + post rules          GPL v3
  ML-TT*, MLB-*, ML-Revathi ... (170 Malayalam fonts)                  indic-archive/ml-ascii-fontmaps maps (CC0), pre/post vowel-sign algorithm
  Nudi, Baraha (Kannada)                                               ascii2unicode mapping + vattakshara / broken-case rules   GPL v3
  Bijoy, SutonnyMJ and the other *MJ fonts (Bangla)                    bijoy2unicode conversion maps and re-ordering             MPL 2.0
The data is in Layouts/archive/info/legacy/tables (copied from the projects, their code is not run). `decode(font name, text)` -> Unicode, or None for a
font without a table; `known(font name)` says whether there is one."""
import collections
import json
import os
import re
import unicodedata
from functools import lru_cache, partial

from . import padma
from . import legacy
from .legacy import CACHE

TABLES = os.path.join(CACHE, "tables")


def key(name):
    return re.sub(r"[^a-z0-9]", "", name.split("+")[-1].lower())


# ------------------------------------------------------------------------------------------------ Preeti family
@lru_cache(maxsize=1)
def _preeti():
    rules = json.load(open(os.path.join(TABLES, "preeti.json"), encoding="utf-8"))
    return {key(n): v["rules"] for n, v in rules.items()}


def _preeti_decode(rules, text):
    out = []
    for word in re.findall(r"\s+|\S+", text):
        for a, b in rules["pre-rules"]:
            word = re.sub(a, b, word)
        w = "".join(rules["character-map"].get(c, c) for c in word)
        for a, b in rules["post-rules"]:
            w = re.sub(a, b, w)
        out.append(w)
    return "".join(out)


# ------------------------------------------------------------------------------------------------ Malayalam
@lru_cache(maxsize=1)
def _ml():
    idx = json.load(open(os.path.join(TABLES, "ml", "index.json")))
    return {key(k): v for k, v in idx.items()}


@lru_cache(maxsize=None)
def _ml_map(fname):
    return json.load(open(os.path.join(TABLES, "ml", fname), encoding="utf-8"))


ML_PRE = ["േ", "െ", "ൈ", "ോ", "ൊ", "ൌ", "ൗ", "്ര"]
ML_POST = ["്യ", "്വ"]


def _ml_decode(m, txt):
    """The reference algorithm of the table's project: a vowel sign typed before its consonant goes after it (after a following -ra / -ya / -va too)."""
    out, n, txt = [], 0, list(txt)
    while n < len(txt):
        c = txt[n]
        if c in " \n" or c not in m:
            out.append(c)
            n += 1
            continue
        v = m[c]
        if v in ML_PRE:
            if n + 2 < len(txt):
                if m.get(txt[n + 1]) == "്ര":
                    out.extend([m.get(txt[n + 2], txt[n + 2]), m.get(txt[n + 1], txt[n + 1]), v])
                    n += 3
                    continue
                if m.get(txt[n + 2]) in ML_POST:
                    out.extend([m.get(txt[n + 1], txt[n + 1]), m.get(txt[n + 2], txt[n + 2]), v])
                    n += 3
                    continue
            if n + 1 < len(txt):
                if m.get(txt[n + 1]) == "െ":
                    out.extend([m.get(txt[n + 2], txt[n + 2]) if n + 2 < len(txt) else "", m.get(txt[n + 1], txt[n + 1]), v])
                    n += 3
                    continue
                out.extend([m.get(txt[n + 1], txt[n + 1]), v])
                n += 2
                continue
        out.append(v)
        n += 1
    return "".join(out)


# ------------------------------------------------------------------------------------------------ Nudi (Kannada)
@lru_cache(maxsize=1)
def _nudi():
    return json.load(open(os.path.join(TABLES, "nudi.json"), encoding="utf-8"))


def _nudi_word(t, w):
    maps, vatt, broken, dep, ign = t["mappings"], t["vattaksharagalu"], t["broken_cases"], t["dependent_vowels"], t["ignoreList"]
    i, out = 0, []
    while i < len(w):
        if w[i] in ign:
            i += 1
            continue
        done = False
        for j in (5, 4, 3, 2, 1):
            if w[i:i + j] in maps:
                out += list(maps[w[i:i + j]])
                i += j
                done = True
                break
        if done:
            continue
        c = w[i]
        if c in vatt:
            if out and out[-1] in dep:
                rep = vatt[c] + out[-1]
                out[-1] = "್"
                out += list(rep)
            else:
                out += ["್", vatt[c]]
        elif c in broken:
            if out and out[-1] in broken[c]["mapping"]:
                out[-1] = broken[c]["mapping"][out[-1]]
            else:
                out += [broken[c]["value"]]
        else:
            out.append(c)
        i += 1
    return "".join(out)


def _nudi_decode(text):
    t = _nudi()
    return " ".join(_nudi_word(t, w) for w in text.split(" "))


# ------------------------------------------------------------------------------------------------ Bijoy (Bangla)
@lru_cache(maxsize=1)
def _bijoy():
    return {k: v for k, v in json.load(open(os.path.join(TABLES, "bijoy.json"), encoding="utf-8")).items()}


def _bmap(text, pairs):
    for a, b in pairs:
        text = re.sub(a, b, text)
    return text


PRE_KAR, POST_KAR = set("িৈে"), set("াোৌৗুূীৃ")
BANJON = set("কখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়ৎংঃঁ") - set("ংঃঁ")
HALANT, NUKTA = "্", "ঁ"


def _bangla_rearrange(s):
    """Re-ordering of the converted Bangla text (reph, vowel signs written before their consonant), as the table's project does."""
    def at(k):
        return s[k] if 0 <= k < len(s) else ""
    i = 0
    while i < len(s):                                       # reph: ra + halant after a halant moves before its cluster
        if i < len(s) - 1 and s[i] == "র" and at(i + 1) == HALANT and at(i - 1) == HALANT:
            j = 1
            while i - j >= 0:
                if at(i - j) in BANJON and at(i - j - 1) == HALANT:
                    j += 2
                elif j == 1 and (at(i - j) in PRE_KAR or at(i - j) in POST_KAR):
                    j += 1
                else:
                    break
            s = s[:i - j] + s[i] + s[i + 1] + s[i - j:i] + s[i + 2:]
            i += 1
            continue
        i += 1
    s = s.replace("্্", "্")
    i = 0
    while i < len(s):
        if i < len(s) - 1 and s[i] == "র" and at(i + 1) == HALANT and at(i - 1) != HALANT and at(i + 2) != HALANT:
            j = 1
            while i - j >= 0:
                if at(i - j) in BANJON and at(i - j - 1) == HALANT:
                    j += 2
                elif j == 1 and (at(i - j) in PRE_KAR or at(i - j) in POST_KAR):
                    j += 1
                else:
                    break
            s = s[:i - j] + s[i] + s[i + 1] + s[i - j:i] + s[i + 2:]
            i += 1
            continue
        if i > 0 and s[i] == HALANT and (at(i - 1) in PRE_KAR or at(i - 1) in POST_KAR or at(i - 1) == NUKTA) and i < len(s) - 1:
            s = s[:i - 1] + s[i] + s[i + 1] + s[i - 1] + s[i + 2:]
        if 0 < i < len(s) - 1 and s[i] == HALANT and at(i - 1) == "র" and at(i - 2) != HALANT and (at(i + 1) in PRE_KAR or at(i + 1) in POST_KAR):
            s = s[:i - 1] + s[i + 1] + s[i - 1] + s[i] + s[i + 2:]
        if i < len(s) - 1 and s[i] in PRE_KAR and at(i + 1) not in " \t\n\r":
            j = 1
            while i + j < len(s) - 1 and at(i + j) in BANJON:
                if at(i + j + 1) == HALANT:
                    j += 2
                else:
                    break
            tail, step = s[i], 0
            if s[i] == "ে" and at(i + j + 1) == "া":
                tail, step = "ো", 1
            elif s[i] == "ে" and at(i + j + 1) == "ৗ":
                tail, step = "ৌ", 1
            s = s[:i] + s[i + 1:i + j + 1] + tail + s[i + j + step + 1:]
            i += j
        if i < len(s) - 1 and at(i) == NUKTA and at(i + 1) in POST_KAR:
            s = s[:i] + s[i + 1] + s[i] + s[i + 2:]
        i += 1
    return s


def _bijoy_decode(text):
    b = _bijoy()
    text = _bmap(text, b["preConversionMap"])
    text = _bmap(text, b["conversionMap"])
    text = _bangla_rearrange(text)
    return _bmap(text, b["postConversionMap"])


# ------------------------------------------------------------------------------------------------ routing
# ------------------------------------------------------------------------------------------------ Geetanjali (Assamese)
@lru_cache(maxsize=1)
def _geet():
    """The ordered regex rules of ConvertGeet2Uni (JS pattern, replacement), compiled."""
    return [(re.compile(a), b) for a, b in json.load(open(os.path.join(TABLES, "geetanjali.json"), encoding="utf-8"))]


def _geet_decode(text):
    for p, r in _geet():
        text = p.sub(r, text)
    return text


# ------------------------------------------------------------------------------------------------ Punjabi (Dr Chatrik, Satluj, Anmol ...)
@lru_cache(maxsize=1)
def _pun_files():
    d = os.path.join(TABLES, "punjabi")
    return {f[:-5]: json.load(open(os.path.join(d, f), encoding="utf-8")) for f in os.listdir(d)}


@lru_cache(maxsize=None)
def _pun_config(font):
    """The mapper of punjabi_font_converter (prabnz, MIT) from `font` to Unicode: legacy char(s) -> Unicode, the groups, the move sets."""
    f = _pun_files()
    src, dst = f[font]["characterCodes"], f["unicode"]["characterCodes"]
    groups = f["groups"] + f["tight_groups"]

    def chars(names, table):
        cs = [table.get(n) for n in names]
        return "" if any(c is None for c in cs) else "".join(cs)

    mapper = {src[n]: c for n, c in dst.items() if n in src}
    gmap = {}
    for sub in groups:
        t = [chars(n, dst) for n in sub]
        t = next((x for x in t if x), "")
        for sc in (chars(n, src) for n in sub):
            if sc and (t or sc):
                gmap[sc] = t
    merged = {**gmap, **mapper}
    right = {dst[n] for n in f["unicode"]["moveRightCharacters"] if n in dst} - {dst[n] for n in f[font].get("moveRightCharacters") or [] if n in dst}
    across = {c for sub in f["tight_groups"] for n in sub for c in [chars(n, dst)] if c}
    return merged, right, across, max([len(k) for k in gmap] or [1])


def _pun_decode(font, text):
    mapper, right, across, width = _pun_config(font)
    out, pend, idx, i = [], "", 0, 0
    while i < len(text):
        for j in range(width, 0, -1):
            if text[i:i + j] in mapper:
                add = mapper[text[i:i + j]]
                i += j - 1
                break
        else:
            add = text[i]
        i += 1
        if add in right:
            if pend:
                out.append(pend)
            pend, idx = add, 0
        elif pend:
            if idx < 1:
                idx = 1
                out.append(add)
            elif add in across:
                out.append(add)
            else:
                out += [pend, add]
                pend, idx = "", 0
        else:
            out.append(add)
    if pend:
        out.append(pend)
    return "".join(out)


# ------------------------------------------------------------------------------------------------ Priyaanka family (Telugu)
@lru_cache(maxsize=1)
def _priya():
    """The glyph-name table, the multi-glyph units and the sign attachment rules of the Priyaanka research decoder (indic-legacy-pdf, MIT)."""
    m = json.load(open(os.path.join(TABLES, "priyaanka.json"), encoding="utf-8"))
    units = {}
    for u in m["multi_glyph_units"]:
        units.setdefault(u["glyphs"][0], []).append(u)
    for v in units.values():
        v.sort(key=lambda u: (-len(u["glyphs"]), -u["support"], u["glyphs"], u["unicode"]))
    return m, units


_TE_SIGNS = frozenset("ాిీుూృౄెేైొోౌౢౣ")


def _te_base(t):
    return any("క" <= c <= "హ" for c in t) and not t.startswith("్")


def _te_sign(t):
    return bool(t) and all(c in _TE_SIGNS for c in t)


def _priya_word(names):
    m, units = _priya()
    ch, i = [], 0
    while i < len(names):
        u = next((u for u in units.get(names[i], []) if tuple(names[i:i + len(u["glyphs"])]) == tuple(u["glyphs"])), None)
        n = len(u["glyphs"]) if u else 1
        ch.append((tuple(names[i:i + n]), u["unicode"] if u else m["table"].get(names[i], names[i])))
        i += n
    rules = {(tuple(r["glyphs"]), r["unicode"]): r for r in m["ordering_rules"] if not r.get("context")}
    em = [t for _, t in ch]
    attached, removed = {}, set()
    for i, (g, t) in enumerate(ch):
        r = rules.get((g, t)) if _te_sign(t) else None
        if r is None or r["direction"] == "none":
            continue
        step = -1 if r["direction"] == "previous" else 1
        j, target = i + step, None
        while 0 <= j < len(ch):
            if _te_base(em[j]):
                target = j
                break
            if em[j] and not (_te_sign(em[j]) or em[j].startswith("్")):
                break
            j += step
        if target is not None:
            attached.setdefault(target, []).append((i, t))
            removed.add(i)
    out, used = [], set(removed)
    for i, v in enumerate(em):
        if i in used:
            continue
        out.append(v)
        if i not in attached:
            continue
        j = i + 1
        while j < len(ch):
            if j in removed or not em[j]:
                j += 1
            elif em[j].startswith("్"):
                out.append(em[j])
                used.add(j)
                j += 1
            else:
                break
        out.extend(t for _, t in sorted(attached[i]))
    return unicodedata.normalize("NFC", "".join(out))


@lru_cache(maxsize=1)
def _agl():
    """Unicode -> glyph names: the AGLFN name, the full AGL list (fi, fl are only there), then the names of its compatibility twins (Greek mu reported for the micro sign: mu)."""
    from fontTools.agl import LEGACY_AGL2UV, UV2AGL
    names = collections.defaultdict(list)
    for u, n in UV2AGL.items():
        names[u].append(n)
    for n, us in sorted(LEGACY_AGL2UV.items()):
        for u in us:
            names[u].append(n)
    for u in list(names):
        t = unicodedata.normalize("NFKC", chr(u))
        if len(t) == 1 and ord(t) != u:
            names[ord(t)] += names[u]
    return names


def _priya_decode(text):
    t = _priya()[0]["table"]
    name = lambda c: next((n for n in _agl().get(ord(c), ()) if n in t or n in _priya()[1]), c)      # a code the table lacks stays as typed (the glyph shapes read it)
    return re.sub(r"\S+", lambda w: _priya_word([name(c) for c in w.group()]), text)


@lru_cache(maxsize=None)
def _chain(name):
    """tables/chains.json: {name: [[JS regex, replacement], ...]}, the ordered replace chain of a published converter (Open-SL Sinhala)."""
    return [(re.compile(a), b) for a, b in json.load(open(os.path.join(TABLES, "chains.json"), encoding="utf-8"))[name]]


@lru_cache(maxsize=1)
def _khmer():
    """tables/khmer/conversiondata.db (KhmerConverterPHP, GPL2: 782 Khmer legacy fonts, Limon, ABC, Mondulkiri ...): ({font key: table number}, [(code map, code
    sequences)]), the fonts that share a table under one number."""
    import sqlite3
    db = sqlite3.connect(f"file:{os.path.join(TABLES, 'khmer', 'conversiondata.db')}?mode=ro", uri=True)
    names, tables, seen = {}, [], {}
    for full, ps, tu in db.execute("select fontFullName, PostscriptName, toUnicode from conversion"):
        i = seen.setdefault(tu, len(seen))
        if i == len(tables):
            m, seq = json.loads(tu.replace("\\\n", "\\n"))              # two rows hold a backslash + line break: the \\n escape
            tables.append((m, _first(seq)))
        for n in (full, ps):
            if n:
                names.setdefault(key(n), i)
    return names, tables


def _longest(m, first, text, script):
    """Text of a run under a table of codes and code sequences (m: one code -> text; first: {code: [(sequence, text)], longest first}): the longest
    sequence at each place, else the code's own text, then logical order (sequence)."""
    out, k = [], 0
    while k < len(text):
        hit = next(((a, b) for a, b in first.get(text[k], ()) if text.startswith(a, k)), None)
        out.append(hit[1] if hit else m.get(text[k], text[k]))
        k += len(hit[0]) if hit else 1
    return sequence(out, script)


def _first(seq):
    """{code: [(sequence, text)]} of a table's code sequences, longest first."""
    first = collections.defaultdict(list)
    for a, b in sorted(seq.items(), key=lambda kv: -len(kv[0])):
        first[a[0]].append((a, b))
    return dict(first)


def _khm_decode(i, text):
    m, first = _khmer()[1][i]
    return _longest(m, first, text, "Khmer")


@lru_cache(maxsize=1)
def _lipi():
    """tables/legacylipi.json (tools/import_tables.py; legacylipi, MIT): {table: (one-code map, code sequences)} of Devanagari legacy fonts."""
    f = os.path.join(TABLES, "legacylipi.json")
    src = json.load(open(f, encoding="utf-8")) if os.path.exists(f) else {}
    return {k: ({c: t for c, t in v.items() if len(c) == 1}, _first({c: t for c, t in v.items() if len(c) > 1})) for k, v in src.items()}


@lru_cache(maxsize=1)
def _payyans():
    """tables/payyans.json (tools/import_tables.py; libindic Payyans maps, GPLv3): {map: (one-code map, code sequences, script of its letters)}."""
    from hwsynth.scripts import script
    f = os.path.join(TABLES, "payyans.json")
    src = json.load(open(f, encoding="utf-8")) if os.path.exists(f) else {}
    out = {}
    for k, v in src.items():
        sc = collections.Counter(script(c) for t in v.values() for c in t if c.isalpha() and script(c) != "Latin").most_common(1)
        out[k] = ({c: t for c, t in v.items() if len(c) == 1}, _first({c: t for c, t in v.items() if len(c) > 1}), sc[0][0] if sc else "Latin")
    return out


LIPI_FONTS = {"dvbwtt": "dvb_tt", "dvbtt": "dvb_tt", "walkmanchanakya": "walkman_chanakya", "chanakya": "chanakya", "shreedev": "shree_dev",
              "shreelipi": "shree_lipi", "shusha": "shusha", "apsdv": "aps_dv"}       # font key prefix -> legacylipi table (its MappingTable variants)


@lru_cache(maxsize=None)
def _tbl(name):
    """A UTFC Tibetan .tbl (GPL3): records split by a space, each the 4-digit decimal code points of the glyph typed with code 33 + index."""
    b = open(os.path.join(TABLES, "tibetan", name), "rb").read().decode("latin1")
    return ["".join(chr(int(x)) for x in re.findall(r"\d{4}", r)) for r in b.split(" ")]


@lru_cache(maxsize=None)
def _charmap(name):
    """tables/charmaps.json: {name: {typed char: Unicode text}} for layouts that are a plain key -> letter table (Georgian Nusx)."""
    return json.load(open(os.path.join(TABLES, "charmaps.json"), encoding="utf-8"))[name]


@lru_cache(maxsize=1)
def _aliases():
    """Fonts that share the table of another published font (tables/aliases.json: font key -> [kind, name of the table]); each was checked
    against the shape of its glyphs (tests/legacy_label), a font that has an exact table of its own never reaches this."""
    f = os.path.join(TABLES, "aliases.json")
    return json.load(open(f)) if os.path.exists(f) else {}


def _alias(k):
    a = _aliases()
    k = next((x for x in sorted(a, key=len, reverse=True) if k.startswith(x) and len(x) >= 5), k) if k not in a else k   # FMMalithix -> fmmalithi
    while k:
        if k in a:
            kind, name = a[k]
            return kind, {"padma": lambda: padma.faces()[name], "preeti": lambda: _preeti()[name], "ml": lambda: _ml().get(name, name), "pun": lambda: name, "geet": lambda: None,
                          "nudi": lambda: None, "bijoy": lambda: None, "cp": lambda: name, "mb": lambda: name, "iconv": lambda: name, "icu": lambda: name, "lipi": lambda: name, "pay": lambda: name, "khm": lambda: int(name), "map": lambda: name,
                          "tbl": lambda: name, "chain": lambda: name, "priya": lambda: name, "labelled": lambda: name}[kind]()      # the arguments run() takes, from the decoders() key the alias names
        m = re.search(r"(bold|italic|regular|normal|plain)$", k)
        if not m:
            return None
        k = k[:m.start()]
    return None


@lru_cache(maxsize=1)
def _ocr():
    """tables/ocr.json (tools/ocr_tables.py): {"_scripts": {family: script}, family: {typed char: text}}, the fonts without any table read from their rendered lines (an entry
    of an older file is [text, ...]); keys starting with '_' are bookkeeping."""
    f = os.path.join(TABLES, "ocr.json")
    raw = json.load(open(f, encoding="utf-8")) if os.path.exists(f) else {}
    return {k: v if k.startswith("_") else {c: t if isinstance(t, str) else t[0] for c, t in v.items()} for k, v in raw.items()}


@lru_cache(maxsize=1)
def _labelled():
    """tables/labelled.json: {font family key: {typed char: Unicode text}}, the glyphs no published table covers, read one by one from their pictures
    (tests/legacy_label, tools/export_labelled.py): they fill the codes a published table leaves as typed, or are the table of a font that has none. The tables read from the
    rendered lines (ocr.json) stand behind the hand labels."""
    f = os.path.join(TABLES, "labelled.json")
    lab = json.load(open(f, encoding="utf-8")) if os.path.exists(f) else {}
    for fam, codes in _ocr().items():
        if not fam.startswith("_"):
            lab[fam] = {**codes, **lab.get(fam, {})}
    return lab


def sequence(pieces, script):
    """Text of glyph pieces set in order, in logical order: a reph piece (ra + virama, set after its cluster) moves before the cluster, then the prefix vowel signs
    (layout.legacy.logical). pieces: the text of each glyph of the run."""
    from .legacy import logical
    if script == "Arabic":
        return rtl("".join(pieces))
    vir = virama(script) if script in _SEQ else ""
    out, start = [], 0
    for pc in pieces:
        if vir and len(pc) == 2 and pc[1] == vir and unicodedata.name(pc[0], "").endswith(" RA"):
            out.insert(start, pc)
            continue
        if pc and unicodedata.category(pc[0]) == "Lo" and not (out and vir and out[-1][-1:] == vir):
            start = len(out)
        out.append(pc)
    return logical("".join(out), script)


@lru_cache(maxsize=None)
def virama(script):
    """The virama (halant, coeng) of the script's block, '' if it has none."""
    from .legacy import ranges_of
    return next((chr(c) for a, b in ranges_of(script) for c in range(a, b + 1) if unicodedata.name(chr(c), "").endswith(("VIRAMA", "SIGN COENG", "HALANT"))), "")


def rtl(text):
    """Visual order -> logical order of a right-to-left run set glyph by glyph: letters (and the spaces between them) are reversed, runs of digits and Latin keep their order,
    and the runs themselves come in reverse."""
    runs, cur, kind = [], "", None
    for ch in text:
        b = unicodedata.bidirectional(ch)
        k = "n" if b in ("EN", "AN", "L") else ("r" if b in ("R", "AL") or unicodedata.category(ch)[0] == "L" else None)
        if k is None:
            k = kind or "r"                                           # a space or sign stays with the run it follows
        if kind is not None and k != kind:
            runs.append((kind, cur))
            cur = ""
        kind, cur = k, cur + ch
    if cur:
        runs.append((kind, cur))
    return unicodedata.normalize("NFC", "".join(r if k == "n" else r[::-1] for k, r in reversed(runs)))


_SEQ = {"Arabic", "Devanagari", "Bengali", "Gurmukhi", "Gujarati", "Oriya", "Tamil", "Telugu", "Kannada", "Malayalam", "Sinhala", "Khmer", "Myanmar"}


@lru_cache(maxsize=1)
def _label_source():
    """tables/labelled_source.json: {family: {typed char: "label" | "guess:<where from>"}}, which entries of labelled.json were answered by hand."""
    f = os.path.join(TABLES, "labelled_source.json")
    src = json.load(open(f, encoding="utf-8")) if os.path.exists(f) else {}
    for fam, codes in _ocr().items():
        if not fam.startswith("_"):
            src[fam] = {**{c: "guess:ocr" for c in codes}, **src.get(fam, {})}
    return src


def label_source(name, ch):
    """"label" (answered by hand), "guess:words|ocr|shape" (best candidate of the labelling page) or None (no entry) for a typed char of the font."""
    fam = _lab_family(name)
    return _label_source().get(fam, {}).get(ch) if fam else None


def _lab_family(name):
    k, a = key(name), _labelled()
    while k:
        if k in a:
            return k
        m = re.search(r"(bold|italic|light|thin|semibold|demibold|heavy|regular|normal|plain|oblique|black|medium)$", k)
        if not m:
            return None
        k = k[:m.start()]
    return None


@lru_cache(maxsize=None)
def route(name):
    """(kind, argument) of the table that decodes the font, or None; a name no table knows is tried again as its family (KANTIPUR,Bold -> kantipur).
    The name is a first guess: legacy.table_for checks it against every table by the text it gives."""
    r = _route(name)
    if r is None and name:
        f = legacy.family(name)
        if f and f != key(name):
            r = _route(f)
    return r


def _route(name):
    k = key(name)
    face = padma.face_of(name)
    if face:
        return "padma", face
    if k in _preeti():
        return "preeti", _preeti()[k]
    k = re.sub(r"(bold|italic|regular|normal)+$", "", k) if k.startswith("ml") else k
    for kk, f in _ml().items():                                       # ML-Revathi / MLB-Revathi / ML-TTRevathi are one encoding; FML-* fonts are another one
        if kk == k or (k.startswith("ml") and kk in (k.replace("ml", "mlb", 1), k.replace("ml", "mltt", 1))):
            return "ml", f
    if "zawgyi" in k:
        return "icu", "Zawgyi-my"
    lipi = next((t for pre, t in sorted(LIPI_FONTS.items(), key=lambda kv: -len(kv[0])) if k.startswith(pre)), None)
    if lipi and lipi in _lipi() and not padma.face_of(name):           # a font Padma knows keeps its Padma table
        return "lipi", lipi
    if k in _khmer()[0]:
        return "khm", _khmer()[0][k]
    if k.startswith("nudi") or k.startswith("baraha"):
        return "nudi", None
    base = re.sub(r"(bold|italic|regular|normal)+$", "", k)
    if base.endswith("mj") or base.startswith("bijoy"):
        return "bijoy", None
    if k.startswith("gurbanilipi"):                                   # its codes are those of Anmol Lipi (checked on a page: the converter's own gurbani_lipi table is the identity)
        return "pun", "anmol"
    for stem in _pun_files():                                         # the Punjabi tables by their own font names (Asees, Anmol Lipi, Satluj ...)
        sk = key(stem)
        if stem not in ("groups", "tight_groups", "unicode") and (k == sk or (len(sk) >= 4 and k.startswith(sk))):
            return "pun", stem
    for pre, fc in (("dvtt", "dvttganesh"), ("dvwtt", "dvwttganesh")):   # the DV-TT fonts (Shridhar, Bhima, Dhruv ...) share the encoding of DV-TT Ganesh / Yogesh / Surekh
        if k.startswith(pre):
            return "padma", padma.faces()[fc]
    for pre, ch in (("akrutiori", "akruti_sarala"), ("orttsarala", "orttsarala"), ("ortt", "orttsarala"), ("apuniaprant", "aprant"), ("aprant", "aprant"), ("shreeori", "shreelipi_odia")):
        if k.startswith(pre):                                         # Odia converters of OdiaWikimedia/Converter (GPL-3), tables/chains.json
            return "chain", ch
    return _alias(k) or (("labelled", fam) if (fam := _lab_family(name)) else None)


def raw(chars, font=None):
    """The typed code string of a run: the characters MuPDF reports, with the 0xF000 offset of symbolic fonts removed (a Priyaanka font: as reported,
    its table is keyed on the glyph names of those characters); a glyph the PDF maps to no character (U+FFFD: ornaments such as separator stars) types nothing."""
    mac = (lambda c: c) if font and (route(font) or ("",))[0] == "priya" else _mac_byte
    return "".join(chr(u & 0xFF) if 0xF000 <= u <= 0xF0FF else mac(chr(u)) for u in chars if u != 0xFFFD)


def _mac_byte(c):
    """A character above Latin-1 that is a code of the Mac Roman table (the font's bytes read as Mac Roman by the text map): the byte, as the Windows tables type it;
    a C1 control (the byte itself): its cp1252 character."""
    if 0x80 <= ord(c) < 0xA0:                                          # a C1 control: the byte as the text map gave it, the Windows tables know it by its cp1252 character
        return bytes([ord(c)]).decode("cp1252", "replace")
    if ord(c) < 256:
        return c
    try:
        c.encode("cp1252")                                             # a Windows character (‡ ” …): the tables type it as it is
        return c
    except UnicodeEncodeError:
        pass
    try:
        return chr(c.encode("mac_roman")[0])
    except UnicodeEncodeError:
        return c


def known(name):
    return route(name) is not None


def decode(name, text):
    """Unicode text of a run typed in the legacy font (published table first; the labelled glyphs replace the codes it leaves as typed)."""
    out = _decode(name, text)
    fam = _lab_family(name)
    if out is None or not fam:
        return out
    m = _labelled()[fam]
    return "".join(m.get(c, c) if _untouched(name, c) else c for c in out)


@lru_cache(maxsize=None)
def _untouched(name, c):
    """True when the font's published table leaves the char as typed (the only codes a label may fill)."""
    r = route(name)
    return r is None or r[0] == "labelled" or _decode(name, c) == c


def _decode(name, text):
    r = route(name)
    return None if r is None else run(*r, text)


def run(kind, arg, text):
    """Text of a run typed under one table: kind and arg as route() gives them."""
    if kind == "labelled":
        m, sc = _labelled()[arg], _ocr().get("_scripts", {}).get(arg)
        if sc:                                                         # a table read from lines (tools/ocr_tables.py): glyphs of one letter are set in visual order
            return sequence([m.get(c, c) for c in text], sc)
        return "".join(m.get(c, c) for c in text)
    if kind == "padma":
        return padma.convert(arg, text)
    if kind == "preeti":
        return _preeti_decode(arg, text)
    if kind == "ml":
        return _ml_decode(_ml_map(arg), text)
    if kind == "nudi":
        return _nudi_decode(text)
    if kind == "geet":
        return _geet_decode(text)
    if kind == "priya":
        return _priya_decode(text)
    if kind == "chain":
        for p, r in _chain(arg):
            text = p.sub(r, text)
        return text
    if kind == "tbl":
        r = _tbl(arg)
        return "".join(r[ord(c) - 33] if 33 <= ord(c) < 33 + len(r) and r[ord(c) - 33] else c for c in text)
    if kind == "cp":
        return "".join(bytes([ord(c)]).decode(arg, "replace") if ord(c) < 256 and c not in " \n" else c for c in text)
    if kind == "map":
        m = _charmap(arg)
        return "".join(m.get(c, c) for c in text)
    if kind == "pun":
        return _pun_decode(arg, text)
    if kind == "mb":
        return _multibyte(arg, text)
    if kind == "khm":
        return _khm_decode(arg, text)
    if kind == "pay":
        m, first, sc = _payyans()[arg]
        return _longest(m, first, text, sc)
    if kind == "lipi":
        return _longest(*_lipi()[arg], text, "Devanagari")
    if kind == "icu":
        return _icu(arg)(text)
    if kind == "iconv":
        m = _iconv_map(arg)
        return "".join(m.get(c, c) for c in text)
    return _bijoy_decode(text)


MULTIBYTE = ("big5", "cp950", "gbk", "gb18030", "shift_jis", "cp932", "euc_jp", "euc_kr", "cp949")


ICU = ("Zawgyi-my",)                                                 # ICU transforms of a legacy encoding (CLDR rules: Zawgyi Burmese)
NATIVE_CODED = {"icu"}                                                # kinds whose fonts type the script's own code points in a wrong order or form: known by the font's name only


@lru_cache(maxsize=None)
def _icu(rule):
    """The system libicui18n transliterator `rule` (utrans_openU / utrans_transUChars): a function text -> text."""
    import ctypes
    import glob
    path = max(glob.glob("/usr/lib/*/libicui18n.so.[0-9]*") + glob.glob("/usr/lib/libicui18n.so.[0-9]*"), key=len)
    lib, v = ctypes.CDLL(path), "_" + re.search(r"\.so\.(\d+)", path).group(1)
    op, tr = getattr(lib, "utrans_openU" + v), getattr(lib, "utrans_transUChars" + v)
    op.restype = ctypes.c_void_p
    op.argtypes = [ctypes.c_char_p, ctypes.c_int32, ctypes.c_int, ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    tr.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int)]
    rid, err = rule.encode("utf-16-le"), ctypes.c_int(0)
    h = op(rid, len(rid) // 2, 0, None, 0, None, ctypes.byref(err))      # UTRANS_FORWARD
    if err.value > 0 or not h:
        raise LookupError(f"ICU has no transform {rule} (error {err.value})")

    def conv(text):
        b = text.encode("utf-16-le")
        cap = 4 * len(b) + 64
        buf, n, lim, e = ctypes.create_string_buffer(b, cap), ctypes.c_int32(len(b) // 2), ctypes.c_int32(len(b) // 2), ctypes.c_int(0)
        tr(h, buf, ctypes.byref(n), cap // 2, 0, ctypes.byref(lim), ctypes.byref(e))
        if e.value > 0:
            raise OSError(f"ICU {rule} failed: error {e.value}")
        return buf.raw[:2 * n.value].decode("utf-16-le")
    return conv


ICONV = ("ARMSCII-8", "GEORGIAN-ACADEMY", "GEORGIAN-PS", "TSCII", "KOI8-T", "KOI8-U", "KOI8-RU", "TIS-620")       # glibc's tables Python's codecs lack


@lru_cache(maxsize=None)
def _iconv(name):
    """glibc iconv (the C library's own tables, iconv_open / iconv): a function bytes -> text for the 8-bit charset name."""
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    libc.iconv_open.restype = ctypes.c_void_p
    libc.iconv_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    libc.iconv.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_size_t)] * 2
    cd = libc.iconv_open(b"UTF-32LE", name.encode())
    if cd in (None, ctypes.c_void_p(-1).value):
        raise LookupError(f"iconv has no {name}")

    def conv(b):
        out = []
        for byte in b:                                                 # one byte at a time: an 8-bit charset, and a byte it lacks stays as typed
            src, dst = ctypes.create_string_buffer(bytes([byte]), 1), ctypes.create_string_buffer(16)
            ps, pd = ctypes.c_char_p(ctypes.addressof(src)), ctypes.c_char_p(ctypes.addressof(dst))
            ns, nd = ctypes.c_size_t(1), ctypes.c_size_t(16)
            ok = libc.iconv(cd, ctypes.byref(ps), ctypes.byref(ns), ctypes.byref(pd), ctypes.byref(nd)) != ctypes.c_size_t(-1).value
            out.append(dst.raw[:16 - nd.value].decode("utf-32-le") if ok else chr(byte))
        return "".join(out)
    return conv


@lru_cache(maxsize=None)
def _iconv_map(name):
    """{typed char: text} of the 8-bit charset (bytes 0x80-0xFF, the codes a legacy font types; ASCII stays)."""
    conv = _iconv(name)
    return {chr(b): t for b in range(0x80, 0x100) if (t := conv(bytes([b]))) and t != chr(b)}


def _multibyte(codec, text):
    """A run typed in a double-byte legacy font (Big5, GBK, Shift-JIS, EUC-KR ...): the typed codes are the bytes of the text; a character above 255 is not a byte and stays."""
    out, run = [], bytearray()
    for c in text:
        if ord(c) < 256:
            run.append(ord(c))
            continue
        out.append(bytes(run).decode(codec, "replace") + c)
        run.clear()
    return "".join(out) + bytes(run).decode(codec, "replace")


CODEPAGES = ("cp1250", "cp1251", "cp1252", "cp1253", "cp1254", "cp1255", "cp1256", "cp1257", "mac_cyrillic", "mac_greek", "cp866", "koi8_r", "cp874")


PROBE = "".join(chr(c) for c in range(0x21, 0x7F)) + "".join(chr(c) for c in range(0xA1, 0x100))      # every code a legacy font types


@lru_cache(maxsize=1)
def scripts_of_decoders():
    """{decoders() key: the scripts a table writes} (its letters of PROBE decoded, each script with >= 10 % of them): table_for weighs only the tables that
    write the document's script."""
    from hwsynth.scripts import script
    out = {}
    for k, f in decoders().items():
        try:
            t = f(PROBE)
        except Exception:                                              # a table that cannot read the probe writes nothing we can tell: weighed everywhere
            out[k] = None
            continue
        c = collections.Counter(script(ch) for ch in t if ch.isalpha())
        n = sum(c.values())
        sc = {sc for sc, v in c.items() if v >= 0.1 * n and sc != "Latin"}   # Latin is the codes left as typed: no sign of what the table writes
        out[k] = sc or None
    return out


def writes(k, script):
    """True when the table writes the script (scripts_of_decoders; CJK-xx: Han, kana or Hangul) or what it writes is unknown."""
    sc = scripts_of_decoders().get(k)
    own = {"Han", "Hiragana", "Katakana", "Hangul"} if script.startswith("CJK") else {script}
    return sc is None or bool(sc & own)


def decoders():
    """{'kind:name': function(text) -> text} for every published table, to compare a font with all of them."""
    def load(name):
        return json.load(open(os.path.join(TABLES, name), encoding="utf-8"))
    d = {f"padma:{n}": partial(run, "padma", f) for n, f in padma.faces().items()}
    d.update({f"preeti:{n}": partial(run, "preeti", t) for n, t in _preeti().items()})
    d.update({f"ml:{n}": partial(run, "ml", f) for n, f in _ml().items()})
    d.update({f"pun:{n}": partial(run, "pun", n) for n in _pun_files() if n not in ("groups", "tight_groups", "unicode")})
    d.update({f"chain:{n}": partial(run, "chain", n) for n in load("chains.json")})
    d.update({f"map:{n}": partial(run, "map", n) for n in load("charmaps.json")})
    d.update({f"cp:{cp}": partial(run, "cp", cp) for cp in CODEPAGES})
    d.update({f"mb:{cp}": partial(run, "mb", cp) for cp in MULTIBYTE})
    d.update({f"tbl:{n}": partial(run, "tbl", n) for n in os.listdir(os.path.join(TABLES, "tibetan")) if n.endswith("2Uni.tbl")})
    d.update({f"{k}:{k}": partial(run, k, None) for k in ("nudi", "geet", "priya", "bijoy")})
    d.update({f"khm:{i}": partial(run, "khm", i) for i in range(len(_khmer()[1]))})
    d.update({f"iconv:{n}": partial(run, "iconv", n) for n in ICONV})
    d.update({f"lipi:{n}": partial(run, "lipi", n) for n in _lipi()})
    d.update({f"pay:{n}": partial(run, "pay", n) for n in _payyans()})
    d.update({f"icu:{n}": partial(run, "icu", n) for n in ICU})
    return d
