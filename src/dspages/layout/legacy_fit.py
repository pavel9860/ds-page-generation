"""Tables of legacy fonts from evidence, as pure functions (no PDF, no OCR, no files except the language corpus): given the typed codes of a font and the text a reader (OCR, an
agent) saw in the same spans, find the table.

    pairs = clean(raw_pairs)                        [(codes, text read)], normalized
    table(pairs, script)                            {code: text} the fit two halves of the pairs agree on (98-100 % of the codes right on fonts with known tables)
    published(tab, decoders)                        the published table that already says the same (>= 90 % of the codes)
    explain(lines, script, decoders, lm)            a published table under which the typed lines read as the language (no pictures, no agents)
    decipher(tab, pairs, lines, script, lm)         more codes by noisy-channel EM decipherment with a character trigram model

The method: IBM model 1 between codes and characters gives candidates; coordinate descent keeps, per code, the text under which the decode of every span (legacy_tables.sequence
puts reph and prefix vowel signs in order) is nearest by edit distance to the text read; EM decipherment (Knight et al. 2006, Ravi & Knight 2008, with the glyph-substitution layer
of Garrette et al. 2016) adds codes the pictures leave open, from a good start (Berg-Kirkpatrick & Klein 2013: EM alone ends in local optima)."""
import collections
import functools
import json
import math
import os
import re
import unicodedata

from . import legacy as L, legacy_tables as T

MIN_LINES = 3                       # a code needs this many spans to be written
MODEL1_ROUNDS, MODEL1_MIN_P = 8, 0.08
CANDIDATES, NGRAM_WINDOW = 4, 3
PRIOR_START, EM_ITERATIONS, POSTERIOR = 0.9, 4, 0.9
LM_MARGIN_TYPED, LM_MARGIN_REAL = 1.5, 3.0
PROBE = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" + "".join(chr(c) for c in range(0xC0, 0x100))


def nfc(text):
    return unicodedata.normalize("NFC", text)


CJK = {"CJK-zh": {"Cjk"}, "CJK-ja": {"Cjk", "Hiragana", "Katakana"}, "CJK-ko": {"Hangul", "Cjk"}}


def in_script(ch, script):
    """Whether a character is written in the script of a language (legacy.LANG_SCRIPT names; CJK covers Han, kana, Hangul)."""
    return script_name(ch) in CJK.get(script, {script})


def script_name(ch):
    """The script a character belongs to, as legacy.LANG_SCRIPT names it ('Devanagari', 'Latin' ...)."""
    return unicodedata.name(ch, " ").split()[0].capitalize()


# ------------------------------------------------------------------------------------------------------------------------------------------------- evidence
def lev(a, b):
    """Edit distance."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def clean(pairs):
    """(codes, text read) without spaces, the text decomposed (a two-part vowel sign is two glyphs), a glyph set k times over itself (a faked bold) counted once; a cropped
    picture or a joined second line does not fit its codes and is dropped."""
    out = []
    for c, e in pairs:
        c, e = re.sub(r"\s", "", c), unicodedata.normalize("NFD", re.sub(r"\s", "", e))
        runs = {len(m.group()) for m in re.finditer(r"(.)\1*", c)}
        if len(runs) == 1 and min(runs) >= 2:
            c = c[::runs.pop()]
        if 3 <= len(c) <= 90 and e and 0.4 <= len(e) / len(c) <= 2.5:
            out.append((c, e))
    return out


def script_of(texts, default):
    """The script of the text read (a family list may file a Bengali font among the Devanagari ones)."""
    c = collections.Counter(script_name(ch) for t in texts for ch in t if ch.isalpha())
    top = c.most_common(1)[0][0] if c else default
    return top if top == "Latin" or top in set(L.LANG_SCRIPT.values()) else default


def is_unicode_font(pairs, share=0.9, minimum=4):
    """True when the text read is the typed codes themselves (median share of the codes found in the text): a font of real Unicode, Latin or CJK, that needs no table."""
    seen = sorted(sum(ch in e for ch in c if not ch.isspace()) / max(sum(not ch.isspace() for ch in c), 1) for c, e in pairs)
    return len(seen) >= minimum and seen[len(seen) // 2] >= share


def is_symbol_font(pairs):
    """True when most spans are replacement characters: a symbol font without a text map, its text cannot be recovered."""
    return len(pairs) >= 2 and sum(c.count("�") > len(c) / 2 for c, _ in pairs) > len(pairs) / 2


# ------------------------------------------------------------------------------------------------------------------------------------------------- the fit
def model1(pairs):
    """{code: [characters, most probable first]}: IBM model 1 between the codes of a span and the characters read from it."""
    t = collections.defaultdict(lambda: 1e-9)
    for c, e in pairs:
        for ch in set(c):
            for x in set(e):
                t[(x, ch)] = 1.0
    for _ in range(MODEL1_ROUNDS):
        cnt, tot = collections.defaultdict(float), collections.defaultdict(float)
        for c, e in pairs:
            cc = list(set(c))
            for x in e:
                z = sum(t[(x, ch)] for ch in cc)
                for ch in cc:
                    cnt[(x, ch)] += t[(x, ch)] / z
                    tot[ch] += t[(x, ch)] / z
        t = collections.defaultdict(lambda: 1e-9, {k: v / tot[k[1]] for k, v in cnt.items()})
    cand = collections.defaultdict(list)
    for (x, ch), p in t.items():
        if p >= MODEL1_MIN_P:
            cand[ch].append((p, x))
    return {ch: [x for _, x in sorted(v, reverse=True)[:CANDIDATES]] for ch, v in cand.items()}


def pieces(tab, codes, script):
    """The text of typed codes under a table, in logical order."""
    return T.sequence([tab.get(ch, ch) for ch in codes], script)


def ngrams(pairs, tab, script, top=8):
    """Candidate texts per code: the 1-3 character pieces of the text read around the place the code sits at (length of the decode before it), by frequency."""
    cnt, base = collections.defaultdict(collections.Counter), collections.Counter()
    for c, e in pairs:
        m, tot, pre = len(e), max(len(pieces(tab, c, script)), 1), 0
        for ch in c:
            w = len(tab.get(ch, ch))
            mid, pre = (pre + w / 2) / tot * m, pre + w
            if script == "Arabic":
                mid = m - mid                                         # the glyphs run left to right, the text is read right to left
            for a in range(max(int(mid - NGRAM_WINDOW), 0), min(int(mid + NGRAM_WINDOW + 1), m)):
                for k in (1, 2, 3):
                    if len(e[a:a + k]) == k:
                        cnt[ch][e[a:a + k]] += 1
        for a in range(m):
            for k in (1, 2, 3):
                if a + k <= m:
                    base[e[a:a + k]] += 1
    n = len(pairs)
    return {ch: [g for g, _ in sorted(((g, v / (1 + base[g] / n)) for g, v in cc.items() if v >= 3), key=lambda t: -t[1])[:top]] for ch, cc in cnt.items()}


def fit(pairs, script, passes=2):
    """{code: text} minimizing the edit distance between the decode of every span and the text read from it (coordinate descent over the candidates of a code: model 1
    characters, pieces around its place, with a virama, nothing, itself)."""
    cand, vir = model1(pairs), T.virama(script)
    tab = {ch: (v[0] if v else "") for ch, v in cand.items()}
    where = collections.defaultdict(list)
    for i, (c, _) in enumerate(pairs):
        for ch in set(c):
            where[ch].append(i)
    for _ in range(passes):
        ng = ngrams(pairs, tab, script)
        for ch in sorted(where, key=lambda k: -len(where[k])):
            base = cand.get(ch, []) + ng.get(ch, [])[:6]
            best = None
            for o in {"", ch} | set(base) | {x + vir for x in base if vir}:
                tab[ch] = o
                cost = sum(lev(pieces(tab, pairs[i][0], script), pairs[i][1]) for i in where[ch][:30])
                if best is None or cost < best[0]:
                    best = (cost, o)
            tab[ch] = best[1]
    return tab


def table(pairs, script):
    """{code: text} of the codes two fits, on the even and on the odd spans, agree on (the precision gate), written in the script, seen on >= MIN_LINES spans."""
    a, b = fit(pairs[0::2], script), fit(pairs[1::2], script)
    lines = collections.Counter(ch for c, _ in pairs for ch in set(c))
    return {ch: nfc(a[ch]) for ch in a if ch in b and a[ch] == b[ch] and a[ch] != ch and lines[ch] >= MIN_LINES
            and any(in_script(x, script) for x in a[ch])}


# ------------------------------------------------------------------------------------------------------------------------------------------------- published tables
def published(tab, decoders, min_codes=8, share=0.9):
    """(agreement, 'kind:name') of the published table that gives the text of >= share of the codes of tab (>= min_codes of them), else None."""
    best = None
    if len(tab) < min_codes:
        return None
    for k, f in decoders.items():
        texts = {ch: _safe(f, ch) for ch in tab}
        ok = sum(t is not None and nfc(t) == nfc(tab[ch]) for ch, t in texts.items())
        if ok / len(tab) >= share and (best is None or ok / len(tab) > best[0]):
            best = (ok / len(tab), k)
    return best


def _safe(f, text):
    try:
        return f(text)
    except Exception:
        return None


@functools.lru_cache(maxsize=None)
def output_scripts(key, f):
    """The scripts a decoder writes (its text for the Latin and Latin-1 letters)."""
    c = collections.Counter(script_name(ch) for ch in (_safe(f, PROBE) or "") if ch.isalpha())
    return frozenset(sc for sc, n in c.items() if n >= 0.3 * max(sum(c.values()), 1))


def explain(lines, script, decoders, lm):
    """(mean log P per character, 'kind:name') of the published table under which the typed lines read best as the language, if that is >= LM_MARGIN_TYPED above the text as
    typed, within LM_MARGIN_REAL of what the model gives real text, and >= 80 % of the letters are in the script; else None. The language model decides, nothing is read."""
    lines = [c for c in lines if len(c.strip()) >= 8][:40]
    if len(lines) < 5 or not lm.real:
        return None
    typed, best = sum(lm.mean(t) for t in lines) / len(lines), None
    for k, f in decoders.items():
        if script not in output_scripts(k, f):
            continue
        out = [_safe(f, c) for c in lines]
        if None in out:
            continue
        letters = [ch for t in out for ch in t if ch.isalpha()]
        if len(letters) < 40 or sum(in_script(ch, script) for ch in letters) < 0.8 * len(letters):
            continue
        m = sum(lm.mean(t) for t in out) / len(out)
        if m >= typed + LM_MARGIN_TYPED and m >= lm.real - LM_MARGIN_REAL and (best is None or m > best[0]):
            best = (m, k)
    return best


def has_high_codes(lines, share=0.05):
    """True when the typed lines hold bytes of a legacy encoding: characters of Latin-1 above ASCII or private use, >= share of the characters. A legacy CJK font (Big5, GBK, Shift-JIS,
    EUC-KR) cannot do without them; a font of ASCII, punctuation or real CJK text has none and needs no table."""
    chars = [c for line in lines for c in line if not c.isspace()]
    return bool(chars) and sum(0x80 <= ord(c) <= 0xFF or 0xE000 <= ord(c) <= 0xF8FF for c in chars) >= share * len(chars)


def is_english(lines, share=0.5):
    """True when most words of the typed lines are English words (wordfreq): a Latin font of real text among documents of another language."""
    from wordfreq import zipf_frequency
    words = [w for c in lines for w in c.split() if len(w) > 2 and w.isascii() and w.isalpha()]
    return len(words) >= 20 and sum(zipf_frequency(w, "en") >= 3 for w in words) >= share * len(words)


def is_native(lines, lm, margin=1.5):
    """True when the typed lines already read as the language (mean log P per character within `margin` of real text): a font of real Unicode, no table may touch it."""
    lines = [c for c in lines if len(c.strip()) >= 8][:60]
    return len(lines) >= 5 and bool(lm.real) and sum(lm.mean(c) for c in lines) / len(lines) >= lm.real - margin


def reads_as_language(tab, lines, script, lm):
    """(decoded, typed, real): mean log P per character under the language model of the lines decoded with the table, of the lines as typed, and of real text of the language.
    A table recognizes its font when `decoded` is far above `typed` and near `real`."""
    lines = [c for c in lines if len(c.strip()) >= 8][:60]
    if len(lines) < 5 or not lm.real:
        return None
    mean = lambda f: sum(lm.mean(f(c)) for c in lines) / len(lines)
    return mean(lambda c: pieces(tab, c, script)), mean(lambda c: c), lm.real


# ------------------------------------------------------------------------------------------------------------------------------------------------- language model
class CharLM:
    """Character trigram, interpolated (Witten-Bell), trained on the real paragraphs of the language (legacy.PARAS)."""
    def __init__(self, lang, cap=2_000_000):
        text = ""
        f = os.path.join(L.PARAS, f"{lang}.json")
        if os.path.exists(f):
            for v in json.load(open(f, encoding="utf-8")).values():
                text += "\n".join(t for t in v if isinstance(t, str)) + "\n"
                if len(text) > cap:
                    break
        self.c, self.ctx = [collections.Counter() for _ in range(3)], [collections.Counter() for _ in range(3)]       # n-gram and context counts, n = 1..3
        self.kinds = [collections.defaultdict(set) for _ in range(3)]
        for line in text.split("\n"):
            line = "^^" + nfc(line) + "$"
            for i in range(2, len(line)):
                for n in (1, 2, 3):
                    g = line[i - n + 1:i + 1]
                    self.c[n - 1][g] += 1
                    self.ctx[n - 1][g[:-1]] += 1
                    self.kinds[n - 1][g[:-1]].add(g[-1])
        self.V, self.memo = max(len(self.c[0]), 1), {}
        lines = [t for t in text.split("\n") if len(t) > 20][:300]
        self.real = sum(self.mean(t) for t in lines) / max(len(lines), 1)            # what real text of the language scores per character

    def p(self, ctx, ch):
        pr = 1.0 / (self.V + 1)
        for n in (1, 2, 3):
            h = ctx[len(ctx) - (n - 1):] if n > 1 else ""
            tot = self.ctx[n - 1][h]
            if tot:
                lam = tot / (tot + len(self.kinds[n - 1][h]))
                pr = lam * self.c[n - 1][h + ch] / tot + (1 - lam) * pr
        return pr

    def score(self, ctx, s):
        """log P(s | the two characters before it)."""
        if (ctx, s) not in self.memo:
            tot, h = 0.0, ctx
            for ch in s:
                tot += math.log(self.p(h[-2:], ch))
                h += ch
            self.memo[(ctx, s)] = tot
        return self.memo[(ctx, s)]

    def mean(self, text):
        """Mean log P per character of a text."""
        return self.score("^^", nfc(text)) / max(len(text), 1)


@functools.lru_cache(maxsize=8)
def char_lm(lang):
    return CharLM(lang)


# ------------------------------------------------------------------------------------------------------------------------------------------------- decipherment
def logsum(terms):
    m = max(terms)
    return m + math.log(sum(math.exp(t - m) for t in terms))


def posteriors(codes, opts, prior, lm):
    """{code: {text: expected count}}: forward-backward over the lattice of candidate texts of one line of typed codes (the language model scores the path)."""
    n, O = len(codes), [opts.get(ch, [ch]) for ch in codes]
    lp = lambda i, o: math.log(prior.get(codes[i], {}).get(o, 1.0))
    tr = lambda q, o: lm.score(q[-2:].rjust(2, "^"), o)
    fw = [{o: lp(0, o) + lm.score("^^", o) for o in O[0]}]
    for i in range(1, n):
        fw.append({o: logsum([fw[i - 1][q] + tr(q, o) for q in O[i - 1]]) + lp(i, o) for o in O[i]})
    bw = [None] * n
    bw[n - 1] = {q: tr(q, "$") for q in O[n - 1]}
    for i in range(n - 2, -1, -1):
        bw[i] = {q: logsum([tr(q, o) + lp(i + 1, o) + bw[i + 1][o] for o in O[i + 1]]) for q in O[i]}
    z = logsum([fw[n - 1][q] + bw[n - 1][q] for q in O[n - 1]])
    return [(codes[i], o, math.exp(fw[i][o] + bw[i][o] - z)) for i in range(n) for o in O[i]]


def em(lines, opts, prior, lm, iters=EM_ITERATIONS):
    """prior {code: {text: P(text | code)}} after EM over the lines."""
    for _ in range(iters):
        gam = collections.defaultdict(lambda: collections.defaultdict(float))
        for c in lines:
            for ch, o, p in posteriors(c, opts, prior, lm):
                gam[ch][o] += p
        for ch, g in gam.items():
            if ch in opts:
                t = sum(g.values()) + 1e-9
                prior[ch] = {o: (g.get(o, 0.0) + 0.01) / (t + 0.01 * len(opts[ch])) for o in opts[ch]}
    return prior


def decipher(tab, pairs, lines, script, lm):
    """tab plus the codes noisy-channel decipherment adds: the hidden text is a path of candidate texts, one per typed code, scored by the language model; EM re-estimates
    P(text | code) from the start the pictures gave. A code the table lacks is added when its posterior is >= POSTERIOR on the even and on the odd lines alike and it is also the
    first candidate of model 1 (two independent evidences; 98-100 % right on 3 fonts with known tables)."""
    cand, ng = model1(pairs), ngrams(pairs, tab, script)
    opts = {ch: list(dict.fromkeys(x for x in ([tab[ch]] if ch in tab else []) + cand.get(ch, []) + ng.get(ch, [])[:6] + [ch] if x)) for ch in set(cand) | set(ng) | set(tab)}
    opts[" "] = [" "]
    lines = [c for c in lines if len(c) >= 3][:300]
    best = []
    for part in (0, 1):
        prior = {ch: {o: ((PRIOR_START if o == tab[ch] else (1 - PRIOR_START) / max(len(os_) - 1, 1)) if ch in tab else 1.0 / len(os_)) for o in os_} for ch, os_ in opts.items()}
        prior = em(lines[part::2], opts, prior, lm)
        best.append({ch: max(pr.items(), key=lambda kv: kv[1]) for ch, pr in prior.items()})
    out = dict(tab)
    for ch in opts:
        if ch not in tab and ch != " " and ch in best[0] and ch in best[1]:
            (a, pa), (b, pb) = best[0][ch], best[1][ch]
            if a == b != ch and min(pa, pb) >= POSTERIOR and cand.get(ch, [None])[0] == a:
                out[ch] = a
    return out


# ------------------------------------------------------------------------------------------------------------------------------------------------- checks of a decode
def invalid(text, script):
    """(clusters that break the script's syllable rules, clusters) of a decoded text (Unicode Indic syllable structure: a dependent sign needs a letter before it,
    one vowel sign per syllable, nothing but a letter or ZWJ / ZWNJ after a virama, one nasal sign); 0 for a script without dependent signs."""
    import regex
    vir = T.virama(script)
    bad = 0
    clusters = regex.findall(r"\X", nfc(text))
    for k, c in enumerate(clusters):
        marks = [ch for ch in c if unicodedata.category(ch) in ("Mn", "Mc")]
        if not marks:
            continue
        base = c[0]
        vowels = [ch for ch in marks if "VOWEL SIGN" in unicodedata.name(ch, "")]
        nasal = [ch for ch in marks if unicodedata.name(ch, "").endswith(("ANUSVARA", "CANDRABINDU"))]
        bad += (unicodedata.category(base) in ("Mn", "Mc") or base.isspace() or len(vowels) > 1 or len(nasal) > 1
                or bool(vir) and any(a == vir and unicodedata.category(b) in ("Mn", "Mc") for a, b in zip(c, c[1:])))
    return bad, len(clusters)
