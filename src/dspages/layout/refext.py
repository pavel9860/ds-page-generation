"""Reference texts a Noto font has no glyph of its own for: the glyphs HarfBuzz makes for letters, vowel signs, half forms and conjuncts.

legacy.reference() knows the glyphs of a Noto font that its cmap and GSUB substitutions map to a text. A legacy font also has the parts of a letter
(the vowel sign left of its letter, the lower half of a conjunct, the stroke under a letter): those glyphs have no text there. Shaping every letter,
letter + sign, letter + virama and letter + virama + letter with HarfBuzz at character level names each output glyph by the shortest text that made it;
the new texts (with the shape of that glyph) are added to the reference, cached beside it."""
import os
import unicodedata as ud

import numpy as np
import uharfbuzz as hb
from fontTools.ttLib import TTFont

from . import legacy


def _texts(script):
    lo, hi = legacy.SPECS[script][1][0]
    ch = [chr(c) for c in range(lo, hi + 1) if ud.category(chr(c)) not in ("Cn", "Cc")]
    base = [c for c in ch if ud.category(c) == "Lo"]
    vir = [c for c in ch if ud.combining(c) == 9]
    marks = [c for c in ch if ud.category(c) in ("Mn", "Mc") and ud.combining(c) != 9]
    out = base + marks + ["◌" + m for m in marks]
    for b in base:
        out += [b + m for m in marks] + [b + v for v in vir]
    out += [b + v + b2 for v in vir for b in base for b2 in base]
    return list(dict.fromkeys(out))


def reference_ext(script):
    """(texts, shapes, geometries): legacy.reference(script) followed by the texts shaping adds. Cached in the legacy cache folder."""
    f = os.path.join(legacy.CACHE, f"refext_{script}.npz")
    if os.path.exists(f):
        z = np.load(f, allow_pickle=True)
        return list(z["T"]), z["S"], z["G"]
    T0, S0, G0 = legacy.reference(script)
    if script not in legacy.SPECS or script.startswith("CJK-"):
        return _with_latin(script, list(T0), np.asarray(S0), np.asarray(G0), f)
    fp = os.path.join(legacy.FONTS, legacy.SPECS[script][0])
    font = hb.Font(hb.Face(hb.Blob.from_file_path(fp)))
    tt = TTFont(fp)
    order, gs, upem = tt.getGlyphOrder(), tt.getGlyphSet(), tt["head"].unitsPerEm
    best = {}
    for t in _texts(script):
        buf = hb.Buffer()
        buf.cluster_level = hb.BufferClusterLevel.CHARACTERS
        buf.add_str(t)
        buf.guess_segment_properties()
        hb.shape(font, buf, {})
        cl = sorted({i.cluster for i in buf.glyph_infos})
        for i in buf.glyph_infos:
            nx = [c for c in cl if c > i.cluster]
            piece = t[i.cluster:nx[0] if nx else len(t)]
            if piece and i.codepoint and (i.codepoint not in best or len(piece) < len(best[i.codepoint])):
                best[i.codepoint] = piece
    T, S, G = list(T0), list(S0), list(G0)
    seen = {ud.normalize("NFC", t) for t in T}
    for g, piece in sorted(best.items()):
        p = ud.normalize("NFC", piece)
        if p in seen:
            continue
        r = legacy.feature(gs, order[g], upem)
        if r is not None:
            T.append(piece)
            S.append(r[0])
            G.append(r[1])
            seen.add(p)
    return _with_latin(script, T, np.asarray(S, np.float32), np.asarray(G, np.float32), f)


LATIN_ADD = [chr(c) for c in range(0x21, 0x7F)] + list("\u2018\u2019\u201c\u201d\u2013\u2014\u2026\u2022\u00b7\u00a9\u00ae\u00b0\u00d7\u00f7\u00a1\u00bf\u00ab\u00bb")


def _with_latin(script, T, S, G, f):
    """A legacy font also draws Latin letters, digits and punctuation (a Korean or Chinese font's English text, the numbers and marks of an Indic one):
    the Latin shapes are added to every script's reference; saved to `f` (the cache)."""
    if script != "Latin":
        TL, SL, GL = legacy.reference("Latin")
        have = {ud.normalize("NFC", t) for t in T}
        pick = [i for i, t in enumerate(TL) if t in LATIN_ADD and ud.normalize("NFC", t) not in have]
        T = list(T) + [TL[i] for i in pick]
        S = np.concatenate([np.asarray(S, np.float32).reshape(-1, np.asarray(SL).shape[1]), np.asarray(SL, np.float32)[pick]])
        G = np.concatenate([np.asarray(G, np.float32).reshape(-1, np.asarray(GL).shape[1]), np.asarray(GL, np.float32)[pick]])
    np.savez(f, T=np.array(T, dtype=object), S=np.asarray(S, np.float32), G=np.asarray(G, np.float32))
    return T, np.asarray(S, np.float32), np.asarray(G, np.float32)


def candidates(feats, script, k=8):
    """legacy.candidates against the extended reference: {glyph id: [(text, distance)]}."""
    T, S, G = reference_ext(script)
    out = {}
    gids = list(feats)
    for a in range(0, len(gids), 48):
        blk = gids[a:a + 48]
        d = legacy._dist(np.array([feats[g][0] for g in blk], np.float32), np.array([feats[g][1] for g in blk], np.float32), S, G)
        for row, g in enumerate(blk):
            seen, cand = set(), []
            for i in np.argsort(d[row])[:4 * k]:
                if T[i] not in seen:
                    seen.add(T[i])
                    cand.append((T[i], float(d[row][i])))
                    if len(cand) == k:
                        break
            out[g] = cand
    return out


INDIC = ["Devanagari", "Bengali", "Gurmukhi", "Gujarati", "Oriya", "Tamil", "Telugu", "Kannada", "Malayalam", "Sinhala"]
SIMILAR_COST = 0.4                                         # distance added to a match found in another script's reference


def _transpose(text, src, dst):
    """The same text in another Indic script by character name (TELUGU LETTER KA -> KANNADA LETTER KA); None when one character has no counterpart."""
    head, out = src.upper() + " ", []
    for ch in text:
        if ord(ch) < 0x0900 or ch in "\u200c\u200d":
            out.append(ch)
            continue
        try:
            nm = ud.name(ch)
            if not nm.startswith(head):
                return None
            out.append(ud.lookup(dst.upper() + " " + nm[len(head):]))
        except (ValueError, KeyError):
            return None
    return "".join(out)


def candidates_similar(feats, script, k=8):
    """The nearest texts of the glyph in the script's own extended reference and, for an Indic script, in the references of the sister scripts
    (the same texts by character name): {glyph id: [(text, distance)]}, nearest first. A glyph a font cut from a letter may look like the part of
    a sister script's letter that the own script's reference does not have."""
    own = candidates(feats, script, k=k)
    if script not in INDIC:
        return own
    out = {g: list(v) for g, v in own.items()}
    for src in INDIC:
        if src == script:
            continue
        for g, lst in candidates(feats, src, k=k).items():
            for t, d in lst:
                tt = _transpose(t, src, script)
                if tt:
                    out[g].append((tt, d + SIMILAR_COST))
    res = {}
    for g, lst in out.items():
        seen, best = set(), []
        for t, d in sorted(lst, key=lambda x: x[1]):
            n = ud.normalize("NFC", t)
            if n not in seen:
                seen.add(n)
                best.append((t, d))
        res[g] = best[:k]
    return res


def line_candidates(feat, script, text, n=3, longest=4):
    """The n texts nearest to the glyph's shape among the substrings (1..longest characters) of the text of its line: the line's reading narrows the
    candidates to what the line can contain. Only substrings the script's extended reference has a shape for are tried."""
    T, S, G = reference_ext(script)
    idx = {}
    for i, t in enumerate(T):
        idx.setdefault(ud.normalize("NFC", t), i)
    text = ud.normalize("NFC", text)
    ids = {idx[text[a:a + m]] for a in range(len(text)) for m in range(1, longest + 1) if text[a:a + m] in idx}
    if not ids:
        return []
    ids = sorted(ids)
    d = legacy._dist(np.asarray(feat[0], np.float32)[None], np.asarray(feat[1], np.float32)[None], np.asarray(S)[ids], np.asarray(G)[ids])[0]
    return [T[ids[j]] for j in np.argsort(d)[:n]]
