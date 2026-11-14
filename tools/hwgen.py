"""Natural handwriting in any script: glyph centrelines traced from handwriting fonts (Noto for scripts without one),
rewritten by a writer model (slant, size, spacing, baseline, elastic shape drift, wobble, tremor, rounding, cursive joins,
end-of-word decay, hooks, flicks, displaced dots, cross-outs, overwrites) and drawn by a pen model (ballpoint, gel,
fountain, pencil, felt tip: speed and pressure set width and ink, overlaps, skips, pooling, graphite grain).
Writers (gene vectors in [0, 1]) are evolved for realism (typicality among real handwritten lines cut from the
handwritten folder, in stroke features and ResNet18 embeddings) and novelty; the archive is the writer bank.

python tools/hwgen.py demo [--bank]           example sentences in many languages
python tools/hwgen.py refs                    real handwritten line references (real_lines.npz)
python tools/hwgen.py evolve [--gens 40]      evolves the writer bank (writers.json)
"""
import argparse
import json
import os
import pickle
import random
import unicodedata
from collections import Counter
from functools import lru_cache
from glob import glob

import cv2
import numpy as np
import regex
from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import skeletonize

from dspages.config import Paths

UVTM = os.environ.get("DSPAGES_UVTM", "/run/media/p/D/ML_DS/UVTM")
HWFONTS = f"{UVTM}/hwfonts"
NOTO = ("/usr/share/fonts/truetype/noto", "/usr/share/fonts/opentype/noto", "/usr/share/fonts/truetype/dejavu")
OUT = f"{Paths().layouts}/test/hw_gen"
BANK = f"{OUT}/writers.json"
REFS = f"{OUT}/real_lines.npz"
PX = 160
WORDLEVEL = {"ARABIC", "SYRIAC", "THAANA", "NKO", "MONGOLIAN", "DEVANAGARI", "BENGALI", "GURMUKHI", "GUJARATI", "ORIYA",
             "TAMIL", "TELUGU", "KANNADA", "MALAYALAM", "SINHALA", "THAI", "LAO", "KHMER", "MYANMAR", "TIBETAN"}
NOSPACE = {"HANI", "KANA", "THAI", "LAO", "KHMER", "MYANMAR", "TIBETAN"}
RTL = {"ARABIC", "HEBREW", "SYRIAC", "THAANA", "NKO"}
REF = {"LATIN": ("x", 1.0), "CYRILLIC": ("х", 1.0), "GREEK": ("ο", 1.0), "ARMENIAN": ("ս", 1.0),
       "GEORGIAN": ("ა", 1.0), "HEBREW": ("ם", 1.15), "ARABIC": ("ن", 1.3), "DEVANAGARI": ("क", 1.5),
       "BENGALI": ("ক", 1.5), "GUJARATI": ("ક", 1.3), "GURMUKHI": ("ਕ", 1.5), "TAMIL": ("க", 1.2), "TELUGU": ("క", 1.4),
       "KANNADA": ("ಕ", 1.4), "MALAYALAM": ("ക", 1.2), "THAI": ("ก", 1.15), "HANI": ("国", 2.0), "KANA": ("あ", 1.8),
       "HANGUL": ("한", 2.0), "ETHIOPIC": ("ሀ", 1.4), "CHEROKEE": ("Ꮳ", 1.6), "CANADIAN": ("ᐃ", 1.3)}
SKIP = ("AmaticSC", "PermanentMarker", "WalterTurncoat", "RockSalt", "Comforter", "Ruthie")
OWN = {"HachiMaruPop": "KANA", "Yomogi": "KANA", "KleeOne": "HANI", "ZenKurenaido": "KANA", "GamjaFlower": "HANGUL",
       "HiMelody": "HANGUL", "Gaegu": "HANGUL", "Nanum": "HANGUL", "MaShanZheng": "HANI", "ZhiMangXing": "HANI",
       "LongCang": "HANI", "LiuJianMaoCao": "HANI", "PlaypenSansArabic": "ARABIC", "PlaypenSansDeva": "DEVANAGARI",
       "PlaypenSansHebrew": "HEBREW", "PlaypenSansThai": "THAI", "ArefRuqaa": "ARABIC", "Gulzar": "ARABIC",
       "Galada": "BENGALI", "Farsan": "GUJARATI", "Gurajada": "TELUGU", "Kavivanar": "TAMIL", "YatraOne": "DEVANAGARI"}
PREFER = {"HEBREW": (("GveretLevin",), 0.8), "CYRILLIC": (("BadScript", "MarckScript", "Caveat"), 0.7)}
JOINABLE = {"LATIN", "CYRILLIC", "GREEK", "ARMENIAN", "GEORGIAN"}
PENS = ("ballpoint", "ballpoint", "ballpoint", "gel", "gel", "fountain", "pencil", "felt")
INKS = ((0.16, 0.24, 0.62), (0.16, 0.24, 0.62), (0.10, 0.13, 0.38), (0.09, 0.09, 0.11), (0.09, 0.09, 0.11),
        (0.30, 0.22, 0.55), (0.10, 0.18, 0.45), (0.72, 0.13, 0.13), (0.12, 0.42, 0.24))
GENES = (("size", 1.9, 4.0), ("xs", 0.78, 1.25), ("slant", -12, 30), ("slant_sd", 0, 6), ("size_sd", 0, 0.14),
         ("rot_sd", 0, 4), ("base_sd", 0, 0.14), ("ls", -0.08, 0.22), ("ls_sd", 0, 0.12), ("ws", 0.4, 0.95),
         ("pitch", 2.5, 4.0), ("slope", -2.5, 2.5), ("bow", -0.5, 0.5), ("elastic", 0, 1), ("wobble", 0, 1),
         ("tremor", 0, 1), ("round", 0, 1), ("connect", 0, 1), ("trail", 0, 1), ("hook", 0, 1), ("flick", 0, 1),
         ("dot_dx", -0.1, 0.4), ("dot_len", 0, 1), ("pen", 0, 1), ("pen_w", 0, 1), ("press_sd", 0, 1), ("ink", 0, 1),
         ("style", 0, 1), ("cross", 0, 0.05), ("fix", 0, 0.03), ("drift", -0.25, 0.25), ("margin_sd", 0, 1),
         ("speed", 0, 1))


def decode(g):
    return {n: lo + (hi - lo) * float(v) for (n, lo, hi), v in zip(GENES, g)}


def script(ch):
    n = unicodedata.name(ch, "")
    w = n.split(" ")[0] if n else "?"
    if w in ("CJK", "KANGXI", "IDEOGRAPHIC"):
        return "HANI"
    if w in ("HIRAGANA", "KATAKANA"):
        return "KANA"
    return w if unicodedata.category(ch)[0] in "LM" else "COMMON"


def word_script(w):
    c = Counter(s for s in map(script, w) if s != "COMMON")
    return c.most_common(1)[0][0] if c else "COMMON"


@lru_cache(None)
def cmaps():
    path = f"{HWFONTS}/_cmap.pkl"
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    from fontTools.ttLib import TTFont
    hw = sorted(glob(f"{HWFONTS}/*.ttf"))
    noto = sorted(f for d in NOTO for f in glob(f"{d}/*") if "Regular" in f or "dejavu/DejaVuSans.ttf" in f)
    out = {}
    for f in hw + noto:
        try:
            out[f] = (f in hw, frozenset(TTFont(f, fontNumber=0).getBestCmap()))
        except Exception:
            pass
    pickle.dump(out, open(path, "wb"))
    return out


@lru_cache(4096)
def fonts_for(chars):
    cs = [ord(c) for c in chars if not c.isspace()]
    cm = cmaps()
    own = {script(c) for c in chars} & set(OWN.values())
    ok = lambda f: not any(k in f for k in SKIP) and all(OWN[k] in own for k in OWN if k in f)
    hw = [f for f, (h, m) in cm.items() if h and ok(f) and all(c in m for c in cs)]
    return hw or [f for f, (h, m) in cm.items() if all(c in m for c in cs)]


@lru_cache(256)
def font(path):
    return ImageFont.truetype(path, PX, layout_engine=ImageFont.Layout.RAQM)


@lru_cache(4096)
def xh_ratio(path, scr):
    """Height of the script's reference letter over the x-height it is written at, per em."""
    ch, k = REF.get(scr, ("x", 1.0))
    f = font(path)
    if ord(ch) not in cmaps()[path][1]:
        ch, k = "x", 1.0
    if ord(ch) not in cmaps()[path][1]:
        return f.getmetrics()[0] / PX * 0.55
    b = f.getbbox(ch, anchor="ls")
    return max((b[3] - b[1]) / PX, 0.2) / k


def trace(b):
    sk = skeletonize(b)
    pts = set(zip(*np.nonzero(sk)))
    nbr = {p: [(p[0] + dy, p[1] + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy or dx) and (p[0] + dy, p[1] + dx) in pts]
           for p in pts}
    node = {p for p, n in nbr.items() if len(n) != 2}
    cl, cid = {}, 0
    for p in node:
        if p in cl:
            continue
        st = [p]
        cl[p] = cid
        while st:
            q = st.pop()
            for r in nbr[q]:
                if r in node and r not in cl:
                    cl[r] = cid
                    st.append(r)
        cid += 1
    centre = {}
    for p, c in cl.items():
        centre.setdefault(c, []).append(p)
    centre = {c: np.mean(v, 0) for c, v in centre.items()}
    used, edges = set(), []
    for p in node:
        for m in nbr[p]:
            if m in node or (p, m) in used:
                continue
            path, prev, cur = [p, m], p, m
            used.add((p, m))
            while cur not in node:
                nx = [r for r in nbr[cur] if r != prev and r not in path[-3:]]
                if not nx:
                    break
                prev, cur = cur, nx[0]
                path.append(cur)
            used.add((cur, prev))
            if cur in node:
                edges.append([cl[p], cl[cur], path])
    seen = {q for _, _, e in edges for q in e} | node
    for p in pts - seen:
        if p in seen:
            continue
        path, prev, cur = [p], None, p
        while True:
            seen.add(cur)
            nx = [r for r in nbr[cur] if r != prev and r not in seen]
            if not nx:
                break
            prev, cur = cur, nx[0]
            path.append(cur)
        path.append(p)
        edges.append([None, None, path])
    lone = set(cl.values()) - {c for a, b, _ in edges for c in (a, b)}
    edges += [[None, None, np.array([centre[c], centre[c] + 0.5])] for c in lone]
    spur = 0.05 * PX
    for _ in range(2):
        deg = Counter(c for a, b, _ in edges for c in (a, b) if c is not None)
        edges = [e for e in edges if not (len(e[2]) < spur and e[0] is not None and
                                          ((deg[e[0]] == 1 and deg[e[1]] >= 3) or (deg[e[1]] == 1 and deg[e[0]] >= 3)))]
    out = []
    for a, b, p in edges:
        p = np.array(p, float)[:, ::-1]
        if a is not None and deg_ok(a, edges):
            p = np.vstack([centre[a][::-1], p])
        if b is not None and deg_ok(b, edges):
            p = np.vstack([p, centre[b][::-1]])
        out.append([a, b, p])
    return chain(out)


def deg_ok(c, edges):
    return sum((a == c) + (b == c) for a, b, _ in edges) >= 2


def chain(edges):
    def tang(e, end):
        p = e[2] if end else e[2][::-1]
        k = min(len(p) - 1, 8)
        d = p[-1 - k] - p[-1]
        return d / (np.hypot(*d) + 1e-9)
    ends = {}
    for i, (a, b, _) in enumerate(edges):
        for end, c in ((0, a), (1, b)):
            if c is not None:
                ends.setdefault(c, []).append((i, end))
    pair = {}
    for c, es in ends.items():
        cand = sorted((float(tang(edges[i], e) @ tang(edges[j], f)), (i, e), (j, f))
                      for x, (i, e) in enumerate(es) for (j, f) in es[x + 1:] if i != j)
        for d, u, v in cand:
            if d < -0.5 and u not in pair and v not in pair:
                pair[u], pair[v] = v, u
    done, strokes = set(), []
    order = [(i, e) for i in range(len(edges)) for e in (0, 1) if (i, e) not in pair] + [(i, 0) for i in range(len(edges))]
    for i, e in order:
        if i in done:
            continue
        seq = []
        while i not in done:
            done.add(i)
            p = edges[i][2]
            seq.append(p if e == 0 else p[::-1])
            nx = pair.get((i, 1 - e))
            if nx is None:
                break
            i, e = nx
        strokes.append(np.vstack(seq))
    return strokes


def resample(p, step):
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    if d[-1] < step:
        return p[[0, -1]] if len(p) > 1 else np.vstack([p, p])
    s = np.linspace(0, d[-1], int(d[-1] / step) + 1)
    return np.c_[np.interp(s, d, p[:, 0]), np.interp(s, d, p[:, 1])]


def smooth(p, sig):
    if sig < 0.3 or len(p) < 4:
        return p
    closed = len(p) > 8 and np.hypot(*(p[0] - p[-1])) < 1e-6
    k = np.exp(-0.5 * (np.arange(-int(3 * sig), int(3 * sig) + 1) / sig) ** 2)
    k /= k.sum()
    q = np.pad(p, ((len(k) // 2,), (0,)), mode="wrap" if closed else "edge")
    o = np.c_[np.convolve(q[:, 0], k, "valid"), np.convolve(q[:, 1], k, "valid")]
    if not closed:
        o[0], o[-1] = p[0], p[-1]
    return o


@lru_cache(200000)
def glyph(path, unit):
    f = font(path)
    asc, desc = f.getmetrics()
    w = f.getlength(unit)
    pad = PX // 2
    im = Image.new("L", (int(w) + 2 * pad, asc + desc + 2 * pad), 0)
    ImageDraw.Draw(im).text((pad, pad + asc), unit, fill=255, font=f, anchor="ls")
    b = np.array(im) > 110
    if b.sum() < 3:
        return (), w / PX
    strokes = []
    for s in trace(b):
        s = smooth(resample(s, 1.5), 2.0)
        s = (s - (pad, pad + asc)) / PX
        strokes.append(resample(s, 0.02))
    strokes.sort(key=lambda s: s[:, 0].min())
    return tuple(s if s[0] @ (1, 0.6) <= s[-1] @ (1, 0.6) else s[::-1] for s in strokes), w / PX


class Field:
    """Smooth random 2D displacement (sum of a few plane waves)."""

    def __init__(self, rng, amp, wl, n=4):
        a = rng.uniform(0, 2 * np.pi, n)
        self.k = np.c_[np.cos(a), np.sin(a)] * (2 * np.pi / rng.uniform(wl, 3 * wl, n))[:, None]
        self.ph = rng.uniform(0, 2 * np.pi, (n, 2))
        self.a = amp / np.sqrt(n)

    def __call__(self, p):
        t = p @ self.k.T
        return p + self.a * np.c_[np.sin(t + self.ph[:, 0]).sum(1), np.sin(t + self.ph[:, 1]).sum(1)]


def noise1d(rng, n, corr, amp):
    if n < 2:
        return np.zeros(n)
    m = max(n, int(20 * corr) + 50)
    k = np.exp(-0.5 * (np.arange(-int(3 * corr) - 1, int(3 * corr) + 2) / max(corr, 0.3)) ** 2)
    x = np.convolve(rng.normal(0, 1, m + len(k) - 1), k, "valid")
    return amp * x[:n] / (x.std() + 1e-9)


class Writer:
    def __init__(self, genes, seed=0):
        self.g = np.asarray(genes, float)
        self.p = decode(self.g)
        self.rng = np.random.default_rng(seed)
        p = self.p
        self.pen = PENS[min(int(p["pen"] * len(PENS)), len(PENS) - 1)]
        self.ink = (0.27, 0.27, 0.29) if self.pen == "pencil" else INKS[min(int(p["ink"] * len(INKS)), len(INKS) - 1)]
        w = p["pen_w"]
        self.width = {"ballpoint": 0.26 + 0.22 * w, "gel": 0.34 + 0.34 * w, "fountain": 0.10 + 0.12 * w,
                      "pencil": 0.28 + 0.3 * w, "felt": 0.55 + 0.5 * w}[self.pen]
        self.nib = (0.15 + 0.3 * w, self.rng.uniform(0.5, 1.0))

    def font(self, word, scr):
        fs = fonts_for(word)
        if not fs:
            return None
        h = (self.p["style"] + (sum(map(ord, scr)) % 97) * 0.6180339) % 1
        pref = [f for f in fs if any(k in f for k in PREFER.get(scr, ((), 0))[0])]
        if pref and h < PREFER[scr][1]:
            return pref[int(h / PREFER[scr][1] * len(pref))]
        return fs[int(h * len(fs))]

    def units(self, word, scr):
        if scr in WORDLEVEL:
            return [word]
        return regex.findall(r"\X", word)

    def word(self, word, em_xh, t):
        """Strokes of one word in mm, origin at its start on the baseline; returns (strokes, advance)."""
        p, rng = self.p, self.rng
        scr = word_script(word)
        rtl = scr in RTL
        size = em_xh * (1 + p["drift"] * (t - 0.5) * 2) * (1 + rng.normal(0, p["size_sd"] * 0.4))
        units = self.units(word, scr)
        if rtl and scr not in WORDLEVEL:
            units = units[::-1]
        x, out, prev = 0.0, [], None
        n = len(units)
        sl0 = np.radians(p["slant"] + rng.normal(0, p["slant_sd"] * 0.5))
        for k, u in enumerate(units):
            f = self.font(u, scr) or self.font(word, scr)
            if f is None:
                continue
            strokes, adv = glyph(f, u)
            em = size / xh_ratio(f, scr)
            kk = (n - 1 - k) if rtl else k
            decay = 1 - p["trail"] * 0.3 * max(0, kk / max(n - 1, 1) - 0.4) / 0.6 if n > 3 else 1
            sx = p["xs"] * (1 + rng.normal(0, p["size_sd"])) * (1 + 0.3 * (1 - decay))
            sy = (1 + rng.normal(0, p["size_sd"])) * decay
            sl = sl0 + np.radians(rng.normal(0, p["slant_sd"]))
            rot = np.radians(rng.normal(0, p["rot_sd"]))
            dy = rng.normal(0, p["base_sd"]) * size
            A = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]]) @ np.array([[1, -np.tan(sl)], [0, 1]])
            A = A @ np.diag([sx * em, sy * em])
            gs = []
            for s in strokes:
                s = self.shape(s, em)
                L = np.hypot(*(s.max(0) - s.min(0)))
                q = s @ A.T + (x, dy)
                dot = L < 0.1 and len(strokes) > 1
                if dot and scr in JOINABLE:
                    q = self.dot(q, size, sl)
                gs.append(dict(p=q, dot=dot))
            if prev is not None and scr in JOINABLE and u.isalpha() and prev[1].isalpha() and rng.random() < p["connect"]:
                self.join(prev[0], gs, size)
            out += gs
            prev = (gs, u)
            x += adv * em * sx + (p["ls"] + rng.normal(0, p["ls_sd"])) * size * 2
        strokes = [s for s in out if s.get("p") is not None]
        return strokes, x

    def shape(self, s, em):
        p, rng = self.p, self.rng
        s = s + noise1d(rng, len(s), 8, p["wobble"] * 0.018)[:, None] * rng.normal(0, 1, 2)
        s = smooth(s, p["round"] * 3.5 + p["speed"] * 1.5)
        if len(s) > 6 and rng.random() < p["hook"] * 0.5:
            d = s[0] - s[3]
            d = d / (np.hypot(*d) + 1e-9)
            h = 0.03 + 0.04 * rng.random()
            s = np.vstack([s[0] + h * np.array([d[1], -d[0]]) * rng.choice((-1, 1)) + d * h * 0.5, s])
        if len(s) > 6 and rng.random() < p["flick"] * 0.6:
            d = s[-1] - s[-4]
            d = d / (np.hypot(*d) + 1e-9)
            s = np.vstack([s, s[-1] + d * (0.04 + 0.12 * rng.random()) + (0, -0.02)])
        return s

    def dot(self, q, size, sl):
        p, rng = self.p, self.rng
        c = q.mean(0) + (p["dot_dx"] * size * rng.uniform(0.3, 1.5), rng.normal(0, 0.08) * size)
        ln = size * p["dot_len"] * rng.uniform(0, 0.35)
        a = np.radians(rng.uniform(-60, 20)) - sl * 0.5
        d = np.array([np.cos(a), np.sin(a)]) * ln / 2
        return np.vstack([c - d, c + d]) if ln > 0.02 else np.vstack([c, c + (0.01, 0.01)])

    def join(self, a, b, size):
        ea = [s for s in a if not s["dot"] and s["p"] is not None]
        eb = [s for s in b if not s["dot"] and s["p"] is not None]
        if not ea or not eb:
            return
        sa = max(ea, key=lambda s: max(s["p"][0] @ (1, 0.4), s["p"][-1] @ (1, 0.4)))
        if sa["p"][0] @ (1, 0.4) > sa["p"][-1] @ (1, 0.4):
            sa["p"] = sa["p"][::-1]
        sb = min(eb, key=lambda s: min(s["p"][0, 0], s["p"][-1, 0]))
        if sb["p"][-1, 0] < sb["p"][0, 0]:
            sb["p"] = sb["p"][::-1]
        p0, p1 = sa["p"][-1], sb["p"][0]
        L = np.hypot(*(p1 - p0))
        if L > 1.1 * size or p0[1] < -1.5 * size or sa is sb:
            return
        d0 = p0 - sa["p"][max(len(sa["p"]) - 4, 0)]
        d1 = sb["p"][min(3, len(sb["p"]) - 1)] - p1
        d0, d1 = d0 / (np.hypot(*d0) + 1e-9), d1 / (np.hypot(*d1) + 1e-9)
        t = np.linspace(0, 1, max(int(L / 0.05), 3))[:, None]
        c0, c1 = p0 + d0 * L * 0.4, p1 - d1 * L * 0.4
        bez = (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * c0 + 3 * (1 - t) * t ** 2 * c1 + t ** 3 * p1
        sb["p"] = np.vstack([sa["p"], bez[1:-1], sb["p"]])
        sa["p"] = None

    def scribble(self, x0, x1, size):
        rng, k = self.rng, self.rng.integers(3)
        y = -size * rng.uniform(0.3, 0.6)
        if k == 0:
            return [dict(p=np.c_[np.linspace(x0, x1, 20), y + np.linspace(0, rng.normal(0, 0.15) * size, 20)], dot=False)]
        if k == 1:
            return [dict(p=np.c_[np.linspace(x0, x1, 20), y + d + np.linspace(0, rng.normal(0, 0.1) * size, 20)], dot=False)
                    for d in (-0.15 * size, 0.15 * size)]
        n = int(rng.integers(5, 12))
        xs = np.linspace(x0, x1, n * 2)
        ys = y + np.tile([-0.5, 0.5], n) * size * rng.uniform(0.6, 1.2)
        return [dict(p=smooth(resample(np.c_[xs, ys], size * 0.05), 2), dot=False)]

    def lines(self, text, width_mm, rules=None):
        """Strokes of text in mm wrapped to width_mm; rules: baseline pitch to follow (ruled paper), else own pitch."""
        p, rng = self.p, self.rng
        xh = p["size"]
        pitch = rules or p["pitch"] * xh
        tokens = []
        for w in text.split():
            scr = word_script(w)
            if scr in NOSPACE and len(w) > 6:
                cs = regex.findall(r"\X", w)
                step = 1 if scr in ("HANI", "KANA") else int(rng.integers(3, 7))
                tokens += [("".join(cs[j:j + step]), scr) for j in range(0, len(cs), step)]
                tokens[-1] = tokens[-1] + (True,)
            else:
                tokens.append((w, scr, True))
        rtl = Counter(word_script(w) in RTL for w in text.split()).most_common(1)[0][0]
        out, line, words = [], 0, []
        x = rng.normal(0, p["margin_sd"] * xh * 0.3)
        nt = len(tokens)
        for i, tk in enumerate(tokens):
            w, scr = tk[0], tk[1]
            space = len(tk) > 2
            cross = rng.random() < p["cross"] and len(w) > 2 and w.isalpha()
            todo = []
            if cross:
                ch = list(w)
                j = int(rng.integers(len(ch)))
                ch[j] = ch[int(rng.integers(len(ch)))] if rng.random() < 0.5 else ch[j] * 2
                todo.append(("".join(ch[:int(rng.integers(2, len(ch) + 1))]), True))
            todo.append((w, False))
            for ww, struck in todo:
                strokes, adv = self.word(ww, xh, i / max(nt - 1, 1))
                if rng.random() < p["fix"] and len(strokes) > 1:
                    s = strokes[int(rng.integers(len(strokes)))]
                    strokes.append(dict(p=s["p"] + rng.normal(0, 0.06, 2) * xh, dot=False))
                if x + adv > width_mm and words:
                    line += 1
                    x = rng.normal(0, p["margin_sd"] * xh * 0.3)
                gap = (p["ws"] * xh * 2 * (1 + rng.normal(0, 0.25))) if space else rng.normal(0.05, 0.05) * xh
                if struck:
                    strokes += self.scribble(-0.05 * xh, adv + 0.05 * xh, xh)
                words.append((line, x, strokes, adv))
                x += adv + (max(gap, 0.1 * xh) if space else gap)
        nl = line + 1
        slope = np.tan(np.radians(p["slope"]))
        warp = Field(rng, p["elastic"] * xh * 0.12, xh * 2.5)
        for line in range(nl):
            ws = [w for w in words if w[0] == line]
            if not ws:
                continue
            lw = max(width_mm, 1)
            base = line * pitch + rng.normal(0, 0.06) * xh * (0 if rules else 1)
            sl = slope + rng.normal(0, 0.006 if rules else 0.012)
            bow = p["bow"] * xh * rng.uniform(0.3, 1)
            for _, x, strokes, adv in ws:
                X = width_mm - x - adv if rtl else x
                for s in strokes:
                    q = s["p"] + (X, 0)
                    u = q[:, 0] / lw
                    q[:, 1] += base + sl * q[:, 0] + bow * 4 * u * (1 - u)
                    out.append(dict(p=q, dot=s["dot"]))
        for s in out:
            s["p"] = warp(s["p"])
            n = len(s["p"])
            if n > 3 and p["tremor"] > 0.05:
                s["p"] = s["p"] + np.c_[noise1d(rng, n, 1.2, 1), noise1d(rng, n, 1.2, 1)] * p["tremor"] * xh * 0.012
        return out, nl * pitch


def render(strokes, w, dpi, shape=None, origin=(0.0, 0.0)):
    """Ink density (H, W) float32 in [0, 1] of strokes in mm; origin: mm at pixel (0, 0)."""
    rng, p, pen = w.rng, w.p, w.pen
    ss = 3
    k = dpi / 25.4
    if shape is None:
        allp = np.vstack([s["p"] for s in strokes])
        origin = allp.min(0) - 3 * w.width - 1
        shape = tuple((np.ceil((allp.max(0) - origin + 3 * w.width + 1) * k)).astype(int)[::-1])
    C = np.zeros(shape, np.float32)
    over = {"ballpoint": 0.25, "gel": 0.2, "fountain": 0.75, "pencil": 0.45, "felt": 0.6}[pen]
    xh = p["size"]
    for s in strokes:
        q = (s["p"] - origin) * k
        q = resample(q, 0.3 / ss) if len(q) > 1 else q
        n = len(q)
        seg = np.diff(q, axis=0, prepend=q[:1])
        sl = np.hypot(*seg.T)
        arc = np.cumsum(sl) / k
        L = arc[-1] + 1e-6
        ang = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]))
        curv = np.abs(np.gradient(ang)) / (sl / k + 1e-3) if n > 2 else np.zeros(n)
        curv = smooth(np.c_[curv, curv], 4)[:, 0] if n > 8 else curv
        v = 1 / (1 + 0.6 * curv * xh)
        ramp = np.minimum(1, arc / (0.15 * xh)) * np.minimum(1, (L - arc) / (0.2 * xh))
        v = v * (0.3 + 0.7 * ramp ** 0.5)
        pr = 0.75 + p["press_sd"] * 0.3 * noise1d(rng, n, 40 * ss, 1) - 0.2 * (v - 0.5)
        pr *= (0.55 + 0.45 * np.minimum(1, (L - arc) / (0.08 * xh))) * (0.85 + 0.15 * np.minimum(1, arc / (0.04 * xh)))
        pr = np.clip(pr, 0.1, 1.3)
        if pen == "fountain":
            wd = w.width + w.nib[0] * np.abs(np.sin(ang - w.nib[1])) * (0.8 + 0.3 * pr)
            d = 0.6 + 0.2 * pr + 0.25 * (1 - v)
        elif pen == "ballpoint":
            wd = w.width * (0.88 + 0.15 * pr)
            d = 0.72 + 0.28 * pr - 0.3 * v * p["speed"] + noise1d(rng, n, 6 * ss, 0.07)
            skip = (v * p["speed"] > 0.55) & (noise1d(rng, n, 10 * ss, 1) > 1.1)
            d = np.where(skip, d * 0.25, d)
            d[:int(0.02 * k * ss)] += 0.15
        elif pen == "gel":
            wd = w.width * (0.82 + 0.3 * pr)
            d = 0.9 + 0.1 * pr
        elif pen == "pencil":
            wd = w.width * (0.9 + 0.2 * pr)
            d = 0.5 + 0.4 * pr - 0.1 * v
        else:
            wd = w.width * (0.92 + 0.15 * pr)
            d = 0.85 + 0.1 * pr
        if s["dot"]:
            wd = wd * 1.15
        r = np.maximum(np.minimum(wd, 0.15 * xh) * k * ss / 2, 0.35)
        d = np.clip(d, 0, 1)
        x0, y0 = np.floor(q.min(0) - r.max() - 2).astype(int)
        x1, y1 = np.ceil(q.max(0) + r.max() + 2).astype(int)
        P = np.zeros(((y1 - y0) * ss, (x1 - x0) * ss), np.uint8)
        Q = ((q - (x0, y0)) * ss * 16).astype(np.int32)
        R = (r * 16).astype(np.int32)
        D = (d * 255).astype(int)
        for i in range(n):
            cv2.circle(P, (int(Q[i, 0]), int(Q[i, 1])), int(R[i]), int(D[i]), -1, cv2.LINE_AA, 4)
        P = cv2.resize(P, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        ya, yb, xa, xb = max(y0, 0), min(y1, shape[0]), max(x0, 0), min(x1, shape[1])
        if ya >= yb or xa >= xb:
            continue
        P = P[ya - y0:yb - y0, xa - x0:xb - x0]
        c = C[ya:yb, xa:xb]
        C[ya:yb, xa:xb] = np.maximum(c, P) + over * np.minimum(c, P) * (1 - np.maximum(c, P))
    mm = lambda x: max(x * k, 0.01)
    if pen == "pencil":
        g = cv2.GaussianBlur(rng.normal(1, 0.5, shape).astype(np.float32), (0, 0), mm(0.04))
        fib = cv2.GaussianBlur(rng.normal(0, 1, shape).astype(np.float32), (0, 0), sigmaX=mm(0.25), sigmaY=mm(0.03))
        C *= np.clip(g * 0.8 + 0.25 * fib / (fib.std() + 1e-6) + 0.2, 0, 1.2)
        C = cv2.GaussianBlur(C, (0, 0), mm(0.03))
    elif pen == "fountain":
        C = np.maximum(C, cv2.GaussianBlur(C, (0, 0), mm(0.1)) * 0.5)
        C = cv2.GaussianBlur(C, (0, 0), mm(0.025))
    elif pen == "felt":
        C = cv2.GaussianBlur(C, (0, 0), mm(0.05))
    else:
        C = cv2.GaussianBlur(C, (0, 0), mm(0.02))
    return np.clip(C, 0, 1), origin


def paper(shape, rng, ruled=None, dpi=150, top=0.0):
    h, w = shape
    base = np.array(rng.choice([(0.97, 0.965, 0.94), (0.99, 0.99, 0.98), (0.96, 0.95, 0.90), (0.98, 0.97, 0.93)]))
    tex = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 1.2) * 0.012
    tex += cv2.resize(rng.normal(0, 1, (h // 40 + 2, w // 40 + 2)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC) * 0.01
    img = np.clip(base[None, None] + tex[..., None], 0, 1)
    if ruled:
        k = dpi / 25.4
        col = np.array((0.62, 0.75, 0.9)) if rng.random() < 0.7 else np.array((0.7, 0.7, 0.7))
        y = top
        while y * k < h:
            yy = int(round(y * k))
            if yy >= 0:
                img[max(yy - 1, 0):yy + 1] = img[max(yy - 1, 0):yy + 1] * 0.4 + col * 0.6
            y += ruled
    return img


def compose(D, ink, bg):
    return bg * (1 - D[..., None] * (1 - np.array(ink)[None, None]))


def to_u8(img):
    return np.ascontiguousarray((np.clip(img, 0, 1) * 255).astype(np.uint8)[..., ::-1])


def block(text, w, width_mm, dpi=150, ruled=False, margin=6.0):
    """Paper image (BGR uint8) of text written by w."""
    pitch = w.p["pitch"] * w.p["size"]
    rules = round(pitch / 0.5) * 0.5 if ruled else None
    strokes, h = w.lines(text, width_mm, rules)
    k = dpi / 25.4
    first = (rules or pitch) * 1.0 + margin
    shape = (int((h + first + margin) * k), int((width_mm + 2 * margin) * k))
    origin = np.array((-margin, -first))
    D, _ = render(strokes, w, dpi, shape, origin)
    bg = paper(shape, w.rng, rules, dpi, top=first % (rules or 1) if rules else 0)
    return to_u8(compose(D, w.ink, bg))


def random_genes(rng):
    return rng.random(len(GENES))


def bank_writer(rng, sigma=0.04):
    bank = json.load(open(BANK))
    g = np.array(bank[int(rng.integers(len(bank)))]["genes"])
    g = np.clip(g + rng.normal(0, sigma, len(g)), 0, 1)
    names = [n for n, _, _ in GENES]
    for n in ("pen", "ink", "pen_w"):
        g[names.index(n)] = rng.random()
    return g


EVAL_TEXT = ("the quick brown fox jumps over the lazy dog and runs", "Съешь же ещё этих мягких французских булок",
             "Sphinx of black quartz, judge my vow 1984", "Le cœur déçu mais l'âme plutôt naïve")


def line_crop(w, text, dpi=120):
    strokes, _ = w.lines(text, 400)
    D, _ = render(strokes, w, dpi)
    bg = np.full(D.shape + (3,), 0.96)
    g = cv2.cvtColor(to_u8(compose(D, w.ink, bg)), cv2.COLOR_BGR2GRAY)
    ys, xs = np.nonzero(D > 0.2)
    if not len(ys):
        return None
    h = ys.max() - ys.min()
    pad = int(0.15 * h) + 2
    return g[max(ys.min() - pad, 0):ys.max() + pad, max(xs.min() - pad, 0):xs.max() + pad]


@lru_cache(1)
def embedder():
    import torch
    import torchvision
    m = torchvision.models.resnet18()
    m.load_state_dict(torch.load(os.path.expanduser("~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth")))
    m.fc = torch.nn.Identity()
    return m.eval()


def embed(crops):
    """Unit ResNet18 (ImageNet) embeddings of grayscale line crops: 64 px high, 639 px folded into three rows."""
    import torch
    xs = []
    for c in crops:
        c = cv2.normalize(c, None, 0, 255, cv2.NORM_MINMAX)
        w = min(int(c.shape[1] * 64 / c.shape[0]), 639)
        t = np.full((64, 639), 255, np.uint8)
        t[:, :w] = cv2.resize(c, (max(w, 1), 64))[:, :639]
        xs.append(np.vstack([t[:, i * 213:(i + 1) * 213] for i in range(3)]))
    with torch.no_grad():
        out = []
        for i in range(0, len(xs), 64):
            x = (torch.tensor(np.array(xs[i:i + 64]) / 255.0, dtype=torch.float32)[:, None].repeat(1, 3, 1, 1) - 0.45) / 0.25
            out.append(embedder()(x).numpy())
    E = np.vstack(out)
    return E / np.linalg.norm(E, axis=1, keepdims=True)


def page_lines(g):
    """Text-line crops of a grayscale page: ink bands of the row profile after removing rules."""
    b = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    for k in ((g.shape[1] // 8, 1), (1, g.shape[0] // 8)):
        b = cv2.subtract(b, cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, k)))
    prof = cv2.GaussianBlur((b > 0).sum(1).astype(np.float32)[:, None], (1, 0), 3)[:, 0]
    on = prof > max(prof.max() * 0.08, 4)
    out, y = [], 0
    while y < len(on):
        if not on[y]:
            y += 1
            continue
        y1 = y
        while y1 < len(on) and on[y1]:
            y1 += 1
        if 15 <= y1 - y <= 120:
            cols = np.nonzero((b[y:y1] > 0).sum(0) > 0)[0]
            if len(cols) and cols[-1] - cols[0] >= 8 * (y1 - y):
                pad = (y1 - y) // 6
                out.append(g[max(y - pad, 0):y1 + pad, max(cols[0] - pad, 0):cols[-1] + pad])
        y = y1
    return out


def clean(c):
    """No rules, staves or bars (no row half ink), enough contrast, some but not too much ink."""
    lo, hi = np.percentile(c, (2, 60))
    ink = c < (lo + hi) / 2
    return hi - lo > 70 and 0.03 < ink.mean() < 0.3 and ink.mean(1).max() < 0.5


def refs(a):
    """Real handwritten lines from the handwritten folder, kept when 5 of their nearest labelled lines
    (tools/hwprint.py labels) are mostly handwriting; saves their embeddings and a contact sheet."""
    import pymupdf
    d = f"{Paths().layouts}/handwritten"
    fs = sorted(f for f in os.listdir(d) if f.endswith(".pdf"))
    random.Random(0).shuffle(fs)
    crops = []
    for f in fs[:a.pages]:
        try:
            pix = pymupdf.open(f"{d}/{f}")[0].get_pixmap(dpi=150, colorspace=pymupdf.csGRAY)
        except Exception:
            continue
        g = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w)
        ls = [c for c in page_lines(g) if clean(c)]
        random.Random(f).shuffle(ls)
        crops += ls[:a.lines]
    lab = pickle.load(open(f"{Paths().layouts}/test/hw_print/labels.pkl", "rb"))
    El, y = embed([x["crop"] for x in lab]), np.array([x["hw"] for x in lab])
    E = embed(crops)
    keep = y[np.argsort(-(E @ El.T), axis=1)[:, :5]].mean(1) >= 0.6
    from hwprint import feats
    F = [feats(c) if k else None for c, k in zip(crops, keep)]
    ok = np.array([f is not None for f in F])
    np.savez(REFS, E=E[ok], F=np.array([f for f in F if f is not None]))
    good = [c for c, k in zip(crops, keep) if k]
    random.Random(1).shuffle(good)
    rows = [cv2.resize(c, (min(int(c.shape[1] * 40 / c.shape[0]), 600), 40)) for c in good[:120]]
    rows = [np.pad(r, ((0, 2), (0, 600 - r.shape[1])), constant_values=255) for r in rows]
    rows += [np.full((42, 600), 255, np.uint8)] * (-len(rows) % 40)
    cv2.imwrite(f"{OUT}/_real_lines.png", np.hstack([np.vstack(rows[i:i + 40]) for i in range(0, len(rows), 40)]))
    print(len(crops), int(keep.sum()))


def evaluate(args):
    from hwprint import feats
    genes, seed = args
    w = Writer(genes, seed)
    cs = [c for t in EVAL_TEXT if (c := line_crop(w, t)) is not None and c.shape[0] > 8]
    fs = [feats(c) for c in cs]
    return [(c, f) for c, f in zip(cs, fs) if f is not None]


class Realism:
    """Typicality among real lines: kNN distance (stroke features) and kNN similarity (embeddings), each as a z-score
    against the same statistic of real lines among the others (leave-one-out); higher is more typical."""

    def __init__(self, k=5):
        r = np.load(REFS)
        self.E, F = r["E"], r["F"]
        self.mu, self.sd = F.mean(0), F.std(0) + 1e-9
        self.F = (F - self.mu) / self.sd
        self.k = k
        self.ref = np.array([self.stats(self.E[i:i + 1], self.F[i:i + 1], loo=True)[0] for i in range(len(F))])
        self.m, self.s = self.ref.mean(0), self.ref.std(0) + 1e-9

    def stats(self, E, Z, loo=False):
        k, a = self.k, int(loo)
        e = np.sort(E @ self.E.T, axis=1)[:, ::-1][:, a:a + k].mean(1)
        d = np.array([np.sort(np.linalg.norm(self.F - z, axis=1))[a:a + k].mean() for z in Z])
        return np.c_[e, -d]

    def __call__(self, E, F):
        return ((self.stats(E, (np.asarray(F) - self.mu) / self.sd) - self.m) / self.s).mean()


def evolve(a):
    """Fitness: realism (Realism) + nov_w * novelty (mean distance of the writer's mean embedding to the 10 nearest of
    archive and population). Archive: realism >= min_q and farther than min_nov from every archived writer."""
    from multiprocessing import Pool
    real = Realism()
    rng = np.random.default_rng(a.seed)
    pop = rng.random((a.pop, len(GENES)))
    archive, A = [], np.zeros((0, 512))
    with Pool(a.workers) as pool:
        for gen in range(a.gens):
            res = pool.map(evaluate, [(g, int(rng.integers(1 << 30))) for g in pop])
            q, desc = np.full(len(pop), -9.0), np.zeros((len(pop), 512))
            for i, cf in enumerate(res):
                if len(cf) == len(EVAL_TEXT):
                    E = embed([c for c, _ in cf])
                    q[i] = real(E, [f for _, f in cf])
                    desc[i] = E.mean(0)
            ref = np.vstack([A, desc])
            nov = np.array([np.sort(np.linalg.norm(ref - d, axis=1))[1:11].mean() for d in desc])
            for g, qq, d in zip(pop, q, desc):
                if qq >= a.min_q and (not len(A) or np.linalg.norm(A - d, axis=1).min() > a.min_nov):
                    archive.append(dict(genes=g.round(4).tolist(), q=round(float(qq), 4)))
                    A = np.vstack([A, d])
            fit = q + a.nov_w * nov
            el = pop[np.argsort(-fit)[:a.pop // 4]]
            kids = []
            while len(kids) < a.pop - len(el):
                i, j = rng.choice(len(el), 2, replace=False)
                c = np.where(rng.random(len(GENES)) < 0.5, el[i], el[j])
                c = np.clip(c + (rng.random(len(GENES)) < 0.2) * rng.normal(0, 0.12, len(GENES)), 0, 1)
                kids.append(np.where(rng.random(len(GENES)) < 0.03, rng.random(len(GENES)), c))
            pop = np.vstack([el, kids])
            print(f"gen {gen} q {q.max():.3f}/{np.median(q):.3f} nov {nov.mean():.3f} archive {len(archive)}", flush=True)
            json.dump(archive, open(BANK, "w"))


SENT = {
    "en": "The quick brown fox jumps over the lazy dog. Call me tomorrow at 5:30, please!",
    "de": "Zwölf Boxkämpfer jagen Viktor quer über den großen Sylter Deich.",
    "fr": "Voix ambiguë d'un cœur qui, au zéphyr, préfère les jattes de kiwis.",
    "es": "El pingüino Wenceslao hizo kilómetros bajo exhaustiva lluvia y frío.",
    "pl": "Zażółć gęślą jaźń, a potem napisz list do babci w Łodzi.",
    "cs": "Příliš žluťoučký kůň úpěl ďábelské ódy.",
    "tr": "Pijamalı hasta yağız şoföre çabucak güvendi.",
    "vi": "Tiếng Việt có dấu: Hà Nội mùa thu đẹp lắm.",
    "ru": "Съешь же ещё этих мягких французских булок да выпей чаю.",
    "uk": "Чуєш їх, доцю, га? Кумедна ж ти, прощайся без ґольфів!",
    "bg": "Жълтата дюля беше щастлива, че пухът, който цъфна, замръзна.",
    "sr": "Љубазни фењерџија чађавог лица хоће да ми покаже штос.",
    "kk": "Қазақ тілі: әдемі ғажайып өлкеде ұлы күй үнін естідік.",
    "mn": "Монгол хэл: өглөөний нар ууланд мандаж байна.",
    "el": "Ξεσκεπάζω την ψυχοφθόρα βδελυγμία, γράφοντας με το χέρι.",
    "hy": "Բարև ձեզ, ես գրում եմ ձեռքով այս նամակը։",
    "ka": "გამარჯობა, ეს წერილი ხელით არის დაწერილი.",
    "he": "דג סקרן שט בים מאוכזב ולפתע מצא חברה.",
    "ar": "صِف خلق خود كمثل الشمس إذ بزغت يحظى الضجيع بها.",
    "fa": "این نامه را با دست نوشتم تا یادگاری بماند.",
    "hi": "मैं यह पत्र अपने हाथ से लिख रहा हूँ।",
    "bn": "আমি নিজের হাতে এই চিঠি লিখছি।",
    "ta": "நான் இந்தக் கடிதத்தை என் கையால் எழுதுகிறேன்.",
    "te": "నేను ఈ ఉత్తరాన్ని నా చేతితో రాస్తున్నాను.",
    "gu": "હું આ પત્ર મારા હાથે લખું છું.",
    "th": "ฉันเขียนจดหมายฉบับนี้ด้วยมือของฉันเอง",
    "zh": "我用手写这封信，希望你一切都好。",
    "ja": "この手紙は自分の手で書きました。お元気ですか。",
    "ko": "이 편지는 제 손으로 직접 썼습니다. 잘 지내세요.",
    "am": "ይህን ደብዳቤ በእጄ ጻፍኩት።",
    "chr": "ᏣᎳᎩ ᎦᏬᏂᎯᏍᏗ ᎠᏍᎦᏯ ᎤᏬᏪᎳᏅ.",
    "iu": "ᐃᓄᒃᑎᑐᑦ ᐊᓪᓚᖑᐊᖅ ᐅᖃᐅᓯᖅ.",
    "num": "Tel. +7 (912) 345-67-89, 24.03.2025, № 1078, 3/4 = 0,75",
}


def demo(a):
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    rows = []
    for i, (lg, t) in enumerate(SENT.items()):
        g = bank_writer(rng) if a.bank else random_genes(rng)
        w = Writer(g, int(rng.integers(1 << 30)))
        img = block(t, w, 150, dpi=a.dpi, ruled=rng.random() < 0.4)
        cv2.putText(img, f"{lg} {w.pen}", (6, 16), 0, 0.45, (0, 0, 200), 1)
        cv2.imwrite(f"{OUT}/{lg}.png", img)
        rows.append(img)
    W = max(r.shape[1] for r in rows)
    sheet = np.vstack([np.pad(r, ((0, 4), (0, W - r.shape[1]), (0, 0)), constant_values=255) for r in rows])
    cv2.imwrite(f"{OUT}/_overview.png", sheet)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("demo")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--bank", action="store_true")
    p = sp.add_parser("refs")
    p.add_argument("--pages", type=int, default=1200)
    p.add_argument("--lines", type=int, default=8)
    p = sp.add_parser("evolve")
    p.add_argument("--gens", type=int, default=40)
    p.add_argument("--pop", type=int, default=48)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-q", type=float, default=0.0)
    p.add_argument("--min-nov", type=float, default=0.05)
    p.add_argument("--nov-w", type=float, default=0.5)
    a = ap.parse_args()
    {"demo": demo, "refs": refs, "evolve": evolve}[a.cmd](a)
