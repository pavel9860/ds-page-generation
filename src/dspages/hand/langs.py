"""Languages and scripts of the text sources: which script a language is written in, and checks that a passage is real text of it."""
import math
from collections import Counter, defaultdict

from hwsynth.scripts import script

EXPECT = {"el": "Greek", "he": "Hebrew", "ar": "Arabic", "fa": "Arabic", "ur": "Arabic", "ps": "Arabic", "sd": "Arabic",
          "ckb": "Arabic", "ku": "Arabic", "hi": "Devanagari", "mr": "Devanagari", "ne": "Devanagari", "sa": "Devanagari",
          "mai": "Devanagari", "bn": "Bengali", "as": "Bengali", "th": "Thai", "ka": "Georgian", "hy": "Armenian", "ta": "Tamil",
          "te": "Telugu", "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam", "pa": "Gurmukhi", "or": "Oriya", "si": "Sinhala",
          "km": "Khmer", "lo": "Lao", "my": "Myanmar", "am": "Ethiopic", "bo": "Tibetan", "zh": "Han", "ja": "Han", "ko": "Hangul",
          "ru": "Cyrillic", "uk": "Cyrillic", "bg": "Cyrillic", "mk": "Cyrillic", "be": "Cyrillic", "kk": "Cyrillic",
          "ky": "Cyrillic", "tg": "Cyrillic", "mn": "Cyrillic", "sr": "Cyrillic"}       # languages not listed are Latin
NO_TEXT_LAYER = {"ur", "ps", "ku", "ckb", "sd"}        # their PDF text layers are in visual order or scrambled: no usable text from them
MERGE = {"Hiragana": "Han", "Katakana": "Han"}
SKIP_SCRIPTS = {"Thaana", "Common", "Inherited", "Unknown"}
GROUP = {"Latin": "latin", "Cyrillic": "cyrillic", "Han": "cjk", "Hiragana": "cjk", "Katakana": "cjk", "Hangul": "cjk"}


def group_of(scr):
    """Script group of the language shares: latin, cyrillic, cjk or other."""
    return GROUP.get(scr, "other")


def script_ok(text, lang, share=None):
    """True when most letters of the text are in the script of the language (Latin when the language has no entry): legacy-encoded PDFs
    decode to Latin gibberish, romanised texts are not written in their script."""
    want = EXPECT.get(lang, "Latin")
    c = Counter(MERGE.get(script(ch), script(ch)) for ch in text if ch.isalpha())
    n = sum(c.values())
    return n > 0 and c[want] >= (share or (0.97 if want == "Latin" else 0.8)) * n


def implausible(texts, cut=0.08):
    """Drops, per non-Latin language, the passages least likely under the character-bigram model of the language: legacy-encoded text
    that decodes to valid but scrambled letters."""
    by = defaultdict(list)
    for i, t in enumerate(texts):
        by[t["lang"]].append(i)
    drop = set()
    for lg, ix in by.items():
        if lg not in EXPECT or len(ix) < 30:
            continue
        bi, uni = Counter(), Counter()
        for i in ix:
            x = texts[i]["text"]
            bi.update(zip(x, x[1:]))
            uni.update(x[:-1])
        score = {}
        for i in ix:
            x = texts[i]["text"]
            ps = [math.log((bi[(a, b)] + 0.1) / (uni[a] + 0.1 * 200)) for a, b in zip(x, x[1:])]
            score[i] = sum(ps) / max(len(ps), 1)
        thr = sorted(score.values())[int(cut * len(ix))]
        drop |= {i for i, v in score.items() if v < thr}
    return [t for i, t in enumerate(texts) if i not in drop]
