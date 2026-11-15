"""Which handwriting fonts the writer may use per script."""
from functools import lru_cache

REALISTIC = {
    "Cyrillic": ("BadScript-Regular", "Caveat", "MarckScript-Regular"),
    "Greek": ("LXGWWenKaiTC-Regular", "Mynerve-Regular", "Mansalva-Regular"),
    "Hebrew": ("GveretLevin-Regular", "Solitreo-Regular"),
    "Arabic": ("ArefRuqaa-Regular", "Gulzar-Regular", "NotoNastaliqUrdu"),
    "Devanagari": ("Amita-Regular", "Dekko-Regular"),
    "Tamil": ("Kavivanar-Regular",),
    "Telugu": ("LakkiReddy-Regular", "NotoSansTelugu"),
    "Malayalam": ("Chilanka-Regular",),
    "Sinhala": ("SinhalaHandwriteConverted-Regular",),
    "Thai": ("Charm-Regular", "Charmonman-Regular"),
    "Lao": ("ChartLao-Regular", "NotoSansLao"),
    "Khmer": ("Fasthand-Regular", "Freehand-Regular"),
    "Myanmar": ("ChartMyanmar-Regular", "Padauk-Regular"),
    "Tibetan": ("ChartTibetan-Regular",),
    "Ethiopic": ("ChartGeez-Regular", "AbyssinicaSIL-Regular", "Menbere"),
    "Armenian": ("ChartArmenian-Regular",),
    "Han": ("LongCang-Regular", "MaShanZheng-Regular", "KleeOne-Regular", "Yomogi-Regular", "ZenKurenaido-Regular"),
    "Hiragana": ("Kapakana", "KleeOne-Regular", "SlacksideOne-Regular", "Yomogi-Regular", "YujiHentaiganaAkari-Regular", "ZenKurenaido-Regular"),
    "Hangul": ("EastSeaDokdo-Regular", "NanumBrushScript-Regular", "NanumGothicCoding-Regular", "NanumPenScript-Regular"),
    "Bengali": ("Galada-Regular", "Atma-Regular", "Alkatra"),
    "Gujarati": ("Mogra-Regular", "Farsan-Regular"),
    "Gurmukhi": ("BhuTukaExpandedOne-Regular", "ChartGurmukhi-Regular"),
    "Kannada": ("Hubballi-Regular", "AkayaKanadaka-Regular"),
    "Oriya": ("BalooBhaina2",),
    "Georgian": ("ChartGeorgian-Regular", "NotoSerifGeorgian"),
    "Meetei_Mayek": ("NotoSansMeeteiMayek",),
    "Thaana": (),
}
STEMS = {s: set(v) for s, v in REALISTIC.items()}


def stem(path):
    return str(path).split("/")[-1].split("[")[0].removesuffix(".ttf")


@lru_cache(maxsize=None)
def _listed(index, script):
    from hwsynth.config import GlyphCfg
    from hwsynth.fonts import core_letters
    core = core_letters(script) or ""
    need = GlyphCfg().coverage * len(core)
    return tuple(sorted(f for f in index.cmap if any(stem(f) == s or stem(f).startswith(s) for s in STEMS[script])
                        and sum(ord(c) in index.cmap[f] for c in core) >= need))


class Restricted:
    """The writer's font index limited to the realistic fonts of each script (see the module doc)."""

    def __init__(self, index):
        self.index = index

    def __getattr__(self, name):
        return getattr(self.index, name)

    def _ok(self, script, f):
        return script not in STEMS or any(stem(f) == s or stem(f).startswith(s) for s in STEMS[script])

    def candidates(self, script, cls="any"):
        """The script's listed fonts among every font of the index that covers the script's core letters (a listed font need not be in hwsynth's own
        handwriting list: ArefRuqaa, Hubballi, Noto Sans Meetei Mayek ...); the index's own choice for a script not listed."""
        if script not in STEMS:
            return self.index.candidates(script, cls)
        return _listed(self.index, script)

    def choose(self, script, u, cls="any"):
        cands = self.candidates(script, cls)
        if not cands:
            return None
        return cands[int(((u + (sum(map(ord, script)) % 97) * 0.6180339) % 1) * len(cands))]

    def writes(self, text):
        """True when every letter of `text` is in some allowed handwriting font (the writer drops a letter no font has)."""
        from hwsynth.scripts import text_script
        have = [f for f in self.index.hand if f in self.index.cmap]
        return all(any(ord(c) in self.index.cmap[f] and self._ok(text_script(c), f) for f in have) for c in set(text) if c.isalpha())

    def for_unit(self, font, unit):
        from hwsynth.scripts import text_script
        scr = text_script(unit)
        if font is not None and self.index.covers(font, unit):
            return font
        if scr in STEMS and not STEMS[scr]:
            return None
        for f in self.index.hand:
            if self.index.covers(f, unit) and self._ok(scr, f):
                return f
        return None


def can_write(scripts):
    """True when at least one of the scripts of a language has a realistic font (or is not restricted)."""
    return any(s not in STEMS or STEMS[s] for s in scripts)
