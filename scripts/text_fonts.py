from __future__ import annotations

import random

FONT_POOL: list[str] = [
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Italic.ttf",
    "/usr/share/fonts/truetype/noto/NotoSerif-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSerif-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSerif-Italic.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/lato/Lato-Regular.ttf",
    "/usr/share/fonts/truetype/lato/Lato-Bold.ttf",
    "/usr/share/fonts/truetype/lato/Lato-Italic.ttf",
]

SCRIPT_FONTS: dict[str, tuple[str, int]] = {
    "ar": ("/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf", 0),
    "fa": ("/usr/share/fonts/truetype/noto/NotoNastaliqUrdu-Regular.ttf", 0),
    "ur": ("/usr/share/fonts/truetype/noto/NotoNastaliqUrdu-Regular.ttf", 0),
    "he": ("/usr/share/fonts/truetype/noto/NotoSansHebrew-Regular.ttf", 0),
    "hi": ("/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf", 0),
    "sa": ("/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf", 0),
    "bn": ("/usr/share/fonts/truetype/noto/NotoSansBengali-Regular.ttf", 0),
    "th": ("/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf", 0),
    "zh": ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 2),
    "ja": ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 0),
    "ko": ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 1),
}

RTL_LANGS = {"ar", "fa", "ur", "he"}


def pick_font(lang: str, rng: random.Random | None = None) -> tuple[str, int]:
    if lang in SCRIPT_FONTS:
        return SCRIPT_FONTS[lang]
    r = rng or random
    return r.choice(FONT_POOL), 0


def is_rtl(lang: str) -> bool:
    return lang in RTL_LANGS


def shape_for_display(text: str, lang: str) -> str:
    if lang not in RTL_LANGS:
        return text
    try:
        if lang in ("ar", "fa", "ur"):
            import arabic_reshaper
            text = arabic_reshaper.reshape(text)
        from bidi.algorithm import get_display
        return get_display(text)
    except Exception:
        return text


_cmap_cache: dict[tuple[str, int], set[int]] = {}


def _cmap_for(font_path: str, font_index: int) -> set[int]:
    key = (font_path, font_index)
    if key not in _cmap_cache:
        from fontTools.ttLib import TTFont
        f = TTFont(font_path, fontNumber=font_index) if font_path.endswith(".ttc") else TTFont(font_path)
        _cmap_cache[key] = set(f.getBestCmap().keys())
    return _cmap_cache[key]


def missing_glyph_ratio(text: str, font_path: str, font_index: int = 0) -> float:
    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return 0.0
    cmap = _cmap_for(font_path, font_index)
    missing = sum(1 for ch in chars if ord(ch) not in cmap)
    return missing / len(chars)
