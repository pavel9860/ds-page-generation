"""Language of a text, checked from the text itself"""
import re
import unicodedata

from .config import LANGDS as C

_KNOWN = []


def known_languages():
    """Languages both detectors can name"""
    if not _KNOWN:
        import os

        import langid
        from langdetect import detector_factory
        langid.classify("warm up")
        ld = {p.split("-")[0] for p in os.listdir(detector_factory.PROFILES_DIRECTORY)}
        _KNOWN.append(ld & set(langid.langid.identifier.nb_classes))
    return _KNOWN[0]




def check_text(text, lang):
    """(status, detected): ok (the text is in `lang`)"""
    import langid
    from dspages.pagegen.source import scripts_of
    from hwsynth.scripts import text_script
    from langdetect import DetectorFactory, detect_langs
    DetectorFactory.seed = 0
    t = " ".join(text.split())
    if len(t.split()) < 20 and len(re.sub(r"\s", "", t)) < 120:
        return "short", None
    lo = t.lower()
    cy = sum("Ѐ" <= ch <= "ӿ" for ch in lo)
    if cy > 100 and (lang in C.signatures or lang == "bg"):
        s = {k: sum(ch in v for ch in lo) / cy * 1000 for k, v in C.signatures.items()}
        got = "kk" if s["kk"] > 8 else "uk" if s["uk"] > 2 else "be" if s["be"] > 2 else "ru" if s["ru"] > 3 else None
        if got == lang or (lang == "bg" and got is None):
            return "ok", lang
        return ("other", got) if got else ("mismatch", None)
    ld = detect_langs(t)[0].lang.split("-")[0] if any(c.isalpha() for c in t) else "?"
    li = langid.classify(t)[0]
    if lang in (ld, li) or (lang in ("no", "nb") and "no" in (ld, li)):
        return "ok", lang
    known = known_languages()
    if lang not in known and text_script(t) in scripts_of(lang) and not ({ld, li} & known):
        return "ok", lang
    if ld == li and ld in known and ld != lang and lang in known:
        return "other", ld
    return "mismatch", None


_NAME2SCRIPT = {"Cjk": "Han"}


def script_of_char(c):
    import unicodedata
    try:
        n = unicodedata.name(c).split()[0].title()
    except ValueError:
        return "?"
    return _NAME2SCRIPT.get(n, "Meetei_Mayek" if n == "Meetei" else n)




def repair(text):
    """The text with what can be repaired repaired, so it need not be rejected"""
    import ftfy
    text = ftfy.fix_text(text, uncurl_quotes=False, fix_character_width=False, normalization="NFC")
    if re.search(C.presentation, text):
        text = unicodedata.normalize("NFKC", text)
    return re.sub("ا([إأآ])ل", "ال\\1", text)


def lexicon_share(text, lang):
    """Share of the words of `text` found in the word-frequency list of its language (wordfreq)"""
    from wordfreq import available_languages, tokenize, zipf_frequency
    lang = C.lexicon_langs.get(lang, lang)
    if lang not in available_languages() or lang in ("zh", "ja", "ko"):
        return None
    words = [w for w in tokenize(text, lang) if len(w) > 1 and any(c.isalpha() for c in w) and not any(c.isdigit() for c in w)]
    return sum(zipf_frequency(w, lang) > 0 for w in words) / len(words) if words else None


def clean_ok(text, lang, max_foreign=0.10, lexicon=True):
    """False for a text with control or private-use characters"""
    from dspages.pagegen.source import scripts_of
    if re.search(C.mojibake, text) or any(ch == "\ufffd" or 0xE000 <= ord(ch) <= 0xF8FF or (ord(ch) < 32 and ch not in "\n\t") for ch in text):
        return False
    words = text.split()
    split = sum(unicodedata.category(w[0])[0] == "M" or (len(w) == 1 and w.isalpha() and (0x600 <= ord(w) < 0x700 or 0x900 <= ord(w) < 0xE00))
                for w in words)
    if split > C.max_split_words * len(words):
        return False
    share = lexicon_share(text, lang) if lexicon else None
    if share is not None and share < (C.min_lexicon if len(words) >= C.lexicon_min_words else C.min_lexicon_short):
        return False
    own = scripts_of(lang) | {"Common"}
    if any(0x80 <= ord(c) <= 0x9F for c in text):
        return False
    if "Latin" not in own and any(ord(c) > 127 and c.isalpha() and script_of_char(c) not in own for c in text):
        return False
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(script_of_char(c) not in own for c in letters) / len(letters) <= max_foreign
