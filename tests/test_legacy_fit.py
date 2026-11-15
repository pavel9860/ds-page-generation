import random

from dspages.layout import legacy_fit as F
from dspages.layout import legacy_tables as T

CIPHER = {"a": "ક", "b": "ખ", "c": "ગ", "d": "ઘ", "e": "ચ", "f": "છ", "g": "જ", "h": "ઝ", "i": "ટ", "j": "ઠ"}


def make_pairs(n=80, seed=1):
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        codes = "".join(rnd.choice("abcdefghij") for _ in range(rnd.randint(6, 14)))
        out.append((codes, "".join(CIPHER[c] for c in codes)))
    return out


def test_lev():
    assert F.lev("kitten", "sitting") == 3 and F.lev("", "ab") == 2 and F.lev("same", "same") == 0


def test_clean_drops_spaces_collapses_faked_bold_and_drops_croppings():
    assert F.clean([("aa bb cc", "ક ખ ગ")]) == [("abc", "કખગ")]
    assert F.clean([("abcdef", "ક")]) == []  # a cropped picture does not fit its codes


def test_fit_recovers_a_substitution_cipher_and_the_gate_keeps_it_exact():
    pairs = F.clean(make_pairs())
    tab = F.table(pairs, "Gujarati")
    assert len(tab) >= 8 and all(CIPHER[c] == t for c, t in tab.items())


def test_published_finds_the_table_that_says_the_same():
    tab = {c: T.decode("Preeti", c) for c in "ksfdlgj;ahpyto"}
    found = F.published(tab, {k: v for k, v in T.decoders().items() if k.startswith(("preeti:", "cp:"))})
    assert found and found[1].startswith("preeti:")


def test_font_of_real_unicode_is_recognized_and_a_legacy_one_is_not():
    assert F.is_unicode_font([("abc def", "abc def")] * 5)
    assert not F.is_unicode_font([("ksf", "कखग")] * 5)


def test_rtl_puts_visual_order_in_logical_order_but_keeps_digit_runs():
    assert T.rtl("هلاقم پاچ تساوخرد مرف") == "فرم درخواست چاپ مقاله"
    assert T.rtl("09126932196  :") == "09126932196  :"


def test_sequence_moves_prefix_vowel_and_reph():
    assert T.sequence(["ि", "क"], "Devanagari") == "कि"
    assert T.sequence(["क", "र्", "म"], "Devanagari") == "र्कम"


def test_double_byte_legacy_fonts_decode_and_plain_text_is_not_one():
    assert (
        T._multibyte("big5", "\xa4\xa4\xa4\xe5 abc") == "中文 abc"
        and T._multibyte("euc_kr", "\xc7\xd1\xb1\xdb") == "한글"
    )
    assert F.has_high_codes(["\xa4\xa4\xa4\xe5 \xa4\xa4"]) and not F.has_high_codes(["()=()log", "2025/11/28"])
