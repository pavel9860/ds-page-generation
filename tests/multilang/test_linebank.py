"""Line bank and field values"""
import random

from dspages.multilang import linebank as LB
from dspages.multilang.config import LINEBANK as C
from dspages.multilang.dettext import Values

TEXT = ("The committee met on Monday, and it approved the budget for the next year. Applications must be sent in writing; late forms "
        "are not accepted by the office under any circumstances whatsoever, as the rules say. Contact the secretary.")


def test_phrases_are_whole_and_short():
    ph = LB.phrases(TEXT, "en")
    assert "Contact the secretary" in ph and "and it approved the budget for the next year" in ph
    assert all(1 <= len(p.split()) <= C.max_words for p in ph)
    assert not any(p != p.strip(C.edge) for p in ph)


def bank(n=400):
    rng = random.Random(1)
    words = [f"w{chr(97 + i % 26)}{chr(97 + i // 26 % 26)}" for i in range(300)]
    return LB.Bank("en", [" ".join(rng.sample(words, rng.randint(1, 10))) for _ in range(n)])


def test_pick_fits_and_never_repeats():
    b, rng, used = bank(), random.Random(2), set()
    got = [b.pick(cap, rng, used, set()) for cap in [5, 9, 14, 20, 30, 45, 60, 80] * 12]
    got = [g for g in got if g]
    assert len(got) > 80
    assert len({LB.key(g) for g in got}) == len(got)
    caps = [5, 9, 14, 20, 30, 45, 60, 80] * 12
    assert all(len(g) <= c for g, c in zip([b.pick(c, rng, used) for c in caps], caps) if g)


def test_pick_lengths_follow_the_field():
    b, rng = bank(), random.Random(3)
    short = [len(b.pick(10, rng, set()) or "") for _ in range(60)]
    long = [len(b.pick(70, rng, set()) or "") for _ in range(60)]
    assert sum(long) / 60 > 2 * sum(short) / 60


def test_stems_block_same_opening():
    b = LB.Bank("en", ["alpha beta one", "alpha beta two", "gamma delta three"])
    used, stems = set(), set()
    got = [b.pick(40, random.Random(i), used, stems) for i in range(6)]
    assert len([g for g in got if g and g.startswith("alpha beta")]) <= 1


def test_people_differ_and_order():
    v = Values("en", random.Random(4))
    people = [v.person() for _ in range(8)]
    assert len({p[2] for p in people}) == 8 and all(p[2] == f"{p[0]} {p[1]}" for p in people)
    h = Values("hu", random.Random(4)).person()
    assert h[2] == f"{h[1]} {h[0]}"


def test_email_ascii_and_date_formats():
    v = Values("ru", random.Random(5))
    p = v.person()
    assert v.email(p).isascii()
    assert len({v.date() for _ in range(40)}) > 30

def test_page_note_from_a_full_bank():
    import numpy as np
    from dspages.multilang.config import Config
    from dspages.multilang.notes import Annotator
    img = np.full((800, 600, 3), 255, np.uint8)
    prints = [(50, 100 + 40 * i, 550, 125 + 40 * i) for i in range(12)]
    words = [(60 + 70 * j, 100 + 40 * i, 120 + 70 * j, 125 + 40 * i, f"word{i}{j}") for i in range(12) for j in range(6)]
    a = Annotator(img, prints, words, "en", "Latin", Config(), random.Random(0))
    a.lines = [f"note about {chr(97 + i % 26)}{chr(97 + i // 26 % 26)}{chr(97 + i // 676)} here" for i in range(C.min_lines + 10)]
    for b in prints[:3]:
        a.note(b)
    assert a.used and all(not any(c.isdigit() for c in text) for text in a.used)


def test_lines_of_marked_scripts_are_cut_at_their_own_marks():
    samples = {"dz": "ལས་འགུལ་འདི་ཨའི་ཌི་ཨི་ནང་ལུ་ནང་འདྲེན་འབད་ནི་ལུ། འདི་ནི་དཔེ་ཅིག་ཨིན། ཁྱོད་ཀྱིས་ཡིག་ཆ་འདི་ལྟ་ནི་ཨིན།",
               "my": "ပရောဂျက်ကို တင်သွင်းရန်၊ အောက်ပါအတိုင်း အဆင့်များကို လုပ်ဆောင်ပါ။ ဤအရာသည် ဥပမာတစ်ခုဖြစ်သည်။",
               "ti": "ነቲ ፕሮጀክት ናብ ንምእታው እዞም ዝስዕቡ ስጉምትታት ንፍጽም፤ እዚ ኣብነት እዩ። ሰነድ ኣንብብ።",
               "hi": "यह एक उदाहरण है। कृपया निम्नलिखित चरणों का पालन करें, और दस्तावेज़ पढ़ें।", "ur": "یہ ایک مثال ہے۔ براہ کرم دستاویز پڑھیں۔"}
    for lang, text in samples.items():
        ph = LB.phrases(text, lang)
        assert len(ph) >= 2 and all(ph), lang
