"""Every number of the pipeline."""
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dspages.pagegen.config import PAGE, GenCfg, PageCfg  # noqa: F401

from . import paths

HW, PRINT = 0, 1
NAMES = ("handwriting", "print")


@dataclass(frozen=True)
class Config(GenCfg):
    data: Path = paths.DATA
    layouts: Path = paths.LAYOUTS
    bg_folders: tuple = ("forms", "books_pdf", "other_docs", "certificates", "exams", "tables", "receipts",
                         "scientific_paper", "magazines", "flyers")
    real_folders: tuple = ("filled_forms", "handwritten", "forms", "books_pdf")
    real_pages: int = 250
    real_seed: int = 11
    workers: int = 6
    seed: int = 0

    dpi: int = 150
    jpeg_quality: tuple = (70, 95)
    min_box_px: int = 2

    text_files_per_lang: int = 25
    text_pages_per_file: int = 4
    text_line_len: tuple = (3, 80)
    text_min_lines: int = 30
    text_max_lines: int = 3000
    text_min_alpha: float = 0.85
    no_font_langs: tuple = ("kn", "or", "dv")     # their PDFs are Latin text; the scripts have no handwriting font
    translated_key: dict = field(default_factory=lambda: {"ja_hira": "script:Hiragana", "ja_kata": "script:Katakana"})
    language_scripts: tuple = ("Bengali", "Gujarati", "Gurmukhi", "Khmer", "Malayalam", "Myanmar", "Sinhala", "Tamil",
                               "Telugu", "Tibetan")
    text_bad: str = "\\{}|<>^~`_=[]"

    ds_name: str = "ds"
    para_name: str = "para"
    bank_name: str = "bank"           # "bank_real": lines cut from real handwritten pages (dspages.multilang.realbank)
    bank_lines: int = 12000
    bank_chunk: int = 100
    bank_group: int = 20
    bank_p_rare: float = 0.2
    bank_p_join: float = 0.3
    p_second_lang: float = 0.1
    bank_p_crop: float = 0.5
    bank_crop_words: tuple = (1, 4)
    bank_width_mm: tuple = (40, 170)
    bank_p_date: float = 0.35
    val_chunk_every: int = 10

    para_pages: int = 300
    para_chunk: int = 10
    para_p_rare: float = 0.15
    para_sentences: tuple = (4, 25)
    para_width_mm: tuple = (110, 190)
    para_dpi: int = 200
    para_aspect: tuple = (1.3, 1.45)

    n_train: int = 9000
    n_val: int = 600
    bg_reuse: int = 3
    bg_attempts: int = 5
    p_blank: float = 0.45
    p_picture: float = 0.05
    picture_conf: float = 0.1
    p_full: float = 0.4
    p_form: float = 0.4
    p_letter: float = 0.5
    p_strip: float = 0.5
    p_strip_top: float = 0.6
    strip_header: float = 0.22
    strip_footer: float = 0.85
    strip_pad: int = 6
    strip_max: float = 0.25
    max_picture_cover: float = 0.5
    min_prints: int = 3
    max_prints: int = 400
    seen_contrast: int = 60
    free_cover: float = 0.02

    field_fill: tuple = (0.3, 1.0)
    field_tries: int = 3
    field_scale: tuple = (0.8, 1.3)
    field_dy: tuple = (-0.1, 0.25)
    field_overlap: float = 0.05
    label_gap: float = 0.4
    label_min_width: float = 4
    rule_above: float = 2
    gap_items: tuple = (1, 8)
    gap_scale: tuple = (0.8, 1.8)
    gap_tries: int = 40
    letter_items: tuple = (6, 30)
    letter_scale: tuple = (0.5, 1.0)
    letter_tries: int = 60
    stack_pitch: tuple = (22, 48)
    stack_scale: tuple = (0.75, 0.95)
    stack_p_ruled: float = 0.5
    stack_min_aspect: float = 8
    stack_candidates: int = 60
    stack_min_items: int = 8
    page_aspect: tuple = (0.68, 0.78)
    page_margin: int = 30
    ink_density: tuple = (0.75, 1.0)
    field_density: tuple = (0.8, 1.0)

    p_scan_text: float = 0.5     # does not give the class away
    p_paper: float = 0.5         # pages on plain white get a paper sheet (handwritten pages are drawn on one already)

    skip_langs: str = "skip_langs.json"

    ticks: bool = False
    marks_per_group: int = 4
    mark_kinds: tuple = ("check", "cross")
    mark_box_pt: tuple = (5, 18)
    mark_aspect: tuple = (0.8, 1.25)
    mark_fill: tuple = (0.0, 0.6)
    mark_scale: tuple = (1.0, 1.6)
    mark_shift: float = 0.15

    results: Path = paths.RESULTS
    eval_conf: float = 0.3
    eval_contain: float = 0.5

    def sha(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, default=str).encode()).hexdigest()[:12]

    @property
    def text_pool(self):
        return self.data / "text.pkl"

    @property
    def bank(self):
        return self.data / self.bank_name

    @property
    def para(self):
        return self.data / self.para_name

    @property
    def picture_items(self):
        return paths.CORPUS / "kindnet_data/items.jsonl"


    @property
    def pictures(self):
        return self.data / "pictures"

    @property
    def picture_model(self):
        return paths.MODELS / "hwdet-s-0.4.0.pt"

    @property
    def ds(self):
        return self.data / self.ds_name


@dataclass(frozen=True)
class LangDs:
    """The language dataset: page types and shares"""
    types: tuple = ("text", "table", "form", "filled", "flyer", "magazine", "handwritten")
    share: dict = field(default_factory=lambda: {"text": 0.34, "table": 0.13, "form": 0.09, "filled": 0.14, "flyer": 0.09, "magazine": 0.07,
                                                 "handwritten": 0.14})
    min_per_type: int = 3                                   # pages of every type in every language, at least
    real_share: float = 0.5                                 # share of a (language, type) quota taken from real pages
    scans_ok: frozenset = frozenset({"form", "flyer", "magazine", "handwritten"})   # scans only where no verified digital page of the type exists
    hand: tuple = ("handwritten", "filled")
    min_pool: int = 40                                      # paragraphs a language should have for varied text pages
    alias: dict = field(default_factory=lambda: {"kz": "kk", "tj": "tg", "fil": "tl", "nb": "no"})
    skip_langs: frozenset = frozenset({"xx", "mul", "mu"})
    side: int = 1024                                        # px of the window, at `dpi`
    dpi: int = 150
    folders: dict = field(default_factory=lambda: {
        "text": ["books_pdf", "scientific_paper", "other_docs", "exams", "bureaucracy"], "table": ["tables"], "form": ["forms"],
        "filled": ["filled_forms"], "flyer": ["flyers"], "magazine": ["magazines"], "handwritten": ["handwritten"]})
    # the search of other folders for a type: its own folders first (the *_tmp pools are the unreviewed ones), then every other folder
    search_folders: dict = field(default_factory=lambda: {
        "text": ["books_pdf", "scientific_paper", "other_docs", "exams", "bureaucracy"],
        "table": ["tables_tmp", "bureaucracy", "bureaucracy_tmp", "scientific_paper", "other_docs", "exams"],
        "form": ["forms_tmp", "bureaucracy_tmp", "bureaucracy", "exams", "other_docs"],
        "flyer": ["flyers_tmp", "other_docs", "certificates_tmp", "certificates", "bureaucracy"],
        "magazine": ["magazines_tmp", "other_docs", "books_pdf"]})
    search_excluded: frozenset = frozenset({"archive", "sheet_music", "maps_blueprints", "blueprints_tmp"})
    thumb: int = 384                                        # px of the thumbnail the page-type classifier reads
    max_pages: int = 20                                     # pages looked at in one document, at most
    per_file_per_type: int = 3                              # pages of one type taken from one document
    search_chunk: int = 10                                  # documents scanned between two checks of the needs
    pagetype: str = str(paths.CORPUS)
    accept: dict = field(default_factory=lambda: {
        "text": ("plain", "formatted", "scientific", "poems", "bureaucracy", "recieps"), "table": ("tables",),
        "form": ("forms_blank", "forms_filled", "bureaucracy"), "filled": ("forms_filled", "forms_blank", "bureaucracy"),
        "flyer": ("flyers",), "magazine": ("magazines",), "handwritten": ("handwritten", "forms_filled")})
    min_mass: float = 0.5                                   # probability the classifier must put on the accepted types
    legacy_font: str = r"kruti|devlys|akruti|shree-?dev|chanakya|surekh|walkman|aps-?dv|shusha|agra|thin_?dv"   # Indic fonts that extract as other letters
    nospace: frozenset = frozenset({"Han", "Hiragana", "Katakana", "Thai", "Lao", "Khmer", "Myanmar", "Tibetan"})
    complex_scripts: frozenset = frozenset({"Devanagari", "Bengali", "Gurmukhi", "Gujarati", "Oriya", "Tamil", "Telugu", "Kannada", "Malayalam", "Sinhala",
                                            "Thai", "Lao", "Khmer", "Myanmar", "Tibetan", "Arabic", "Han", "Hiragana", "Katakana", "Hangul", "Ethiopic",
                                            "Meetei_Mayek"})
    font_pt: dict = field(default_factory=lambda: {"text": (9, 13), "table": (8, 11), "form": (9, 12), "magazine": (8.5, 11), "flyer": (10, 14)})
    max_units: dict = field(default_factory=lambda: {"text": 60, "table": 60, "form": 40, "magazine": 20, "flyer": 25})   # blocks / rows tried per page
    margin_max: float = 0.05                                # margin of a page: 0 to this share of the window, one size per page
    cyrillic: frozenset = frozenset({"ru", "uk", "be", "bg", "kk", "mk", "sr", "ky", "mn", "tg"})
    signatures: dict = field(default_factory=lambda: {"kk": "әғқңөұүһ", "uk": "їєґ", "be": "ў", "ru": "ыэё"})   # letters only that language has
    mojibake: str = r"Ã.|â€|Ð.Ð|Ø.Ù|Â[ -¿]"
    presentation: str = "[\ufb50-\ufdff\ufe70-\ufefc]"   # Arabic presentation forms
    lexicon_langs: dict = field(default_factory=lambda: {"no": "nb", "tl": "fil", "hr": "sh", "bs": "sh"})   # language -> its wordfreq list
    min_lexicon: float = 0.9                                # share of the words a paragraph of lexicon_min_words or more must have in the word list
    min_lexicon_short: float = 0.5                          # the same for a shorter text (headings, list items hold names)
    lexicon_min_words: int = 6
    max_split_words: float = 0.04                           # share of words that are a lone letter or start with a combining mark

    @property
    def pt(self):
        return self.side * 72 / self.dpi                    # the window in PDF points


LANGDS = LangDs()


@dataclass(frozen=True)
class LineBankCfg:
    """Short written lines per language: how they are cut from corpus paragraphs, built, and picked for a field."""
    max_words: int = 10
    min_lines: int = 1500                                   # a language with fewer gets pages of every other folder, then translated lines
    pages_per_lang: int = 150
    tried_factor: int = 6                                   # pages looked at per page used, at most
    page_samples: int = 3                                   # pages taken from one file
    min_page_chars: int = 400
    max_page_width: int = 900                               # pt
    max_paragraphs: int = 60
    skip_folders: tuple = ("handwritten",)
    skip_langs: frozenset = frozenset({"dv"})
    unspaced: frozenset = frozenset({"th", "lo", "km", "my", "bo", "dz", "zh", "ja"})
    unit: dict = field(default_factory=lambda: {"zh": 4, "ja": 5})
    unit_default: int = 6                                   # graphemes counted as one word in a script written without spaces
    edge: str = " \t\r\n,;:.!?()[]{}«»\"'“”‘’„‚‹›…-–—•·|/\\*#"
    sentence: str = r"(?<=[.!?;。։؟।۔])\s+|(?<=[།။።])\s*|\n+"          # the Tibetan, Myanmar and Ethiopic marks need no space after them
    clause: str = r"[,;:()\[\]«»“”„–—|،؛၊፣፤]+| - | • "
    tries: int = 40
    share: tuple = (0.3, 1.0)
    short_cap: int = 12
    short_share: tuple = (0.6, 1.0)
    min_cap: int = 2
    lo_ratio: float = 0.75
    lo_step: float = 0.01
    hi_ratio: float = 1.25
    hi_step: float = 0.03
    fallback: int = 50                                      # longest lines that fit, when the window holds none
    stem_words: int = 2                                     # words that make an opening


@dataclass(frozen=True)
class FillCfg:
    """Answers for form fields: the kind the label names (in English), people per form."""
    keys: tuple = (("postcode", r"post(al)?\s*code|zip\b"), ("phone", r"phone|\btel\b|mobile|\bfax\b"), ("email", r"e-?mail"),
                   ("date", r"\bdate\b|\bborn\b|\bbirth|\bday\b|\bmonth\b|\byear\b"), ("city", r"\bcity\b|\btown\b"),
                   ("country", r"\bcountry\b|nationality"), ("name", r"\bname\b|surname|signature|\bsigned\b|\bsign\b"),
                   ("address", r"address|street"), ("number", r"\bno\b\.?|number|quantity|\bqty\b|amount|\bcount\b|\bage\b|total|price|\bcost\b|\bid\b"))
    # matched against the label translated into English (fill_forms): one table for every language
    tries: int = 8
    numeric: str = r"^[\d\s.,:/+()\-–—%€$₽£¥№x]+$"


@dataclass(frozen=True)
class HwLinesCfg:
    """Boxes around the handwritten text lines of a page image, from its ink."""
    dpi: int = 150
    min_h: int = 15
    max_h: int = 160
    gap: float = 3.0                                        # gap in the ink that splits a row into columns, in row heights
    block: int = 31                                         # adaptive threshold
    offset: int = 15
    rule_div: int = 8                                       # a horizontal or vertical ink run of 1/8 of the page is ruling
    open_kernel: int = 2                                    # opening of the ink: single-pixel noise
    sigma: float = 3.0
    on_share: float = 0.08                                  # profile level of a row, of its maximum
    on_min: float = 4.0
    min_width: float = 2.0                                  # box width, in row heights
    min_height: float = 0.3
    lag: tuple = (25, 200)                                  # line pitch range for the autocorrelation, px
    ac_min: float = 0.15
    valley: float = 0.6                                     # valleys at least this share of the pitch apart
    prominence: float = 0.05


@dataclass(frozen=True)
class HwPickCfg:
    """Real handwritten pages from the corpus: classifier, thresholds, render."""
    hi: float = 0.95
    mid: float = 0.5
    lo: float = 0.35
    cap: int = 15                                           # pages of one language
    n_neg: int = 30000                                      # random corpus pages of other types as negatives
    c: float = 3.0
    side: int = 1024
    quality: int = 90
    neg_labels: frozenset = frozenset({"plain", "formatted", "tables", "magazines", "scientific", "bureaucracy", "flyers", "forms_blank", "maps", "blueprints"})
    chunk: int = 100000


@dataclass(frozen=True)
class TranslateCfg:
    rich: tuple = ("en", "fr", "es", "de", "it", "ru", "pt", "pl", "nl", "cs", "uk", "tr", "ro", "hu", "sv")   # languages with enough text to translate from
    workers: int = 1                                        # parallel requests; Google answers 429 above about one per second
    gap: float = 2.0                                        # pause after each request, per worker
    tries: int = 4                                          # a 429 is waited out: wait * try seconds, 100 s in all
    wait: float = 10.0
    url: str = "https://translate.googleapis.com/translate_a/single"
    margin: float = 2.0


@dataclass(frozen=True)
class ValuesCfg:
    """Short values people write by hand: dates, identifiers, people, e-mail."""
    last_first: frozenset = frozenset({"hu", "ja", "zh", "ko", "vi"})          # family name written first
    joined: frozenset = frozenset({"zh", "ja"})                                 # no space between the names
    month_names: frozenset = frozenset({"en", "de", "fr", "es", "it", "pt", "nl", "sv", "da", "no", "ca", "ro", "tr", "hu", "cs"})   # the month name stands as written in a date
    p_month_name: float = 0.25
    years: tuple = (1995, 2026)
    p_group_digits: float = 0.35
    domains: tuple = ("gmail.com", "mail.com", "yahoo.com", "outlook.com", "hotmail.com", "proton.me", "icloud.com", "live.com", "aol.com", "gmx.com",
                      "yandex.com", "zoho.com", "msn.com", "web.de", "email.com", "inbox.com")
    handle: tuple = (2, 4)                                                      # syllables of an ASCII handle for a name in another script




LINEBANK, FILL, HWLINES, HWPICK, TRANSLATE, VALUES = (LineBankCfg(), FillCfg(), HwLinesCfg(), HwPickCfg(), TranslateCfg(), ValuesCfg())
