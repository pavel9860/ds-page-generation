"""Tunable numbers of this package, one section per module."""

# content
SCAN_MAX_SIDE = 512
LINE_KSIZE = 15
LINE_THRESH = 36
PATCH_INK_EPS = 0.01
WIDTHS = 6
POSITIONS = 9

# deskew
_DOWNSCALE_PX = 800
_MAX_DEG = 15.0
_COARSE_STEP = 1.0
_N_STRIPS = 5
_SKIP_ANGLE_DEG = 0.25
_MAX_APPLIED_DEG = 10.0
_MIN_STRIP_INK = 0.01
_MIN_GAIN = 1.1
_MIN_STRIPS = 3
_MAX_SPREAD_DEG = 1.0

# generate
INK_BORDER = 0.005

# indic
PREFIX = {
    0x093F,
    0x09BF,
    0x09C7,
    0x09C8,
    0x0A3F,
    0x0ABF,
    0x0BC6,
    0x0BC7,
    0x0BC8,
    0x0D46,
    0x0D47,
    0x0D48,
    0x0DD9,
    0x0DDA,
    0x0DDB,
    0x17C1,
    0x17C2,
    0x17C3,
    0x1031,
    0x0B47,
    0x0B48,
}  # + Oriya e, ai (drawn left of the consonant)
RA = {0x0930, 0x09B0, 0x0AB0, 0x0A30, 0x0B30}
VIRAMA = {0x094D, 0x09CD, 0x0ACD, 0x0A4D, 0x0BCD, 0x0C4D, 0x0CCD, 0x0D4D, 0x0DCA, 0x17D2, 0x1039, 0x0B4D}
NUKTA = {0x093C, 0x09BC, 0x0ABC, 0x0A3C, 0x0B3C}

# legacy
SPECS = {
    "Devanagari": ("NotoSansDevanagari-Regular.ttf", [(0x0900, 0x097F)], 3),
    "Bengali": ("NotoSansBengali-Regular.ttf", [(0x0980, 0x09FF)], 3),
    "Gurmukhi": ("NotoSansGurmukhi-Regular.ttf", [(0x0A00, 0x0A7F)], 3),
    "Gujarati": ("NotoSansGujarati-Regular.ttf", [(0x0A80, 0x0AFF)], 3),
    "Oriya": ("NotoSansOriya-Regular.ttf", [(0x0B00, 0x0B7F)], 3),
    "Tamil": ("NotoSansTamil-Regular.ttf", [(0x0B80, 0x0BFF)], 3),
    "Telugu": ("NotoSansTelugu-Regular.ttf", [(0x0C00, 0x0C7F)], 3),
    "Kannada": ("NotoSansKannada-Regular.ttf", [(0x0C80, 0x0CFF)], 3),
    "Malayalam": ("NotoSansMalayalam-Regular.ttf", [(0x0D00, 0x0D7F)], 3),
    "Sinhala": ("NotoSansSinhala-Regular.ttf", [(0x0D80, 0x0DFF)], 3),
    "Thai": ("NotoSansThai-Regular.ttf", [(0x0E00, 0x0E7F)], 2),
    "Lao": ("NotoSansLao-Regular.ttf", [(0x0E80, 0x0EFF)], 2),
    "Tibetan": ("NotoSerifTibetan-Regular.ttf", [(0x0F00, 0x0FFF)], 3),
    "Myanmar": ("NotoSansMyanmar-Regular.ttf", [(0x1000, 0x109F)], 3),
    "Khmer": ("NotoSansKhmer-Regular.ttf", [(0x1780, 0x17FF)], 3),
    "Ethiopic": ("NotoSansEthiopic-Regular.ttf", [(0x1200, 0x139F)], 1),
    "Armenian": ("NotoSansArmenian-Regular.ttf", [(0x0530, 0x058F)], 1),
    "Georgian": ("NotoSansGeorgian-Regular.ttf", [(0x10A0, 0x10FF)], 1),
    "Hebrew": ("NotoSansHebrew-Regular.ttf", [(0x0590, 0x05FF)], 1),
    "Arabic": ("NotoNaskhArabic-Regular.ttf", [(0x0600, 0x06FF)], 2),
    "Cyrillic": ("NotoSans-Regular.ttf", [(0x0400, 0x04FF)], 1),
    "Greek": ("NotoSans-Regular.ttf", [(0x0370, 0x03FF), (0x1F00, 0x1FFF)], 1),  # + Greek Extended (polytonic)
    "Latin": ("NotoSans-Regular.ttf", [(0x0020, 0x024F)], 1),
}
CJK = {"Han": [(0x4E00, 0x9FFF)], "Kana": [(0x3040, 0x30FF)], "Hangul": [(0xAC00, 0xD7A3)]}
GRID = 24
SIZE = 96
CJK_LANG = {
    "zh": (2, ("Han",)),
    "ja": (0, ("Han", "Kana")),
    "ko": (1, ("Hangul", "Han")),
}  # Noto Sans CJK collection index, character sets
SAME = 0.9  # distance under which two outlines are the same glyph (576-d L2 plus the weighted geometry)
FIT_PAGES = 3  # pages of the document whose text the table of a font is fitted on
NATIVE = 0.3  # a font whose characters are this share in the script's ranges is real Unicode text
DRAWN = 0.95  # drawn_scripts: a script's glyph distance under this share of Latin's
SCREEN = 600  # characters a candidate table must turn into the script before it is scored
COVER = 0.9  # share of the sample's letters a labelled table must know (it leaves the others as typed)
TRUST = 2 / 3  # share of codes the glyph-shape fit must read as some candidate table does to count
PATH_GLYPHS = 50  # a page with this many glyph outlines sets (some of) its text as outlines

# legacy_fit
MIN_LINES = 3  # a code needs this many spans to be written

# paper
YELLOW_LUMA = 0.24  # Rec. 709 luma of a yellow cast (0, 0.25, 0.8) per unit strength
STAIN_LUMA = 0.57  # of a brown stain (0.3, 0.6, 1.0)
BLUR_LEVELS = (0.0, 0.7, 1.4, 2.4, 3.6, 5.0)  # px, the pre-blurred levels of a variable blur
CLOUD_STEP = 4  # clouds (>= 2 mm) are evaluated on every 4th pixel

# pictures
STRIDE = 32

# refext
SIMILAR_COST = 0.4  # distance added to a match found in another script's reference

# reffonts
MAX_APART = 2  # search: the glyph texts with up to this many neighbours swapped

# refont
STYLE_OF = {
    "": 0,
    "regular": 0,
    "roman": 0,
    "book": 0,
    "bold": 1,
    "italic": 2,
    "oblique": 2,
    "bolditalic": 3,
    "boldoblique": 3,
}
GSUB_ROUNDS = 8  # passes over a font's substitutions to name the glyphs made from glyphs made from mapped ones
THAI_PUA = {
    0xF700: 0x0E10,
    0xF701: 0x0E34,
    0xF702: 0x0E35,
    0xF703: 0x0E36,
    0xF704: 0x0E37,
    0xF705: 0x0E48,
    0xF706: 0x0E49,
    0xF707: 0x0E4A,
    0xF708: 0x0E4B,
    0xF709: 0x0E4C,
    0xF70A: 0x0E48,
    0xF70B: 0x0E49,
    0xF70C: 0x0E4A,
    0xF70D: 0x0E4B,
    0xF70E: 0x0E4C,
    0xF70F: 0x0E0D,
    0xF710: 0x0E31,
    0xF711: 0x0E4D,
    0xF712: 0x0E47,
    0xF713: 0x0E48,
    0xF714: 0x0E49,
    0xF715: 0x0E4A,
    0xF716: 0x0E4B,
    0xF717: 0x0E4C,
    0xF718: 0x0E38,
    0xF719: 0x0E39,
    0xF71A: 0x0E3A,
}
TRACE_NAME = 24  # characters of a font name MuPDF keeps in a text trace

# sources
RASTER_ZOOM = 2.0

# text
_MIN_SCRIPT_CMAP_SIZE = 40
