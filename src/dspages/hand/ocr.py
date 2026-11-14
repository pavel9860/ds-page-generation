"""A page read by Tesseract with its structure: text lines with positions and confidence, grouped into blocks (paragraph, heading, list
item) with boxes and sizes, ready for structure.to_items.

blocks(pdf_path, page_no, lang) -> [(kind, text, (x0, y0, x1, y1), size)] in pixels of a 200 dpi render, or None
"""
import re

import numpy as np
import pymupdf
from PIL import Image

from .config import PATHS
from .langs import script_ok

TESS = PATHS.tessdata                                       # tessdata_fast of the languages
CODE = {"en": "eng", "hi": "hin", "mr": "mar", "ne": "nep", "bn": "ben", "as": "asm", "ta": "tam", "te": "tel", "kn": "kan", "ml": "mal", "gu": "guj",
        "pa": "pan", "or": "ori", "si": "sin", "km": "khm", "my": "mya", "ur": "urd", "ps": "pus", "sd": "snd", "ku": "kur", "ckb": "kur",
        "sa": "san", "mai": "hin", "am": "amh", "lo": "lao", "th": "tha", "tg": "tgk", "hy": "hye", "bg": "bul"}
INDIC = ("ta", "te", "kn", "ml", "bn", "as", "hi", "mr", "ne", "mai", "sa", "gu", "pa", "or", "si", "my", "km")
LATIN_BITS = re.compile(r"(?<=\s)[A-Za-z]{1,3}(?=\s|$)")
BULLET = re.compile(r"^(?:[-–•*·]|\(?\d{1,2}[.)]|\(?[a-zA-Zऀ-෿][.)])\s")
STOP = re.compile(r"[.!?:;।۔؟。]\W*$")
MIN_CONF = 82                                              # mean word confidence of a line


NO_SPACE = {"th": (0x0e00, 0x0e7f), "lo": (0x0e80, 0x0eff), "km": (0x1780, 0x17ff), "my": (0x1000, 0x109f)}


def unspace(text, lang):
    """Tesseract puts spaces between the letters of scripts that are written without them: removed between two letters of the script."""
    if lang not in NO_SPACE:
        return text
    lo, hi = NO_SPACE[lang]
    inside = lambda c: lo <= ord(c) <= hi
    out = []
    for i, c in enumerate(text):
        if c == " " and 0 < i < len(text) - 1 and inside(text[i - 1]) and inside(text[i + 1]):
            continue
        out.append(c)
    return "".join(out)


def tidy(text, lang):
    """OCR noise: stray short Latin tokens between native words, the ZWNJ some models sprinkle, runs of punctuation."""
    t = text.replace("\u200c", "") if lang in INDIC else text
    t = re.sub(r"[|_~=]{2,}|[-–]{3,}", " ", LATIN_BITS.sub("", t))
    return re.sub(r"\s+", " ", t).strip()


def lines_of(api):
    import tesserocr
    L = tesserocr.RIL
    it = api.GetIterator()
    if it is None:
        return
    while True:
        text = it.GetUTF8Text(L.TEXTLINE)
        box = it.BoundingBox(L.TEXTLINE)
        conf = it.Confidence(L.TEXTLINE)
        para = it.IsAtBeginningOf(L.PARA)
        block = it.IsAtBeginningOf(L.BLOCK)
        if text and box:
            yield text.strip(), box, conf, para, block
        if not it.Next(L.TEXTLINE):
            break


def blocks(path, page_no, lang, dpi=200):
    import tesserocr
    pdf = pymupdf.open(path)
    pix = pdf[page_no].get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    img = Image.fromarray(np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w))
    with tesserocr.PyTessBaseAPI(path=TESS, lang=f"{CODE[lang]}+eng", psm=tesserocr.PSM.AUTO) as api:
        api.SetImage(img)
        api.Recognize()
        ls = [x for x in lines_of(api)]
    paras, cur = [], []
    for text, box, conf, para, block in ls:
        if (para or block) and cur:
            paras.append(cur)
            cur = []
        cur.append((unspace(tidy(text, lang), lang), box, conf))
    if cur:
        paras.append(cur)
    out = []
    for p in paras:
        p = [(t, b, c) for t, b, c in p if t and c >= MIN_CONF]
        if not p:
            continue
        t = re.sub(r"\s+", " ", " ".join(x[0] for x in p)).strip()
        if len(t) < 3 or not script_ok(t, lang, 0.9):
            continue
        x0, y0 = min(b[0] for _, b, _ in p), min(b[1] for _, b, _ in p)
        x1, y1 = max(b[0] + b[2] for _, b, _ in p), max(b[1] + b[3] for _, b, _ in p)
        h = float(np.median([b[3] for _, b, _ in p]))
        out.append(dict(t=t, box=(x0, y0, x1, y1), h=h, n=len(p)))
    if not out:
        return None
    out.sort(key=lambda o: (o["box"][1], o["box"][0]))
    merged = []
    for o in out:                                          # a paragraph the OCR cut at a line break goes on
        q = merged[-1] if merged else None
        if q and not STOP.search(q["t"]) and 0 <= o["box"][1] - q["box"][3] < 0.9 * q["h"] and q["n"] >= 1 and abs(o["h"] - q["h"]) < 0.25 * q["h"] \
                and o["box"][0] < q["box"][2] and not BULLET.match(o["t"]):
            q["t"] += " " + o["t"]
            q["box"] = (min(q["box"][0], o["box"][0]), q["box"][1], max(q["box"][2], o["box"][2]), o["box"][3])
            q["n"] += o["n"]
        else:
            merged.append(dict(o))
    out = merged
    seen, uniq = [], []
    for o in out:                                          # a block the OCR gave twice (columns of several scripts, overlaps)
        key = re.sub(r"\W+", "", o["t"])
        if len(key) >= 12 and any(key in k2 or k2 in key for k2 in seen):
            continue
        seen.append(key)
        uniq.append(o)
    out = uniq
    body = float(np.median([o["h"] for o in out if o["n"] >= 2] or [o["h"] for o in out]))
    res = []
    for o in out:
        k = "p"
        if BULLET.match(o["t"]) and len(o["t"]) <= 600:
            k = "li"
        elif o["n"] <= 2 and len(o["t"]) <= 90 and not STOP.search(o["t"]) and o["h"] >= 1.2 * body:
            k = "h"                                        # a heading is larger than the body text
        res.append((k, o["t"], o["box"], o["h"]))
    return sorted(res, key=lambda r: (r[2][1], r[2][0]))
