from __future__ import annotations

import os
import re

WATERMARK_RE = re.compile(r"\s*oceanofpdf\.com\s*", re.IGNORECASE)
WATERMARK_DOMAIN_RE = re.compile(r"[^\s]*\b[a-z0-9-]+\.(?:com|net|org)\b[^\s]*", re.IGNORECASE)
WATERMARK_DOMAIN_CORE_RE = re.compile(r"[a-z0-9-]+\.(?:com|net|org)", re.IGNORECASE)
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
HYPHEN_WRAP_RE = re.compile(r"(\w)-\s*\n\s*(\w)", re.UNICODE)
SOFT_NEWLINE_RE = re.compile(r"[ \t]*\n[ \t]*")
MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

CONTROL_CHAR_CORRUPTION_RATIO = 0.005
MAX_PARAGRAPH_CHARS = 4000
PAGE_BREAK = "\n\x0c\n"


def join_pages(page_texts: list[str]) -> str:
    return PAGE_BREAK.join(page_texts)


def split_pages(blob: str) -> list[str]:
    return blob.split(PAGE_BREAK)


def cached_txt_path(pdf_path: str, layouts_root: str) -> str | None:
    rel = os.path.relpath(pdf_path, layouts_root)
    parts = rel.split(os.sep)
    if len(parts) != 3 or parts[0] != "books" or parts[1] == "Texts":
        return None
    topic, filename = parts[1], parts[2]
    stem = os.path.splitext(filename)[0]
    return os.path.join(layouts_root, "books", "Texts", topic, f"{stem}.txt")


def load_cached_pages(pdf_path: str, layouts_root: str) -> list[str] | None:
    txt_path = cached_txt_path(pdf_path, layouts_root)
    if txt_path is None or not os.path.isfile(txt_path):
        return None
    with open(txt_path, encoding="utf-8") as fh:
        return split_pages(fh.read())


def _dewrap_paragraph(para: str) -> str:
    para = HYPHEN_WRAP_RE.sub(r"\1\2", para)
    para = SOFT_NEWLINE_RE.sub(" ", para)
    para = MULTI_SPACE_RE.sub(" ", para)
    return para.strip()


def _split_oversized(para: str, max_len: int = MAX_PARAGRAPH_CHARS) -> list[str]:
    if len(para) <= max_len:
        return [para]
    chunks = []
    cur: list[str] = []
    cur_len = 0
    for sentence in SENTENCE_SPLIT_RE.split(para):
        if cur and cur_len + len(sentence) > max_len:
            chunks.append(" ".join(cur))
            cur, cur_len = [], 0
        if len(sentence) > max_len:
            if cur:
                chunks.append(" ".join(cur))
                cur, cur_len = [], 0
            for i in range(0, len(sentence), max_len):
                chunks.append(sentence[i:i + max_len])
            continue
        cur.append(sentence)
        cur_len += len(sentence)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def _is_watermark_paragraph(para: str) -> bool:
    if len(para) < 60 and WATERMARK_DOMAIN_RE.search(para):
        return True
    collapsed = re.sub(r"\s+", "", para)
    if not collapsed:
        return False
    covered = sum(len(m) for m in WATERMARK_DOMAIN_CORE_RE.findall(collapsed))
    return covered / len(collapsed) > 0.5


def clean_plain_text_blob(raw: str) -> str:
    raw = WATERMARK_RE.sub(" ", raw)
    raw = CONTROL_CHAR_RE.sub("", raw)
    raw_paras = re.split(r"\n{2,}", raw)
    paras = [chunk for p in raw_paras for chunk in _split_oversized(p)]
    cleaned = [_dewrap_paragraph(p) for p in paras]
    return "\n\n".join(p for p in cleaned if p and not _is_watermark_paragraph(p))


REPEAT_WINDOW_CHARS = 80
MAX_REPEAT_FRAC = 0.5


def is_degenerate_repetition(text: str, window_chars: int = REPEAT_WINDOW_CHARS,
                             max_repeat_frac: float = MAX_REPEAT_FRAC) -> bool:
    """True if one chunk of text (e.g. a mis-extracted repeated watermark
    or header line) covers most of the extracted text -- valid characters
    throughout, so CONTROL_CHAR_CORRUPTION_RATIO doesn't catch it."""
    flat = text.replace("\n", " ")
    windows = [flat[i:i + window_chars] for i in range(0, len(flat) - window_chars, window_chars)]
    if len(windows) < 4:
        return False
    from collections import Counter
    most_common_count = Counter(windows).most_common(1)[0][1]
    return most_common_count / len(windows) > max_repeat_frac


DOMINANT_IMAGE_AREA_FRAC = 0.10


def extract_pdf_page_text(page) -> str:
    page_area = page.rect.width * page.rect.height
    for info in page.get_image_info():
        x0, y0, x1, y1 = info["bbox"]
        if page_area and (x1 - x0) * (y1 - y0) / page_area > DOMINANT_IMAGE_AREA_FRAC:
            return ""
    blocks = page.get_text("blocks")
    raw_total = sum(len(b[4]) for b in blocks if b[6] == 0)
    control_total = sum(len(CONTROL_CHAR_RE.findall(b[4])) for b in blocks if b[6] == 0)
    if raw_total and control_total / raw_total > CONTROL_CHAR_CORRUPTION_RATIO:
        return ""

    paras = []
    for b in blocks:
        if b[6] != 0:
            continue
        text = WATERMARK_RE.sub(" ", b[4])
        text = CONTROL_CHAR_RE.sub("", text)
        for chunk in _split_oversized(text):
            cleaned = _dewrap_paragraph(chunk)
            if cleaned and not _is_watermark_paragraph(cleaned):
                paras.append(cleaned)
    return "\n\n".join(paras)


_STOP_EN = set("the and of to in is was that he she it with as for on at by an be this his her they were are not"
               .split())
_CYR_RE = re.compile("[\u0400-\u04ff]")


def classify_language(path: str) -> str:
    """en / eu (other Latin script) / cyr, by script and English stopword density of the file's start."""
    txt = open(path, encoding="utf-8", errors="ignore").read(20000)
    letters = [c for c in txt if c.isalpha()]
    if not letters:
        return "eu"
    if sum(1 for c in letters if _CYR_RE.match(c)) / len(letters) > 0.3:
        return "cyr"
    words = re.findall(r"[a-zA-Z]+", txt.lower())
    return "en" if sum(w in _STOP_EN for w in words) / max(1, len(words)) > 0.15 else "eu"
