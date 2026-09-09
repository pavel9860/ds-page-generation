import os

import fitz
from PIL import Image

from text_extract import clean_plain_text_blob, join_pages, split_pages

L = "/run/media/me/D/ML_DS/UVTM/Layouts"
RENDER_LONG_EDGE = 1600
TEXTS_CHARS_PER_PAGE = 1800

_texts_pages_cache: dict[str, list[str]] = {}


def render_pdf_page(path: str, page_idx: int) -> Image.Image:
    doc = fitz.open(path)
    page = doc[page_idx]
    rect = page.rect
    zoom = max((RENDER_LONG_EDGE + 1) / max(rect.width, rect.height),
               1030 / min(rect.width, rect.height))
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    mode = "RGB" if pix.n < 4 else "RGBA"
    img = Image.frombytes(mode, (pix.width, pix.height), pix.samples).convert("RGB")
    doc.close()
    return img


def _paginate(cleaned: str) -> list[str]:
    pages, cur, cur_len = [], [], 0
    for para in cleaned.split("\n\n"):
        if cur and cur_len + len(para) > TEXTS_CHARS_PER_PAGE:
            pages.append("\n\n".join(cur))
            cur, cur_len = [], 0
        cur.append(para)
        cur_len += len(para)
    if cur:
        pages.append("\n\n".join(cur))
    return pages


def _texts_file_pages(rel_path: str) -> list[str]:
    if rel_path in _texts_pages_cache:
        return _texts_pages_cache[rel_path]

    src_path = os.path.join(L, rel_path)
    cache_path = src_path + ".pages.cache"
    if os.path.isfile(cache_path) and os.path.getmtime(cache_path) >= os.path.getmtime(src_path):
        with open(cache_path, encoding="utf-8") as fh:
            pages = split_pages(fh.read())
    else:
        with open(src_path, encoding="utf-8", errors="ignore") as fh:
            raw = fh.read()
        pages = _paginate(clean_plain_text_blob(raw))
        try:
            with open(cache_path, "w", encoding="utf-8") as fh:
                fh.write(join_pages(pages))
        except OSError:
            pass

    _texts_pages_cache[rel_path] = pages
    return pages
