import os

import fitz
from PIL import Image, ImageDraw, ImageFont

from text_extract import clean_plain_text_blob, join_pages, split_pages
from text_fonts import SCRIPT_FONTS

L = "/run/media/me/D/ML_DS/UVTM/Layouts"
RENDER_LONG_EDGE = 1600
TEXTS_CHARS_PER_PAGE = 1800

_texts_pages_cache: dict[str, list[str]] = {}

CJK_RANGES = ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF), (0x3000, 0x303F))


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in CJK_RANGES)


def _patch_broken_cjk_glyphs(img: Image.Image, page, zoom: float) -> None:
    raw = page.get_text("rawdict")
    spans = [span
             for block in raw.get("blocks", [])
             for line in block.get("lines", [])
             for span in line.get("spans", [])
             if any(_is_cjk(ch["c"]) for ch in span.get("chars", []))]
    if not spans:
        return

    font_path, font_index = SCRIPT_FONTS["zh"]
    draw = ImageDraw.Draw(img)
    font_cache: dict[int, ImageFont.FreeTypeFont] = {}
    for span in spans:
        size = max(6, round(span["size"] * zoom))
        font = font_cache.get(size)
        if font is None:
            font = ImageFont.truetype(font_path, size, index=font_index)
            font_cache[size] = font

        sx0, sy0, sx1, sy1 = [v * zoom for v in span["bbox"]]
        draw.rectangle([sx0, sy0, sx1, sy1], fill=(255, 255, 255))

        x = span["chars"][0]["origin"][0] * zoom
        y = span["chars"][0]["origin"][1] * zoom
        for ch in span["chars"]:
            draw.text((x, y), ch["c"], font=font, fill=(0, 0, 0), anchor="ls")
            x += font.getlength(ch["c"])


def _render_pdf_page(path: str, page_idx: int, long_edge: int, min_short_edge: int = 0) -> Image.Image:
    doc = fitz.open(path)
    page = doc[page_idx]
    rect = page.rect
    zoom = (long_edge + 1) / max(rect.width, rect.height)
    if min_short_edge:
        zoom = max(zoom, min_short_edge / min(rect.width, rect.height))
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    mode = "RGB" if pix.n < 4 else "RGBA"
    img = Image.frombytes(mode, (pix.width, pix.height), pix.samples).convert("RGB")
    _patch_broken_cjk_glyphs(img, page, zoom)
    doc.close()
    return img


def render_pdf_page(path: str, page_idx: int) -> Image.Image:
    return _render_pdf_page(path, page_idx, RENDER_LONG_EDGE, min_short_edge=1030)


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
