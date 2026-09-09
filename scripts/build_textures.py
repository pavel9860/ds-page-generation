import csv
import os
import random
import sys

import fitz
import numpy as np
from PIL import Image, ImageFont

sys.path.insert(0, os.path.dirname(__file__))
import glyph_render
from page_source import _texts_file_pages, render_pdf_page
from text_bounds import detect_content_box_scan
from text_extract import extract_pdf_page_text, load_cached_pages
from text_fonts import is_rtl, pick_font, shape_for_display

L = "/run/media/me/D/ML_DS/UVTM/Layouts"

IMAGE_SIZE = 1024

A4_MM = (210.0, 297.0)
TEXT_SIZE = (1024, round(1024 * A4_MM[1] / A4_MM[0]))
PX_PER_MM = TEXT_SIZE[0] / A4_MM[0]
MM_PER_PT = 0.3528

MAX_MARGIN_FRAC = 0.05
ALIGN_OPTIONS = ["left", "right", "justify"]
FONT_SIZE_PT_RANGE = (8, 16)
LINE_SPACING_RANGE = (1.15, 1.4)
FILL_RANGE = (1.00, 1.00)


def pt_to_px(pt: float) -> int:
    return round(pt * MM_PER_PT * PX_PER_MM)


def random_margins(rng: random.Random) -> tuple[float, float, float, float]:
    return tuple(rng.uniform(0, MAX_MARGIN_FRAC) for _ in range(4))


def render_source_raster(row: dict) -> Image.Image:
    fmt = row["format"]
    path = os.path.join(L, row["file"])
    if fmt == "pdf":
        return render_pdf_page(path, int(row["page"]))
    if fmt == "image":
        return Image.open(path).convert("RGB")
    raise ValueError(fmt)


def _place_axis(b0: float, b1: float, dim: float, side: float, bound: float,
                 m_lo: float, m_hi: float) -> float:
    if side < dim:
        return b0
    slack = side - dim
    lo_share = m_lo / (m_lo + m_hi) if (m_lo + m_hi) > 1e-9 else 0.5
    pos = b0 - slack * lo_share
    return min(max(pos, 0.0), max(0.0, bound - side))


def make_image_texture(row: dict, rng: random.Random) -> Image.Image:
    img = render_source_raster(row)
    W, H = img.size
    bgr = np.array(img)[:, :, ::-1].copy()
    try:
        bx0, by0, bx1, by1 = detect_content_box_scan(bgr)
    except Exception:
        bx0, by0, bx1, by1 = 0, 0, W, H
    bw, bh = max(1.0, bx1 - bx0), max(1.0, by1 - by0)

    m_left, m_right, m_top, m_bottom = random_margins(rng)
    need_w = bw / max(1e-3, 1 - m_left - m_right)
    need_h = bh / max(1e-3, 1 - m_top - m_bottom)
    side = max(min(max(need_w, need_h), W, H), 8.0)

    x0 = _place_axis(bx0, bx1, bw, side, W, m_left, m_right)
    y0 = _place_axis(by0, by1, bh, side, H, m_top, m_bottom)

    crop = img.crop((round(x0), round(y0), round(x0 + side), round(y0 + side)))
    return crop.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)


def _code_lines_for(row: dict, min_lines: int) -> list[str]:
    path = os.path.join(L, row["file"])
    with open(path, encoding="utf-8", errors="ignore") as fh:
        all_lines = fh.read().split("\n")
    start_line = int(row["page"]) * 50
    lines = all_lines[start_line:]
    if len(lines) < min_lines:
        lines = all_lines[max(0, len(all_lines) - min_lines):]
    return lines


def _collect_wrapped(n_pages: int, start: int, min_chars: int, get_page_text) -> str:
    parts: list[str] = []
    total = 0
    for offset in range(n_pages):
        i = (start + offset) % n_pages
        text = get_page_text(i)
        if text:
            parts.append(text)
            total += len(text)
        if total >= min_chars:
            break
    return "\n\n".join(parts)


def _get_text_for_fill(row: dict, min_chars: int) -> str:
    rel = row["file"]
    page_idx = int(row["page"])

    if row["source_dataset"] == "Texts":
        pages = _texts_file_pages(rel)
        return _collect_wrapped(len(pages), page_idx, min_chars, lambda i: pages[i])

    pdf_path = os.path.join(L, rel)
    cached_pages = load_cached_pages(pdf_path, L)
    if cached_pages is not None:
        return _collect_wrapped(len(cached_pages), page_idx, min_chars, lambda i: cached_pages[i])

    doc = fitz.open(pdf_path)
    try:
        def get_page_text(i: int) -> str:
            try:
                return extract_pdf_page_text(doc[i])
            except Exception:
                return ""
        return _collect_wrapped(doc.page_count, page_idx, min_chars, get_page_text)
    finally:
        doc.close()


NO_SPACE_LANGS = {"zh", "ja"}


def _tokenize(text: str, lang: str) -> list[str]:
    if lang in NO_SPACE_LANGS:
        return [ch for ch in text if not ch.isspace()]
    return text.split()


def _token_width(font, font_key: tuple, token: str) -> float:
    return sum(glyph_render.advance(font, font_key, ch) for ch in token)


def _wrap_lines(tokens: list[str], font, font_key: tuple, max_w: float,
                 sep_w: float, max_lines: int) -> list[list[str]]:
    lines, line, line_w = [], [], 0.0
    for token in tokens:
        w = _token_width(font, font_key, token)
        new_w = line_w + (sep_w if line else 0.0) + w
        if line and new_w > max_w:
            lines.append(line)
            if len(lines) >= max_lines:
                return lines
            line, line_w = [token], w
        else:
            line.append(token)
            line_w = new_w
    if line:
        lines.append(line)
    return lines


def _place_line(canvas: glyph_render.PageCanvas, font, font_key: tuple, tokens: list[str],
                 x0: float, x1: float, y: int, align: str, is_last: bool, lang: str, sep: str) -> None:
    rtl = is_rtl(lang)
    text = sep.join(tokens)
    display = shape_for_display(text, lang) if rtl else text

    if align == "justify" and not is_last and len(tokens) > 1 and not rtl and lang not in NO_SPACE_LANGS:
        widths = [_token_width(font, font_key, t) for t in tokens]
        gap = (x1 - x0 - sum(widths)) / max(1, len(tokens) - 1)
        x = x0
        for i, t in enumerate(tokens):
            for ch in t:
                x += canvas.place_char(font, font_key, ch, x, y)
            x += gap if i < len(tokens) - 1 else 0
        return

    w = _token_width(font, font_key, display)
    x = x1 - w if (align == "right" or (rtl and align != "left")) else x0
    for ch in display:
        x += canvas.place_char(font, font_key, ch, x, y)


def text_layout_params(rng: random.Random, lang: str) -> dict:
    m_left, m_right, m_top, m_bottom = random_margins(rng)
    font_size = pt_to_px(rng.randint(*FONT_SIZE_PT_RANGE))
    line_spacing = rng.uniform(*LINE_SPACING_RANGE)
    align = rng.choice(ALIGN_OPTIONS)
    fill = rng.uniform(*FILL_RANGE)
    font_path, font_index = pick_font(lang, rng)
    return dict(m_left=m_left, m_right=m_right, m_top=m_top, m_bottom=m_bottom,
                font_size=font_size, line_spacing=line_spacing, align=align, fill=fill,
                font_path=font_path, font_index=font_index)


def make_text_texture(row: dict, rng: random.Random) -> Image.Image:
    lang = row["lang"]
    out_w, out_h = TEXT_SIZE
    p = text_layout_params(rng, lang)
    x0, x1 = round(p["m_left"] * out_w), out_w - round(p["m_right"] * out_w)
    y0, y1 = round(p["m_top"] * out_h), out_h - round(p["m_bottom"] * out_h)
    content_w, content_h = x1 - x0, y1 - y0

    font_size = p["font_size"]
    line_h = max(font_size + 1, round(font_size * p["line_spacing"]))
    align = p["align"]
    target_lines = max(1, round((content_h // line_h) * p["fill"]))

    font_path, font_index = p["font_path"], p["font_index"]
    font = ImageFont.truetype(font_path, font_size, index=font_index)
    font_key = (font_path, font_index, font_size)

    canvas = glyph_render.PageCanvas(out_w, out_h)

    if row["format"] == "text":
        lines = [ln[:200] for ln in _code_lines_for(row, target_lines + 5)][:target_lines]
        y = y0
        for ln in lines:
            tokens = [w for w in ln.split(" ") if w] or [" "]
            _place_line(canvas, font, font_key, tokens, x0, x1, y, "left", True, lang, " ")
            y += line_h
        return Image.fromarray(canvas.flush(), mode="L").convert("RGB")

    sep = "" if lang in NO_SPACE_LANGS else " "
    sep_w = glyph_render.advance(font, font_key, sep) if sep else 0.0
    avg_char_w = max(1.0, _token_width(font, font_key, "n" * 20) / 20)
    min_chars = int(target_lines * (content_w / avg_char_w) * 1.3)
    tokens = _tokenize(_get_text_for_fill(row, min_chars), lang)

    lines = _wrap_lines(tokens, font, font_key, content_w, sep_w, target_lines)
    y = y0
    for i, line_tokens in enumerate(lines):
        _place_line(canvas, font, font_key, line_tokens, x0, x1, y, align, i == len(lines) - 1, lang, sep)
        y += line_h

    return Image.fromarray(canvas.flush(), mode="L").convert("RGB")


def build_texture(row: dict, rng: random.Random) -> Image.Image:
    fmt = row["format"]
    if fmt in ("pdf", "image"):
        return make_image_texture(row, rng)
    if fmt in ("text_extracted", "text"):
        return make_text_texture(row, rng)
    raise ValueError(fmt)


def main(manifest_path: str, out_dir: str, n: int, seed: int) -> None:
    with open(manifest_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    random.seed(seed)
    sample = random.sample(rows, n)

    os.makedirs(out_dir, exist_ok=True)
    ok = fail = 0
    for i, row in enumerate(sample):
        try:
            rng = random.Random(int(row["seed"]) if row.get("seed") else hash(row["id"]))
            img = build_texture(row, rng)
            out_name = f"{i:03d}_{row['category']}_{row['format']}.jpg"
            img.convert("RGB").save(os.path.join(out_dir, out_name), quality=92)
            ok += 1
        except Exception as e:
            fail += 1
            print(f"  FAILED row {row['id']} ({row['file']}, page {row['page']}): {e}", file=sys.stderr)
    print(f"built {ok}/{n} textures to {out_dir} ({fail} failed)")


if __name__ == "__main__":
    manifest = sys.argv[1] if len(sys.argv) > 1 else "/run/media/me/D/ML_DS/UVTM/Layouts/test/text_pdf_manifest.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "/run/media/me/D/ML_DS/UVTM/Layouts/test/textures_10"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 11
    main(manifest, out, n, seed)
