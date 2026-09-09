import csv
import os
import random
import sys

import fitz
import numpy as np
from PIL import Image, ImageFont

sys.path.insert(0, os.path.dirname(__file__))
import glyph_render
from page_source import _render_pdf_page, _texts_file_pages, render_pdf_page
from text_bounds import classify_patches_scan, detect_content_box_scan, detect_text_box_pdf, patch_fill_ratio_boxes
from text_extract import load_cached_pages
from text_fonts import is_rtl, pick_font, shape_for_display

L = "/run/media/me/D/ML_DS/UVTM/Layouts"

IMAGE_SIZE = 1024

A4_RATIO = 297.0 / 210.0
TEXT_RATIO = 1.0
TEXT_WIDTH_MM = 210.0
TEXT_SIZE = (1024, round(1024 * TEXT_RATIO))
PX_PER_MM = TEXT_SIZE[0] / TEXT_WIDTH_MM
MM_PER_PT = 0.3528

MAX_MARGIN_FRAC = 0.08
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


def _axis_layout(b0: float, b1: float, dim: float, side: float, m_lo: float, m_hi: float,
                  rng: random.Random) -> tuple[float, float, float]:
    """Returns (margin_lo_px, content_src_start, content_len): content_len source pixels
    starting at content_src_start get placed in the canvas starting at margin_lo_px, so the
    margin on both sides of that placed content is always exactly m_lo*side / m_hi*side —
    padded with extra slack if the content is smaller than the margin-reserved span, or
    cropped to that span (never touching the reserved margins) if it's bigger."""
    content_span = side * (1 - m_lo - m_hi)
    if dim <= content_span:
        slack = content_span - dim
        return m_lo * side + rng.uniform(0, slack), b0, dim
    start = rng.uniform(b0, b1 - content_span)
    return m_lo * side, start, content_span


def image_crop_box(bx0: float, by0: float, bx1: float, by1: float,
                    m_left: float, m_right: float, m_top: float, m_bottom: float,
                    rng: random.Random) -> tuple[float, float, float, float, float, float, float]:
    bw = bx1 - bx0
    bh = by1 - by0
    side = max(bw / max(1e-3, 1 - m_left - m_right), 8.0)
    mx, sx, lx = _axis_layout(bx0, bx1, bw, side, m_left, m_right, rng)
    my, sy, ly = _axis_layout(by0, by1, bh, side, m_top, m_bottom, rng)
    return side, mx, sx, lx, my, sy, ly


MIN_PATCH_FILL_PDF = 0.60
MIN_PATCH_FILL_SCAN = 0.60
MAX_PHOTO_PATCH_FRACTION = 0.6
MARGIN_LIM_CAP = 0.2


def _margin_limited_box(content_box: tuple[float, float, float, float],
                         page_w: float, page_h: float) -> tuple[float, float, float, float]:
    cx0, cy0, cx1, cy1 = content_box
    m_left = min(max(cx0, 0.0) / page_w, MARGIN_LIM_CAP)
    m_top = min(max(cy0, 0.0) / page_h, MARGIN_LIM_CAP)
    m_right = min(max(page_w - cx1, 0.0) / page_w, MARGIN_LIM_CAP)
    m_bottom = min(max(page_h - cy1, 0.0) / page_h, MARGIN_LIM_CAP)
    return (m_left * page_w, m_top * page_h, page_w - m_right * page_w, page_h - m_bottom * page_h)


def predict_fill_ok(fmt: str, path: str, page: int) -> bool:
    if fmt == "pdf":
        doc = fitz.open(path)
        pdf_page = doc[page]
        content_box = detect_text_box_pdf(pdf_page)
        boxes = None
        if content_box is not None:
            boxes = list(pdf_page.get_text("words")) + [im["bbox"] for im in pdf_page.get_image_info()]
            page_w, page_h = pdf_page.rect.width, pdf_page.rect.height
        doc.close()
        if content_box is not None:
            measure_box = _margin_limited_box(content_box, page_w, page_h)
            return patch_fill_ratio_boxes(boxes, measure_box) >= MIN_PATCH_FILL_PDF

        # No text layer (scanned page) — only path that needs an actual raster.
        img = _render_pdf_page(path, page, 400)
        bgr = np.array(img)[:, :, ::-1].copy()
        page_h, page_w = bgr.shape[:2]
        content_box = detect_content_box_scan(bgr)
        measure_box = _margin_limited_box(content_box, page_w, page_h)
        n_empty, n_photo, n_text = classify_patches_scan(bgr, measure_box)
        return (n_photo + n_text) / (n_empty + n_photo + n_text) >= MIN_PATCH_FILL_SCAN

    img = Image.open(path).convert("RGB")
    bgr = np.array(img)[:, :, ::-1].copy()

    page_h, page_w = bgr.shape[:2]
    content_box = detect_content_box_scan(bgr)
    measure_box = _margin_limited_box(content_box, page_w, page_h)
    n_empty, n_photo, n_text = classify_patches_scan(bgr, measure_box)
    n_occupied = n_photo + n_text
    if n_occupied / (n_empty + n_occupied) < MIN_PATCH_FILL_SCAN:
        return False
    return (n_photo / n_occupied if n_occupied else 0.0) < MAX_PHOTO_PATCH_FRACTION


def make_image_texture(row: dict, rng: random.Random) -> Image.Image:
    img = render_source_raster(row)
    W, H = img.size
    bgr = np.array(img)[:, :, ::-1].copy()
    bx0, by0, bx1, by1 = detect_content_box_scan(bgr)

    m_left, m_right, m_top, m_bottom = random_margins(rng)
    side, mx, sx, lx, my, sy, ly = image_crop_box(bx0, by0, bx1, by1, m_left, m_right, m_top, m_bottom, rng)

    side_i = round(side)
    canvas = Image.new("RGB", (side_i, side_i), (255, 255, 255))
    sx0, sy0, lx_i, ly_i = round(sx), round(sy), round(lx), round(ly)
    ix0, iy0 = max(sx0, 0), max(sy0, 0)
    ix1, iy1 = min(sx0 + lx_i, W), min(sy0 + ly_i, H)
    if ix1 > ix0 and iy1 > iy0:
        canvas.paste(img.crop((ix0, iy0, ix1, iy1)), (round(mx) + ix0 - sx0, round(my) + iy0 - sy0))
    return canvas.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)


CODE_LINES_PER_PAGE = 50
CODE_TAB_WIDTH = 4


_code_lines_cache: dict[str, list[str]] = {}


def code_lines_for(row: dict) -> list[str]:
    path = os.path.join(L, row["file"])
    all_lines = _code_lines_cache.get(path)
    if all_lines is None:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            all_lines = fh.read().split("\n")
        _code_lines_cache[path] = all_lines
    start = int(row["page"]) * CODE_LINES_PER_PAGE
    return all_lines[start:start + CODE_LINES_PER_PAGE]


def collect_wrapped(n_pages: int, start: int, min_chars: int, get_page_text) -> str:
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


def get_book_text_for_fill(row: dict, min_chars: int) -> str:
    rel = row["file"]
    page_idx = int(row["page"])
    if row["source_dataset"] == "Texts":
        pages = _texts_file_pages(rel)
    else:
        pages = load_cached_pages(os.path.join(L, rel), L)
    return collect_wrapped(len(pages), page_idx, min_chars, lambda i: pages[i])


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


def load_font(font_key: tuple):
    path, index, size = font_key
    return ImageFont.truetype(path, size, index=index)


def text_geometry(p: dict) -> dict:
    out_w, out_h = TEXT_SIZE
    x0, x1 = round(p["m_left"] * out_w), out_w - round(p["m_right"] * out_w)
    y0, y1 = round(p["m_top"] * out_h), out_h - round(p["m_bottom"] * out_h)
    font_size = p["font_size"]
    line_h = max(font_size + 1, round(font_size * p["line_spacing"]))
    target_lines = max(1, round(((y1 - y0) // line_h) * p["fill"]))
    font_key = (p["font_path"], p["font_index"], font_size)
    return dict(out_w=out_w, out_h=out_h, x0=x0, x1=x1, y0=y0, y1=y1,
                content_w=x1 - x0, line_h=line_h, target_lines=target_lines, font_key=font_key)


def required_chars(p: dict, g: dict | None = None, font=None) -> int:
    g = g or text_geometry(p)
    font = font or load_font(g["font_key"])
    avg_char_w = max(1.0, _token_width(font, g["font_key"], "n" * 20) / 20)
    return int(g["target_lines"] * (g["content_w"] / avg_char_w) * 1.3)


def make_text_texture(row: dict, rng: random.Random) -> Image.Image:
    lang = row["lang"]
    p = text_layout_params(rng, lang)
    g = text_geometry(p)
    font = load_font(g["font_key"])
    font_key = g["font_key"]
    canvas = glyph_render.PageCanvas(g["out_w"], g["out_h"])

    sep = "" if lang in NO_SPACE_LANGS else " "
    sep_w = glyph_render.advance(font, font_key, sep) if sep else 0.0
    min_chars = required_chars(p, g, font)
    tokens = _tokenize(get_book_text_for_fill(row, min_chars), lang)

    lines = _wrap_lines(tokens, font, font_key, g["content_w"], sep_w, g["target_lines"])
    y = g["y0"]
    for i, line_tokens in enumerate(lines):
        _place_line(canvas, font, font_key, line_tokens, g["x0"], g["x1"], y, p["align"],
                     i == len(lines) - 1, lang, sep)
        y += g["line_h"]

    return Image.fromarray(canvas.flush(), mode="L").convert("RGB")


def make_code_texture(row: dict, rng: random.Random) -> Image.Image:
    p = text_layout_params(rng, row["lang"])
    g = text_geometry(p)
    font = load_font(g["font_key"])
    font_key = g["font_key"]
    canvas = glyph_render.PageCanvas(g["out_w"], g["out_h"])

    lines = code_lines_for(row)[:g["target_lines"]]
    y = g["y0"]
    for ln in lines:
        x = g["x0"]
        for ch in ln.expandtabs(CODE_TAB_WIDTH):
            if x >= g["x1"]:
                break
            x += canvas.place_char(font, font_key, ch, x, y)
        y += g["line_h"]

    return Image.fromarray(canvas.flush(), mode="L").convert("RGB")


def build_texture(row: dict, rng: random.Random) -> Image.Image:
    fmt = row["format"]
    if fmt in ("pdf", "image"):
        return make_image_texture(row, rng)
    if fmt == "text_extracted":
        return make_text_texture(row, rng)
    if fmt == "text":
        return make_code_texture(row, rng)
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
