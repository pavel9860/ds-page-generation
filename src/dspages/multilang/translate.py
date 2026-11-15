"""Pages for slots (language, type) that have no source page"""

import json
import re
from pathlib import Path

import pymupdf
import regex

PROMPT = """Translate every `text` into {language}. These are lines of a document page (a form, a table or running text): keep
numbers, dates, codes, e-mail addresses and punctuation, translate words, keep it short enough to take the room of the original.
Write the text in {language}'s own script. Write {{id: translation}} for every item."""
WORD2, WORD3 = (
    regex.compile(r"(?:\p{L}\p{M}*){%d}" % n) for n in (2, 3)
)  # letters with their marks (a vowel sign is no letter in re)
RTL = {"ar", "fa", "ur", "ps", "ckb", "ku", "sd", "ug", "he", "yi", "dv"}


def lines(page, lang=None):
    """[{text, bbox, size}] of the text lines of a page that hold a word (numbers and codes alone are left as printed), read by the one page reader
    (dspages.hand.texts.page_lines: legacy fonts of lang decoded)."""
    from dspages.hand.texts import page_lines

    return [
        dict(text=l["text"], bbox=l["bbox"], size=l["size"]) for l in page_lines(page, lang) if WORD2.search(l["text"])
    ]


def paragraphs(page, max_blocks=25, lang=None):
    """[{text, bbox, size, bold, role}] of the page as the reader sees it (dspages.hand.texts.page_paragraphs), the max_blocks largest in reading order."""
    from dspages.hand.texts import page_paragraphs

    out = [o for o in page_paragraphs(page, lang) if WORD3.search(o["text"])]
    return sorted(
        sorted(out, key=lambda o: -(o["bbox"][2] - o["bbox"][0]) * (o["bbox"][3] - o["bbox"][1]))[:max_blocks],
        key=lambda o: (o["bbox"][1], o["bbox"][0]),
    )


def batches(pages, lang, work_dir, per_file=40):
    """in_NNNN.json of the lines of the pages [(path, page)] and meta.json {id"""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    items, meta = [], {}
    for path, no in pages:
        with pymupdf.open(path) as doc:
            for k, ln in enumerate(lines(doc[no])):
                i = f"{len(items):06d}"
                items.append({"id": i, "text": ln["text"]})
                meta[i] = dict(path=path, page=no, bbox=ln["bbox"], size=ln["size"])
    for n in range(0, len(items), per_file):
        (work_dir / f"in_{n // per_file:04d}.json").write_text(json.dumps(items[n : n + per_file], ensure_ascii=False))
    (work_dir / "prompt.txt").write_text(PROMPT.format(language=lang))
    (work_dir / "meta.json").write_text(json.dumps(meta))
    return len(items)


def translations(work_dir):
    got = {}
    for f in sorted(Path(work_dir).glob("out_*.json")):
        got.update(json.loads(f.read_text()))
    return got


def render(path, page_no, items, font, lang, out=None, drawn=(), erase="items", dpi=None):
    """Page `page_no` of `path` with text replaced: each item (bbox, size, text[, bold]) set in the font file `font` (shaped, right to left for RTL
    languages, wrapped in its box, shrunk to fit). erase: "items" (only the items' boxes) or "all" (every text block; pictures and rules stay).
    drawn: boxes of text drawn as glyph outlines, their drawings go too. Saves the page to the pdf `out` when given; returns its pixmap (grey, `dpi`)
    when dpi is given."""
    with pymupdf.open(path) as src, pymupdf.open() as doc:
        doc.insert_pdf(
            src, from_page=page_no, to_page=page_no, widgets=erase != "all"
        )  # the page alone, without the document's structure tree (a broken one
        # crashes MuPDF in select); erase "all": without its form fields too (their text would stay; a cyclic field tree overflows MuPDF's stack)
        pg = doc[0]
        if drawn:
            for bbox in drawn:
                pg.add_redact_annot(
                    pymupdf.Rect(bbox) + (-1, -1, 1, 1), fill=None
                )  # a point of margin: a glyph's ink may reach past the box of its outline points
            pg.apply_redactions(
                images=pymupdf.PDF_REDACT_IMAGE_NONE, graphics=pymupdf.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED
            )
        boxes = (
            [b["bbox"] for b in pg.get_text("dict")["blocks"] if b["type"] == 0]
            if erase == "all"
            else [it[0] for it in items]
        )
        for bbox in boxes:
            pg.add_redact_annot(pymupdf.Rect(bbox), fill=None)
        pg.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE, graphics=pymupdf.PDF_REDACT_LINE_ART_NONE)
        font = Path(font)
        arch = pymupdf.Archive(str(font.parent))
        d = ' dir="rtl"' if lang in RTL else ""
        for bbox, size, text, *bold in items:
            r = pymupdf.Rect(bbox)
            r.x1 = max(
                r.x1, r.x0 + 3 * size
            )  # a box narrower than three characters (a short label) takes its text anyway
            html = f"<p{d}>{text.replace('&', '&amp;').replace('<', '&lt;')}</p>"
            css = (
                f"@font-face {{font-family: tf; src: url({font.name});}} p {{font-family: tf; font-size: {size:.1f}pt; margin: 0; line-height: 1.15; "
                f"{'font-weight: bold;' if bold and bold[0] else ''}}}"
            )
            pg.insert_htmlbox(r, html, css=css, archive=arch, scale_low=0.3)
        if out:
            doc.subset_fonts()  # insert_htmlbox embeds the whole font file per item: only the glyphs used, once
            doc.save(out, garbage=3, deflate=True)  # (garbage=4 merges look-alike objects and can change the page)
        return pg.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY) if dpi else None


def font_for(text, index):
    """A print font file covering the text, Noto first (plain text faces), from the hwsynth font index."""
    ok = [f for f in index.cmap if index.covers(f, text)]
    if not ok:
        return max(index.cmap, key=lambda f: sum(ord(c) in index.cmap[f] for c in text if not c.isspace()))
    return (
        next((f for f in ok if "Noto" in f and "Sans" in f), None)
        or next((f for f in ok if "Noto" in f), None)
        or ok[0]
    )


def fonts_for(text, index, k=6):
    """Up to k print-face font files covering the text (fonts of the print class, no display, pixel or symbol faces; Noto first), for variety between the rendered pages; the hand-class fonts only when a script has no print font."""
    bad = r"Symbol|Emoji|Icon|Display|Mono|Math|Handjet|Pop|Pixel|Bungee|Press"
    ok = [f for f in index.cmap if index.covers(f, text) and not re.search(bad, f)]
    pr = [f for f in ok if index.meta[f].get("class") == "print"] or ok
    plain = [
        f
        for f in pr
        if re.search(
            r"Noto|GoogleSans|Roboto|OpenSans|Inter|Lato|SourceSans|PTSans|Merri|IBMPlex|Arimo|Tinos|DejaVu", f
        )
    ]
    return sorted(plain or pr, key=lambda f: ("Noto" not in f, f))[:k] or [font_for(text, index)]
