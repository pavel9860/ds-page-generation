"""Where the evidence about a legacy font comes from: the corpus documents that use it, the pictures of its text spans and glyphs, and what readers (Tesseract, agents) saw in them.

    Corpus(families)                           documents per font family and their languages (manifest.sqlite)
    traces(fam, docs, paths)                   the text spans of the family in the documents
    typed_codes(fam, ...)                      the typed codes of its spans (no reading)
    ocr_pairs(fam, script, ...)                [(typed codes, text)] read by Tesseract in the script of the font
    dump_lines / dump_sheet                    pictures for an agent to read (.tmp/reads/<family>/): lines.json + NN.png, sheet.png + glyphs.json (second: sheet2.png ...)
    agent_pairs(fam), agent_glyphs(fam)        what the agent wrote back (texts.json, glyph_texts.json + glyph_texts2.json: the glyphs both readings agree on)
    signature(fam)                             a string that changes when the evidence of a family does"""
import collections
import glob
import json
import os
import sqlite3
import unicodedata

import pymupdf
from PIL import Image

from . import legacy as L, legacy_fit as F, legacy_tables as T

LAYOUTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(T.TABLES))))
READS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), ".tmp", "reads")
TESS_OF = {"Devanagari": "hin", "Tamil": "tam", "Telugu": "tel", "Gujarati": "guj", "Oriya": "ori", "Sinhala": "sin", "Khmer": "khm", "Bengali": "ben", "Kannada": "kan",
           "Malayalam": "mal", "Gurmukhi": "pan", "Thai": "tha", "Arabic": "urd", "Myanmar": "mya"}
MIN_CONF, DPI = 75, 300


class Corpus:
    """The documents of the manifest that use a set of font families: .docs {family: [document]}, .paths {document: pdf path}, .lang {document: language}."""
    def __init__(self, families, docs_per_family=12):
        db = sqlite3.connect("file:" + os.path.join(LAYOUTS, "manifest.sqlite") + "?mode=ro", uri=True)
        self.paths = {h: p for h, p in db.execute("select hash, path from files where ext = '.pdf'")}
        self.docs = collections.defaultdict(list)
        for d, f in db.execute("select doc, font from fonts"):
            k = L.family(f)
            if k in families and d in self.paths and len(self.docs[k]) < docs_per_family and d not in self.docs[k]:
                self.docs[k].append(d)
        self.lang = dict(db.execute("select doc, lang from plan"))

    def language(self, fam, script):
        """The most common language of the documents of the family written in the script (else any language of the script)."""
        c = collections.Counter(self.lang.get(d) for d in self.docs[fam] if L.LANG_SCRIPT.get(self.lang.get(d)) == script)
        return c.most_common(1)[0][0] if c else next((lg for lg, sc in L.LANG_SCRIPT.items() if sc == script), None)

    def scan(self, fam, script, limit=150):
        """For a font the manifest's font table does not list: the documents of the languages of the script that use it (every such document is opened: first 6 pages)."""
        from concurrent.futures import ProcessPoolExecutor
        jobs = [(fam, d, self.paths[d]) for d, lg in self.lang.items() if L.LANG_SCRIPT.get(lg) == script and d in self.paths]
        with ProcessPoolExecutor(4) as ex:
            return [d for d in ex.map(_has_font, jobs, chunksize=16) if d][:limit]

    def open(self, doc):
        try:
            return pymupdf.open(os.path.join(LAYOUTS, self.paths[doc]))
        except Exception as e:
            print("unreadable:", self.paths[doc], repr(e)[:100], flush=True)
            return None


def _has_font(job):
    fam, doc, path = job
    try:
        pdf = pymupdf.open(os.path.join(LAYOUTS, path))
        return doc if any(L.family(tr["font"]) == fam for pg in list(pdf)[:6] for tr in pg.get_texttrace()) else None
    except Exception as e:
        print("unreadable:", path, repr(e)[:100], flush=True)
        return None


def traces(corpus, fam, pages=3, min_chars=4):
    """(page, trace) of every text span of the family in its documents (first `pages` pages)."""
    for d in corpus.docs[fam]:
        pdf = corpus.open(d)
        for pg in list(pdf)[:pages] if pdf else []:
            for tr in pg.get_texttrace():
                if L.family(tr["font"]) == fam and len(tr["chars"]) >= min_chars:
                    yield pg, tr


def lines(corpus, fam, pages=3, min_chars=3):
    """(page, typed codes, bbox) of the text lines of the family: the spans of the font on one baseline, left to right, one line (a span of a ligature font is one or two glyphs;
    a gap of more than 3 characters' width starts a new line)."""
    for d in corpus.docs[fam]:
        pdf = corpus.open(d)
        for pg in list(pdf)[:pages] if pdf else []:
            rows = collections.defaultdict(list)
            for tr in pg.get_texttrace():
                if L.family(tr["font"]) == fam and tr["chars"]:
                    rows[round(tr["chars"][0][2][1] / 3)].append(tr)
            for trs in rows.values():
                trs.sort(key=lambda t: t["bbox"][0])
                group = []
                for t in trs + [None]:
                    if group and (t is None or t["bbox"][0] - group[-1]["bbox"][2] > 3 * (group[-1]["bbox"][3] - group[-1]["bbox"][1])):
                        codes = "".join(codes_of(g) for g in group)
                        if len(codes.strip()) >= min_chars:
                            yield pg, codes, (min(g["bbox"][0] for g in group), min(g["bbox"][1] for g in group), max(g["bbox"][2] for g in group), max(g["bbox"][3] for g in group))
                        group = []
                    if t is not None:
                        group.append(t)


def codes_of(tr):
    return "".join(chr(u) for u, g, _, _ in tr["chars"] if g > 0)


def typed_codes(corpus, fam, cap=400, pages=6):
    """The typed codes of the text lines of the font (>= 3 characters), sorted, at most cap of them."""
    out = set()
    for _, codes, _ in lines(corpus, fam, pages):
        out.add(codes)
        if len(out) >= 10 * cap:
            break
    return sorted(out)[:cap]


def span_pictures(corpus, fam, limit, pages=3, min_chars=3):
    """[(typed codes, grey picture of the text line)]."""
    out = []
    for pg, codes, (x0, y0, x1, y1) in lines(corpus, fam, pages, min_chars):
        pm = pg.get_pixmap(clip=pymupdf.Rect(x0 - 2, y0 - 2, x1 + 2, y1 + 2), dpi=DPI, colorspace=pymupdf.csGRAY)
        if pm.width >= 8 and pm.height >= 8:
            out.append((codes, Image.frombytes("L", (pm.width, pm.height), pm.samples)))
        if len(out) >= limit:
            break
    return out


def ocr_pairs(corpus, fam, script, limit=1500, pages=3):
    """[(typed codes, text read by Tesseract)] of the spans, when the script has a Tesseract model."""
    if script not in TESS_OF:
        return []
    import tesserocr
    from dspages.hand.ocr import TESS
    api = tesserocr.PyTessBaseAPI(path=TESS, lang=TESS_OF[script] + "+eng", psm=tesserocr.PSM.SINGLE_LINE)
    out = []
    for codes, im in span_pictures(corpus, fam, limit, pages):
        api.SetImage(im)
        text = api.GetUTF8Text().strip()
        if text and api.MeanTextConf() >= MIN_CONF:
            out.append((codes, text))
    return out


# ------------------------------------------------------------------------------------------------------------------------------------------------- for the agents
def folder(fam):
    path = os.path.join(READS, fam)
    os.makedirs(path, exist_ok=True)
    return path


def _dump_json(fam, name, data):
    json.dump(data, open(os.path.join(folder(fam), name), "w", encoding="utf-8"), ensure_ascii=False)


def _load_json(fam, name):
    f = os.path.join(READS, fam, name)
    return json.load(open(f, encoding="utf-8")) if os.path.exists(f) else None


def dump_lines(corpus, fam, n=60, pages=10, more=False):
    """n distinct text lines of the font (>= 3 characters) as pictures and their codes; returns how many. First round: NN.png + lines.json (the agent writes texts.json); more=True:
    other lines than the ones already dumped, mNN.png + lines2.json (texts2.json)."""
    seen = set(_load_json(fam, "lines.json") or []) if more else set()
    out = []
    for codes, im in span_pictures(corpus, fam, 8 * n, pages):
        if len(codes.strip()) >= 3 and codes not in seen:
            seen.add(codes)
            out.append((codes, im))
    out = out[::max(len(out) // n, 1)][:n]
    for i, (_, im) in enumerate(out):
        im.save(os.path.join(folder(fam), f"{'m' if more else ''}{i:02d}.png"))
    if out:
        _dump_json(fam, "lines2.json" if more else "lines.json", [c for c, _ in out])
    return len(out)


def font_program(corpus, fam):
    """(pymupdf.Font of the embedded program of the family, Counter of the codes its spans use) from its documents."""
    use, font = collections.Counter(), None
    for d in corpus.docs[fam]:
        pdf = corpus.open(d)
        for pg in list(pdf)[:3] if pdf else []:
            use.update(c for tr in pg.get_texttrace() if L.family(tr["font"]) == fam for c in codes_of(tr))
            for xref, _, _, base, *_ in pg.get_fonts():
                if font is None and xref > 0 and L.family(base) == fam and pdf.extract_font(xref)[3]:
                    font = pymupdf.Font(fontbuffer=pdf.extract_font(xref)[3])
        if font and sum(use.values()) > 500:
            break
    return font, use


def dump_sheet(corpus, fam, n=80, second=False):
    """The most used glyphs of the font drawn from its own program on one numbered picture (sheet.png + glyphs.json); second=True: reversed order, bigger cells (sheet2.png +
    glyphs2.json), the independent second reading. Returns the number of glyphs."""
    font, use = font_program(corpus, fam)
    drawn = lambda c: c if font.has_glyph(ord(c)) else chr(0xF000 + ord(c))
    codes = [c for c, _ in use.most_common() if c.strip() and font and (font.has_glyph(ord(c)) or font.has_glyph(0xF000 + ord(c)))][:n]
    if not codes:
        return 0
    cell, cols = (120, 8) if second else (96, 10)
    codes = codes[::-1] if second else codes
    page = pymupdf.open().new_page(width=cols * cell, height=((len(codes) + cols - 1) // cols) * cell)
    tw = pymupdf.TextWriter(page.rect)
    for i, c in enumerate(codes):
        x, y = (i % cols) * cell, (i // cols) * cell
        page.draw_rect(pymupdf.Rect(x, y, x + cell, y + cell), color=(0.7, 0.7, 0.7), width=0.5)
        tw.append((x + 22, y + 62), drawn(c), font=font, fontsize=44)
        page.insert_text((x + 4, y + 14), str(i), fontsize=11, color=(0.8, 0, 0))
    tw.write_text(page)
    tag = "2" if second else ""
    page.get_pixmap(dpi=96).save(os.path.join(folder(fam), f"sheet{tag}.png"))
    _dump_json(fam, f"glyphs{tag}.json", codes)
    return len(codes)


def agent_pairs(fam):
    """[(typed codes, text)] an agent wrote for the pictures of dump_lines (texts.json, and texts2.json for the second round)."""
    out = []
    for lines, texts in (("lines.json", "texts.json"), ("lines2.json", "texts2.json")):
        codes, read = _load_json(fam, lines), _load_json(fam, texts)
        if codes and read:
            out += list(zip(codes, read))
    return out


def agent_glyphs(fam):
    """{typed code: text} where the two independent readings of the sheets agree (blanks and identities left out); {} until both exist."""
    got = []
    for tag in ("", "2"):
        codes, texts = _load_json(fam, f"glyphs{tag}.json"), _load_json(fam, f"glyph_texts{tag}.json")
        if codes and texts:
            got.append({c: F.nfc(t) for c, t in zip(codes, texts) if isinstance(t, str) and t.strip() and t != c})
    return {c: t for c, t in got[0].items() if got[1].get(c) == t} if len(got) == 2 else {}


def signature(fam):
    """A string that changes when the readings (the json files) of the family change; the pictures can be regenerated and are not part of it."""
    path = os.path.join(READS, fam)
    return ",".join(f"{f}:{os.path.getmtime(os.path.join(path, f)):.0f}" for f in sorted(os.listdir(path)) if f.endswith(".json")) if os.path.isdir(path) else ""


def dump_paths(path, page, per_sheet=30, dpi=200, min_px=48, second=False):
    """The lines of text a page draws as glyph paths (legacy.path_glyphs) as numbered pictures for an agent: .tmp/reads/paths/<stem>_<page>/sheetNN.png (per_sheet lines each,
    line k of the page numbered k) + lines.json [glyphs of each line]; the agent writes texts.json [text of each line, "" for a line that is no text]. second=True: the
    independent second reading, 10 lines a sheet (second_NN.png; the agent writes texts2.json). Returns the lines."""
    from PIL import ImageDraw
    pdf = pymupdf.open(path)
    lines = L.path_glyphs(pdf[page])
    if not lines:
        return 0
    out = os.path.join(READS, "paths", f"{os.path.splitext(os.path.basename(path))[0]}_{page}")
    per_sheet = 10 if second else per_sheet
    os.makedirs(out, exist_ok=True)
    for s in range(0, len(lines), per_sheet):
        crops = []
        for k in range(s, min(s + per_sheet, len(lines))):
            box = pymupdf.Rect(lines[k][0])
            for r in lines[k]:
                box |= r
            pad = box.height
            pix = pdf[page].get_pixmap(dpi=max(dpi, round(min_px * 72 / max(box.height, 1))), clip=box + (-pad, -pad / 2, pad, pad / 2))    # small text drawn min_px tall at least
            crops.append((k, Image.frombytes("RGB", (pix.width, pix.height), pix.samples)))
        W, H = max(im.width for _, im in crops) + 60, sum(im.height + 8 for _, im in crops)
        sheet = Image.new("RGB", (W, H), "white")
        y = 0
        for k, im in crops:
            ImageDraw.Draw(sheet).text((4, y + im.height // 2 - 6), str(k), fill=(200, 0, 0))
            sheet.paste(im, (60, y))
            ImageDraw.Draw(sheet).line((0, y + im.height + 4, W, y + im.height + 4), fill=(180, 180, 180))
            y += im.height + 8
        sheet.save(os.path.join(out, f"{'second' if second else 'sheet'}{s // per_sheet:02d}.png"))
    json.dump([len(l) for l in lines], open(os.path.join(out, "lines.json"), "w"))
    return len(lines)


def collect_paths():
    """The agents' two readings of the path lines (.tmp/reads/paths/*/texts.json, texts2.json) -> legacy.PATH_READ: a line's text where both agree (NFC,
    white space aside), else the third reading (texts3.json, of the lines they disagree on) -> [(folder, line)] still undecided."""
    got, open_ = {}, []
    norm = lambda t: unicodedata.normalize("NFC", " ".join(t.split()))
    for d in sorted(glob.glob(os.path.join(READS, "paths", "*", "lines.json"))):
        d = os.path.dirname(d)
        glyphs = json.load(open(os.path.join(d, "lines.json")))
        reads = [json.load(open(f, encoding="utf-8")) for f in (os.path.join(d, f"texts{t}.json") for t in ("", "2", "3")) if os.path.exists(f)]
        if any(len(r) != len(glyphs) for r in reads[:2]):
            raise ValueError(f"{d}: a reading of another number of lines than {len(glyphs)}")
        if len(reads) < 2:
            open_ += [(os.path.basename(d), k) for k in range(len(glyphs))]
            continue
        third = reads[2] if len(reads) > 2 else {}
        texts = [norm(a) if norm(a) == norm(b) else norm(third[str(k)]) if str(k) in third else None for k, (a, b) in enumerate(zip(*reads[:2]))]
        open_ += [(os.path.basename(d), k) for k, t in enumerate(texts) if t is None]
        if None not in texts:
            got[os.path.basename(d)] = dict(glyphs=glyphs, texts=texts)
    json.dump(got, open(L.PATH_READ, "w", encoding="utf-8"), ensure_ascii=False)
    L.path_read.cache_clear()
    return open_
