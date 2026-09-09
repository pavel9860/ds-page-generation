from __future__ import annotations

import csv
import glob
import hashlib
import os
import random
import sys
import time
import unicodedata
from dataclasses import dataclass

sys.stdout.reconfigure(line_buffering=True)

import fitz

sys.path.insert(0, os.path.dirname(__file__))
from build_textures import text_layout_params  # noqa: E402
from text_extract import clean_plain_text_blob, extract_pdf_page_text, load_cached_pages  # noqa: E402
from text_fonts import missing_glyph_ratio  # noqa: E402

MAX_MISSING_GLYPH_RATIO = 0.01

L = "/run/media/me/D/ML_DS/UVTM/Layouts"
OUT_DIR = "/run/media/me/D/ML_DS/UVTM/Layouts/test"
OUT_CSV = os.path.join(OUT_DIR, "text_pdf_manifest.csv")
TOTAL_TARGET = 100_000
SEED = 42
MIN_TEXT_CHARS = 200
TEXTS_CHARS_PER_PAGE = 1800
CODE_LINES_PER_PAGE = 50

BOOK_TOPIC_DIRS = {"technical", "prose_fiction", "reference", "poetry", "graphic_novel",
                    "philosophy", "popular_science", "religious", "political", "humor"}

DOCLAYNET_CATEGORY_MAP = {
    "financial_reports": "table_heavy_record",
    "manuals": "manuals_instructions",
    "laws_and_regulations": "legal_case_law_statute",
    "scientific_articles": "scientific_paper",
    "patents": "patent",
    "government_tenders": "government_tax_forms",
}

EDITION_LANG = {
    "Spanish": "es", "Malaysian": "ms", "Malay": "ms", "Chinese": "zh", "Danish": "da",
    "Greek": "el", "Serbian": "sr", "Turkish": "tr", "Urdu": "ur", "Polish": "pl",
    "Japanese": "ja", "Bulgarian": "bg", "German": "de", "Czech": "cs", "Vietnamese": "vi",
    "Hebrew": "he", "Persian": "fa", "Bengali": "bn", "Korean": "ko", "French": "fr",
    "Italian": "it", "Portuguese": "pt", "Dutch": "nl", "Swedish": "sv", "Finnish": "fi",
    "Hungarian": "hu", "Romanian": "ro",
}

TEXTS_LANG = {
    "eng": "en", "english": "en", "french": "fr", "fr": "fr", "german": "de", "dutch": "nl",
    "czech": "cs", "bosn": "bs", "bulg": "bg", "esp": "es", "spanish": "es", "portuguese": "pt",
    "danish": "da",
}

EXT_LANG = {".py": "python", ".js": "javascript", ".go": "go", ".java": "java", ".rb": "ruby"}

random.seed(SEED)


@dataclass
class Row:
    category: str
    source_dataset: str
    fmt: str
    file: str
    page: int
    lang: str
    seed: int = 0
    note: str = ""


ROWS: list[Row] = []


def stable_seed(*parts: str) -> int:
    h = hashlib.sha1("::".join(parts).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def lang_from_books_filename(base: str) -> str:
    stem = os.path.splitext(base)[0]
    tag = os.path.splitext(stem)[1].lstrip(".")
    if len(tag) == 2 and tag.isalpha():
        return tag.lower()
    for word, code in EDITION_LANG.items():
        if f"_{word}_Edition" in base:
            return code
    for ch in base:
        name = unicodedata.name(ch, "")
        if "HEBREW" in name:
            return "he"
        if "ARABIC" in name:
            return "ar"
        if "CJK" in name or "HIRAGANA" in name or "KATAKANA" in name:
            return "zh"
        if "BENGALI" in name:
            return "bn"
        if "GREEK" in name:
            return "el"
    return "en"


def lang_from_texts_filename(base: str) -> str:
    stem = os.path.splitext(base)[0].lower()
    for token, code in TEXTS_LANG.items():
        if stem.endswith(f"_{token}") or f"_{token}_" in stem or token in stem.split("_"):
            return code
    return "en"


def add_pdf_pages_textaware(category: str, source_dataset: str, pdf_path: str, lang_fn) -> tuple[int, int]:
    rel = os.path.relpath(pdf_path, L)
    cached_pages = load_cached_pages(pdf_path, L)
    doc = None
    if cached_pages is None:
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"  [skip, unopenable] {rel}: {e}")
            return 0, 0
    lang = lang_fn(os.path.basename(pdf_path)) if callable(lang_fn) else lang_fn
    n_text = n_native = 0
    page_count = len(cached_pages) if cached_pages is not None else doc.page_count
    for p in range(page_count):
        if cached_pages is not None:
            text = cached_pages[p]
        else:
            try:
                text = extract_pdf_page_text(doc[p])
            except Exception:
                text = ""
        seed = stable_seed(rel, str(p))
        ok = len(text) >= MIN_TEXT_CHARS
        if ok:
            rng = random.Random(seed)
            layout = text_layout_params(rng, lang)
            ok = missing_glyph_ratio(text, layout["font_path"], layout["font_index"]) <= MAX_MISSING_GLYPH_RATIO
        if ok:
            ROWS.append(Row(category, source_dataset, "text_extracted", rel, p, lang, seed=seed))
            n_text += 1
        else:
            ROWS.append(Row(category, source_dataset, "pdf", rel, p, lang))
            n_native += 1
    if doc is not None:
        doc.close()
    return n_text, n_native


def add_pdf_all_pages(category: str, source_dataset: str, pdf_path: str, lang: str) -> int:
    rel = os.path.relpath(pdf_path, L)
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"  [skip, unopenable] {rel}: {e}")
        return 0
    n = doc.page_count
    doc.close()
    for p in range(n):
        ROWS.append(Row(category, source_dataset, "pdf", rel, p, lang))
    return n


def timed(label: str, fn) -> None:
    t0 = time.time()
    print(f"{label}...")
    fn()
    print(f"  [{time.time()-t0:.1f}s]")


def build_scientific_paper() -> None:
    n = 0
    for f in sorted(glob.glob(f"{L}/scientific_paper/arxiv_pdfs/*.pdf")):
        n += add_pdf_all_pages("scientific_paper", "scientific_paper", f, "en")
    print(f"  {n} pages")


def build_commonforms() -> None:
    n = 0
    for f in sorted(glob.glob(f"{L}/commonforms_val_subset/*.png")):
        rel = os.path.relpath(f, L)
        ROWS.append(Row("business_form_memo_letter", "commonforms_val_subset", "image", rel, 0, "und"))
        n += 1
    print(f"  {n} pages")


def build_xfund_funsd() -> None:
    n = 0
    xfund_dir = f"{L}/XFUND and FUNSD"
    for f in sorted(glob.glob(f"{xfund_dir}/*.train/*.jpg")) + sorted(glob.glob(f"{xfund_dir}/*.val/*.jpg")):
        rel = os.path.relpath(f, L)
        lang = os.path.basename(os.path.dirname(f)).split(".")[0]
        ROWS.append(Row("business_form_memo_letter", "XFUND_FUNSD", "image", rel, 0, lang))
        n += 1
    print(f"  {n} pages")


def build_forms() -> None:
    n_text = n_native = 0
    files = sorted(glob.glob(f"{L}/Forms/**/*.pdf", recursive=True))
    for i, f in enumerate(files, 1):
        t, n = add_pdf_pages_textaware("business_form_memo_letter", "Forms", f, "en")
        n_text += t; n_native += n
        if i % 25 == 0:
            print(f"    [{i}/{len(files)}]")
    print(f"  {n_text} text_extracted + {n_native} native pages")


def build_handwriting() -> None:
    n = 0
    for f in sorted(glob.glob(f"{L}/IAM Handwriting forms/*.png")):
        rel = os.path.relpath(f, L)
        ROWS.append(Row("handwriting", "IAM_Handwriting_forms", "image", rel, 0, "en"))
        n += 1
    print(f"  {n} pages")


def build_dictionary() -> None:
    n = 0
    for f in sorted(glob.glob(f"{L}/dictionares/*.pdf")):
        n += add_pdf_all_pages("dictionary", "dictionares", f, "multi")
    print(f"  {n} pages")


def build_code_listing() -> None:
    n = 0
    for f in sorted(glob.glob(f"{L}/code_listing/**/*", recursive=True)):
        if not os.path.isfile(f):
            continue
        with open(f, encoding="utf-8", errors="ignore") as fh:
            n_lines = sum(1 for _ in fh)
        n_pages = max(1, -(-n_lines // CODE_LINES_PER_PAGE))
        rel = os.path.relpath(f, L)
        prog_lang = EXT_LANG.get(os.path.splitext(f)[1].lower(), "text")
        for p in range(n_pages):
            ROWS.append(Row("code_listing", "code_listing", "text", rel, p, prog_lang))
        n += n_pages
    print(f"  {n} pages")


def build_doclaynet() -> None:
    counts: dict[str, int] = {}
    for doc_category, cat in DOCLAYNET_CATEGORY_MAP.items():
        for f in sorted(glob.glob(f"{L}/DocLayNet-v1.2/{doc_category}/*.png")):
            rel = os.path.relpath(f, L)
            ROWS.append(Row(cat, "DocLayNet-v1.2", "image", rel, 0, "en", note=f"doc_category={doc_category}"))
            counts[cat] = counts.get(cat, 0) + 1
    for cat, n in sorted(counts.items()):
        print(f"  {cat}: {n} pages")


def build_books_technical() -> None:
    n_text = n_native = 0
    for f in sorted(glob.glob(f"{L}/books/technical/*.pdf")):
        t, n = add_pdf_pages_textaware("books_technical", "books/technical", f, lang_from_books_filename)
        n_text += t; n_native += n
    print(f"  {n_text} text_extracted + {n_native} native pages")


def build_books_nontechnical() -> None:
    n_text = n_native = 0
    for f in sorted(glob.glob(f"{L}/books/*/*.pdf")):
        topic = os.path.basename(os.path.dirname(f))
        if topic in ("technical", "Texts"):
            continue
        t, n = add_pdf_pages_textaware("books_nontechnical", f"books/{topic}", f, lang_from_books_filename)
        n_text += t; n_native += n
    print(f"  {n_text} text_extracted + {n_native} native pages")


def load_texts_files() -> list[str]:
    return sorted(
        f for f in glob.glob(f"{L}/books/Texts/**/*.txt", recursive=True)
        if os.path.relpath(f, f"{L}/books/Texts").split(os.sep)[0] not in BOOK_TOPIC_DIRS
    )


def paginate_text_file(path: str) -> list[str]:
    with open(path, encoding="utf-8", errors="ignore") as fh:
        raw = fh.read()
    paras = clean_plain_text_blob(raw).split("\n\n")
    pages: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for para in paras:
        if cur and cur_len + len(para) > TEXTS_CHARS_PER_PAGE:
            pages.append("\n\n".join(cur))
            cur, cur_len = [], 0
        cur.append(para)
        cur_len += len(para)
    if cur:
        pages.append("\n\n".join(cur))
    return pages


def build_texts_and_newspapers(remainder: int) -> None:
    texts_cache: dict[str, list[str]] = {}
    texts_pages: list[tuple[str, int, str]] = []
    for f in load_texts_files():
        rel = os.path.relpath(f, L)
        pages = paginate_text_file(f)
        texts_cache[rel] = pages
        lang = lang_from_texts_filename(os.path.basename(f))
        for p in range(len(pages)):
            texts_pages.append((rel, p, lang))
    texts_pages.sort(key=lambda t: (t[1], t[0]))

    print(f"Texts/ pagination done, {len(texts_pages)} candidate pages")
    texts_target = min(remainder, len(texts_pages))
    n_texts = n_skipped = 0
    for rel, p, lang in texts_pages[:texts_target]:
        seed = stable_seed(rel, str(p))
        rng = random.Random(seed)
        layout = text_layout_params(rng, lang)
        if missing_glyph_ratio(texts_cache[rel][p], layout["font_path"], layout["font_index"]) > MAX_MISSING_GLYPH_RATIO:
            n_skipped += 1
            continue
        ROWS.append(Row("books_nontechnical", "Texts", "text_extracted", rel, p, lang, seed=seed))
        n_texts += 1
    print(f"  {n_texts} pages used ({n_skipped} skipped for font coverage)")

    leftover = remainder - n_texts
    print(f"newspaper_magazine (Pdf/, filling remaining {leftover} pages)...")
    newspaper_files = sorted(glob.glob(f"{L}/Pdf/**/*.pdf", recursive=True))
    random.shuffle(newspaper_files)
    n_news = 0
    for f in newspaper_files:
        if n_news >= leftover:
            break
        rel = os.path.relpath(f, L)
        try:
            doc = fitz.open(f)
        except Exception:
            continue
        n = doc.page_count
        doc.close()
        take = min(n, leftover - n_news)
        for p in range(take):
            ROWS.append(Row("newspaper_magazine", "Pdf", "pdf", rel, p, "und"))
        n_news += take
    print(f"  {n_news} pages")


def write_manifest() -> None:
    random.shuffle(ROWS)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "category", "source_dataset", "format", "file", "page", "lang", "seed", "note"])
        for i, r in enumerate(ROWS):
            w.writerow([i, r.category, r.source_dataset, r.fmt, r.file, r.page, r.lang, r.seed, r.note])
    print(f"wrote {OUT_CSV}")

    totals: dict[str, int] = {}
    for r in ROWS:
        totals[r.category] = totals.get(r.category, 0) + 1
    print("\nper-category totals:")
    for cat, n in sorted(totals.items(), key=lambda t: -t[1]):
        print(f"  {cat:28s} {n:7d}")

    fmt_totals: dict[str, int] = {}
    for r in ROWS:
        fmt_totals[r.fmt] = fmt_totals.get(r.fmt, 0) + 1
    print("\nper-format totals:")
    for fmt, n in sorted(fmt_totals.items(), key=lambda t: -t[1]):
        print(f"  {fmt:16s} {n:7d}")


def main() -> None:
    timed("scientific_paper", build_scientific_paper)
    timed("business_form_memo_letter (commonforms_val_subset)", build_commonforms)
    timed("business_form_memo_letter (XFUND + FUNSD)", build_xfund_funsd)
    timed("business_form_memo_letter (Forms)", build_forms)
    timed("handwriting", build_handwriting)
    timed("dictionary", build_dictionary)
    timed("code_listing", build_code_listing)
    timed("DocLayNet-v1.2", build_doclaynet)
    timed("books_technical", build_books_technical)
    timed("books_nontechnical", build_books_nontechnical)

    remainder = TOTAL_TARGET - len(ROWS)
    print(f"\nfixed categories total: {len(ROWS)}; remainder: {remainder}")
    build_texts_and_newspapers(remainder)

    print(f"\nTOTAL rows: {len(ROWS)} (target {TOTAL_TARGET})")
    write_manifest()


if __name__ == "__main__":
    main()
