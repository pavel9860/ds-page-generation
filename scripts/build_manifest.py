from __future__ import annotations

import collections
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
from build_textures import collect_wrapped, predict_fill_ok, required_chars, text_layout_params  # noqa: E402
from text_extract import clean_plain_text_blob, load_cached_pages  # noqa: E402
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

    def __post_init__(self) -> None:
        if not self.seed:
            self.seed = stable_seed(self.file, str(self.page))


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
    lang = lang_fn(os.path.basename(pdf_path)) if callable(lang_fn) else lang_fn
    cached_pages = load_cached_pages(pdf_path, L)
    if cached_pages is None:
        return 0, add_pdf_all_pages(category, source_dataset, pdf_path, lang)

    n_text = n_native = 0
    for p, text in enumerate(cached_pages):
        seed = stable_seed(rel, str(p))
        ok = len(text) >= MIN_TEXT_CHARS
        if ok:
            layout = text_layout_params(random.Random(seed), lang)
            ok = missing_glyph_ratio(text, layout["font_path"], layout["font_index"]) <= MAX_MISSING_GLYPH_RATIO
        if ok:
            min_chars = required_chars(layout)
            available = collect_wrapped(len(cached_pages), p, min_chars, lambda i: cached_pages[i])
            ok = len(available) >= min_chars
        if ok:
            ROWS.append(Row(category, source_dataset, "text_extracted", rel, p, lang, seed=seed))
            n_text += 1
        elif safe_fill_ok("pdf", pdf_path, p):
            ROWS.append(Row(category, source_dataset, "pdf", rel, p, lang, seed=seed))
            n_native += 1
    return n_text, n_native


def safe_fill_ok(fmt: str, path: str, page: int) -> bool:
    try:
        return predict_fill_ok(fmt, path, page)
    except Exception as e:
        print(f"  [skip, detection failed] {path} page {page}: {e}")
        return False


def add_pdf_all_pages(category: str, source_dataset: str, pdf_path: str, lang: str, limit: int = -1) -> int:
    rel = os.path.relpath(pdf_path, L)
    try:
        n = fitz.open(pdf_path).page_count
    except Exception as e:
        print(f"  [skip, unopenable] {rel}: {e}")
        return 0
    kept = 0
    for p in range(n):
        if kept == limit:
            break
        seed = stable_seed(rel, str(p))
        if safe_fill_ok("pdf", pdf_path, p):
            ROWS.append(Row(category, source_dataset, "pdf", rel, p, lang, seed=seed))
            kept += 1
    return kept


def add_image_row(category: str, source_dataset: str, path: str, lang: str, note: str = "") -> bool:
    rel = os.path.relpath(path, L)
    seed = stable_seed(rel, "0")
    ok = safe_fill_ok("image", path, 0)
    if ok:
        ROWS.append(Row(category, source_dataset, "image", rel, 0, lang, seed=seed, note=note))
    return ok


def timed(label: str, fn) -> None:
    t0 = time.time()
    print(f"{label}...")
    fn()
    print(f"  [{time.time()-t0:.1f}s]")


PDF_SOURCES = [
    ("scientific_paper", "scientific_paper", "scientific_paper/arxiv_pdfs/*.pdf", "en"),
    ("business_form_memo_letter", "Forms", "Forms/**/*.pdf", "en"),
    ("dictionary", "dictionares", "dictionares/*.pdf", "multi"),
]


def build_pdf_sources() -> None:
    for category, source_dataset, pattern, lang in PDF_SOURCES:
        n = 0
        for f in sorted(glob.glob(f"{L}/{pattern}", recursive=True)):
            n += add_pdf_all_pages(category, source_dataset, f, lang)
        print(f"  {source_dataset}: {n} pages")


def build_commonforms() -> None:
    n = sum(add_image_row("business_form_memo_letter", "commonforms_val_subset", f, "und")
            for f in sorted(glob.glob(f"{L}/commonforms_val_subset/*.png")))
    print(f"  {n} pages")


def build_xfund_funsd() -> None:
    xfund_dir = f"{L}/XFUND and FUNSD"
    files = sorted(glob.glob(f"{xfund_dir}/*.train/*.jpg")) + sorted(glob.glob(f"{xfund_dir}/*.val/*.jpg"))
    n = sum(add_image_row("business_form_memo_letter", "XFUND_FUNSD", f,
                           os.path.basename(os.path.dirname(f)).split(".")[0]) for f in files)
    print(f"  {n} pages")


def build_handwriting() -> None:
    n = 0
    for f in sorted(glob.glob(f"{L}/IAM Handwriting forms/*.png")):
        ROWS.append(Row("handwriting", "IAM_Handwriting_forms", "image", os.path.relpath(f, L), 0, "en"))
        n += 1
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
    counts: collections.Counter = collections.Counter()
    for doc_category, cat in DOCLAYNET_CATEGORY_MAP.items():
        for f in sorted(glob.glob(f"{L}/DocLayNet-v1.2/{doc_category}/*.png")):
            if add_image_row(cat, "DocLayNet-v1.2", f, "en", note=f"doc_category={doc_category}"):
                counts[cat] += 1
    for cat, n in sorted(counts.items()):
        print(f"  {cat}: {n} pages")


def build_books() -> None:
    n_text = n_native = 0
    for f in sorted(glob.glob(f"{L}/books/*/*.pdf")):
        topic = os.path.basename(os.path.dirname(f))
        if topic == "Texts":
            continue
        category = "books_technical" if topic == "technical" else "books_nontechnical"
        t, n = add_pdf_pages_textaware(category, f"books/{topic}", f, lang_from_books_filename)
        n_text += t; n_native += n
    print(f"  {n_text} text_extracted + {n_native} native pages")


def load_texts_files() -> list[str]:
    return sorted(
        f for f in glob.glob(f"{L}/books/Texts/**/*.txt", recursive=True)
        if os.path.relpath(f, f"{L}/books/Texts").split(os.sep)[0] not in BOOK_TOPIC_DIRS
    )


def paginate_text_file(path: str) -> list[str]:
    with open(path, encoding="utf-8", errors="ignore") as fh:
        paras = clean_plain_text_blob(fh.read()).split("\n\n")
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
        texts_pages += [(rel, p, lang) for p in range(len(pages))]
    texts_pages.sort(key=lambda t: (t[1], t[0]))

    print(f"Texts/ pagination done, {len(texts_pages)} candidate pages")
    texts_target = min(remainder, len(texts_pages))
    n_texts = n_skipped = 0
    for rel, p, lang in texts_pages[:texts_target]:
        seed = stable_seed(rel, str(p))
        layout = text_layout_params(random.Random(seed), lang)
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
        n_news += add_pdf_all_pages("newspaper_magazine", "Pdf", f, "und", limit=leftover - n_news)
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

    for label, counts in [("category", collections.Counter(r.category for r in ROWS)),
                           ("format", collections.Counter(r.fmt for r in ROWS))]:
        print(f"\nper-{label} totals:")
        for k, n in counts.most_common():
            print(f"  {k:28s} {n:7d}")


def main() -> None:
    timed("pdf sources (scientific_paper, Forms, dictionary)", build_pdf_sources)
    timed("business_form_memo_letter (commonforms_val_subset)", build_commonforms)
    timed("business_form_memo_letter (XFUND + FUNSD)", build_xfund_funsd)
    timed("handwriting", build_handwriting)
    timed("code_listing", build_code_listing)
    timed("DocLayNet-v1.2", build_doclaynet)
    timed("books", build_books)

    remainder = TOTAL_TARGET - len(ROWS)
    print(f"\nfixed categories total: {len(ROWS)}; remainder: {remainder}")
    build_texts_and_newspapers(remainder)

    print(f"\nTOTAL rows: {len(ROWS)} (target {TOTAL_TARGET})")
    write_manifest()


if __name__ == "__main__":
    main()
