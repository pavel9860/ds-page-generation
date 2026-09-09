import glob
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from build_manifest import (L, add_pdf_pages_textaware, lang_from_books_filename,
                             stable_seed, ROWS, PDF_SOURCES)
from build_textures import build_texture
from PIL import Image

random.seed(7)

OUT_TEXT = "/run/media/me/D/ML_DS/UVTM/Layouts/test/textures_texts_200"
OUT_IMG = "/run/media/me/D/ML_DS/UVTM/Layouts/test/textures_pdfimg_200"
os.makedirs(OUT_TEXT, exist_ok=True)
os.makedirs(OUT_IMG, exist_ok=True)

target_text = 200
target_img = 200

book_files = sorted(glob.glob(f"{L}/books/*/*.pdf"))
book_files = [f for f in book_files if os.path.basename(os.path.dirname(f)) != "Texts"]
random.shuffle(book_files)

n_text = 0
for f in book_files:
    if n_text >= target_text:
        break
    before = len(ROWS)
    t, _ = add_pdf_pages_textaware("books", f"books/{os.path.basename(os.path.dirname(f))}", f, lang_from_books_filename)
    new_rows = [r for r in ROWS[before:] if r.fmt == "text_extracted"]
    for r in new_rows:
        if n_text >= target_text:
            break
        rng = random.Random(r.seed)
        row = {"file": r.file, "page": r.page, "format": r.fmt, "lang": r.lang,
               "category": r.category, "source_dataset": r.source_dataset}
        img = build_texture(row, rng)
        img.convert("RGB").save(os.path.join(OUT_TEXT, f"{n_text:03d}_{r.category}.jpg"), quality=92)
        n_text += 1
    del ROWS[before:]
print(f"text_extracted: {n_text}/{target_text}")

img_files = []
for _, source_dataset, pattern, lang in PDF_SOURCES:
    img_files += [(f, "pdf", lang) for f in glob.glob(f"{L}/{pattern}", recursive=True)]
img_files += [(f, "image", "und") for f in glob.glob(f"{L}/commonforms_val_subset/*.png")]
img_files += [(f, "image", "en") for f in glob.glob(f"{L}/DocLayNet-v1.2/*/*.png")]
random.shuffle(img_files)

from build_textures import predict_fill_ok

n_img = 0
for f, fmt, lang in img_files:
    if n_img >= target_img:
        break
    rel = os.path.relpath(f, L)
    page = 0
    if fmt == "pdf":
        import fitz
        try:
            npages = fitz.open(f).page_count
        except Exception:
            continue
        page = random.randrange(npages)
    seed = stable_seed(rel, str(page))
    if not predict_fill_ok(fmt, f, page):
        continue
    row = {"file": rel, "page": page, "format": fmt, "lang": lang, "category": "sample", "source_dataset": "sample", "seed": seed}
    img = build_texture(row, random.Random(seed))
    img.convert("RGB").save(os.path.join(OUT_IMG, f"{n_img:03d}_{fmt}.jpg"), quality=92)
    n_img += 1
print(f"pdf/image: {n_img}/{target_img}")
