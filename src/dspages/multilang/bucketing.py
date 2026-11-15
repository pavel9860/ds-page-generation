"""How alike do the languages' printed pages look?"""
import json
import re
from pathlib import Path

import pymupdf

from . import paths
from . import translate as T
from .buckets import BUCKETS
from .paths import LAYOUTS as _L

LAYOUTS = str(_L) + "/"
PAGES = [("books_pdf/c4e59b61e1ff_manual_en.pdf", 29), ("books_pdf/dfb6ea7dea49_report_en.pdf", 117),
         ("books_pdf/627a082ca739_manual_en.pdf", 59),
         ("table", "purchases"), ("table", "timetable"),
         ("forms/0bc4443ec707_education_en.pdf", 0), ("forms/86baa68c80eb_social_en.pdf", 1)]
KIND = {0: "text", 1: "text", 2: "text", 3: "tables", 4: "tables", 5: "forms", 6: "forms"}
TABLES = {
    "purchases": ["Monthly purchases", ["Item", "Quantity", "Price", "Total"], ["Notebooks", "40", "2.50", "100.00"],
                  ["Pencils", "120", "0.40", "48.00"], ["Folders", "25", "1.20", "30.00"], ["Markers", "60", "0.90", "54.00"],
                  ["Paper", "10", "4.80", "48.00"], ["Total", "", "", "280.00"]],
    "timetable": ["Weekly timetable", ["Day", "Subject", "Room", "Teacher"], ["Monday", "Mathematics", "12", "Mr Stone"],
                  ["Tuesday", "History", "7", "Ms Green"], ["Wednesday", "Chemistry", "3", "Dr Lake"],
                  ["Thursday", "Music", "1", "Ms Hill"], ["Friday", "Geography", "9", "Mr Field"]],
}


def table_blocks(name):
    """Blocks of a written-out table"""
    title, *rows = TABLES[name]
    out = [dict(text=title, bbox=(0, -30, 300, -10), size=12.0, bold=True, role="h")]
    for r, row in enumerate(rows):
        for c, t in enumerate(row):
            if t:
                out.append(dict(text=t, bbox=(c * 100, r * 20, c * 100 + 90, r * 20 + 15), size=10.0, bold=r == 0, role="cell"))
    return out


ALIAS = {"kz", "tj", "fil", "nb"}
NAMES = {"ku": "Kurdish in Arabic script", "ckb": "Central Kurdish (Sorani)", "mni": "Meitei in Meetei Mayek script", "dz": "Dzongkha",
         "mai": "Maithili in Devanagari", "dv": "Dhivehi in Thaana script", "sr": "Serbian in Cyrillic", "ug": "Uyghur in Arabic script",
         "or": "Odia", "as": "Assamese", "ht": "Haitian Creole", "ti": "Tigrinya", "sd": "Sindhi in Arabic script", "uz": "Uzbek (Latin)",
         "mn": "Mongolian (Cyrillic)", "yi": "Yiddish", "sa": "Sanskrit in Devanagari", "bo": "Tibetan", "fa": "Persian", "zh": "Chinese (simplified)"}


def languages():
    return sorted(l for ls in BUCKETS.values() for l in ls.split() if l not in ALIAS)


def prepare(work, only=None):
    """work/src.json (blocks of the source pages with their bbox and size) and work/in_<lang>_<page>.json {lang"""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    src = []
    for f, i in PAGES:
        if f == "table":
            src.append(dict(path="", page=0, blocks=table_blocks(i)))
            continue
        with pymupdf.open(LAYOUTS + f) as doc:
            src.append(dict(path=LAYOUTS + f, page=i, blocks=T.paragraphs(doc[i])))
    (work / "src.json").write_text(json.dumps(src))
    count = 0
    for lang in languages():
        for n, s in enumerate(src):
            if only is not None and (lang, n) not in only:
                continue
            count += 1
            items = [{"id": f"{n}_{k}", "text": b["text"]} for k, b in enumerate(s["blocks"]) if re.search(r"[A-Za-z]{2}", b["text"])]
            (work / f"in_{lang}_{n}.json").write_text(json.dumps(dict(lang=lang, name=NAMES.get(lang, lang), items=items),
                                                               ensure_ascii=False))
    return count


def render(work):
    """work/png/<lang>_<page>.png for every translated batch (out_<lang>_<page>.json)."""
    from hwsynth.writer import _index
    work = Path(work)
    (work / "png").mkdir(exist_ok=True)
    index = _index()
    src = json.loads((work / "src.json").read_text())
    done, failed = 0, []
    for f in sorted(work.glob("out_*.json")):
        lang, n = f.stem[4:].rsplit("_", 1)
        try:
            tr = json.loads(f.read_text())
            s = src[int(n)]
            blocks = [(b["bbox"], b["size"], b["bold"], tr[f"{n}_{k}"]) for k, b in enumerate(s["blocks"]) if f"{n}_{k}" in tr]
            font = T.font_for("".join(b[3] for b in blocks)[:400], index)
            T.render(s["path"], s["page"], [(b[0], b[1], b[3], b[2]) for b in blocks], font, lang, erase="all", dpi=100).save(str(work / "png" / f"{lang}_{n}.png"))
            done += 1
        except Exception as e:
            failed.append((f.name, repr(e)[:60]))
    return done, failed


def embed(work, pagetype_dir=None):
    """work/emb.json {lang: [vector per page]} from the SigLIP 2 embedder of corpus_collecting/pagetype (needs torch)."""
    import sys

    import numpy as np
    from PIL import Image
    sys.path.insert(0, str(pagetype_dir or paths.CORPUS))
    from pagetype.embedder import Embedder
    work = Path(work)
    files = sorted((work / "png").glob("*.png"))
    emb = Embedder()([Image.open(f).convert("RGB") for f in files])
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    out = {}
    for f, e in zip(files, emb, strict=True):
        out.setdefault(f.stem.rsplit("_", 1)[0], []).append(e.tolist())
    (work / "emb.json").write_text(json.dumps(out))
    return len(out)
