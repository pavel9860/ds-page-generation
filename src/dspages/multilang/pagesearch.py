"""Pages of a wanted type among the documents of other folders (tables inside bureaucracy files, flyers among other documents, ...)."""

import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .config import LANGDS as C
from .realpages import parse_lang, text_layer_ok


def files_of(layouts, langs):
    """{lang: [path]} of the PDFs and images of the languages in every folder that is not junk."""
    by = {}
    for d in sorted(os.listdir(layouts)):
        if d in C.search_excluded or not (Path(layouts) / d).is_dir():
            continue
        for f in os.listdir(Path(layouts) / d):
            lang = parse_lang(f)
            if lang in langs and f.lower().endswith((".pdf", ".png", ".jpg", ".jpeg")):
                by.setdefault(lang, []).append(str(Path(layouts) / d / f))
    return by


def thumb_job(job):
    """(path, page) -> (path, page, jpeg bytes, digital, words) or None."""
    path, page = job
    import io

    from PIL import Image

    try:
        if path.lower().endswith(".pdf"):
            import pymupdf

            with pymupdf.open(path) as doc:
                if page >= len(doc):
                    return None
                pg = doc[page]
                if pg.rect.width > 900:
                    return None
                words = len(pg.get_text("words"))
                digital = text_layer_ok(pg) if words else True
                k = C.thumb / max(pg.rect.width, pg.rect.height)
                pix = pg.get_pixmap(matrix=pymupdf.Matrix(k, k), colorspace=pymupdf.csRGB)
                im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        else:
            im, words, digital = Image.open(path).convert("RGB"), 0, False
            im.thumbnail((C.thumb, C.thumb))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=90)
        return path, page, buf.getvalue(), digital, words
    except Exception:
        return None


def sample_pages(path, per_file, seed=0):
    if not path.lower().endswith(".pdf"):
        return [(path, 0)]
    import pymupdf

    try:
        with pymupdf.open(path) as doc:
            n = len(doc)
    except Exception:
        return []
    return [(path, p) for p in sorted(random.Random(f"{seed}{path}").sample(range(n), min(n, per_file)))]


def classifier():
    sys.path.insert(0, C.pagetype)
    from pagetype import PageTypeClassifier

    return PageTypeClassifier()


def scan(files, out_jsonl, clf, ex, per_file=None, batch=32):
    """Classifies sampled pages of `files` ([(path, lang)]) with `clf`, rendering them in the pool `ex`"""
    import io

    from PIL import Image

    out_jsonl, per_file = Path(out_jsonl), per_file or C.max_pages
    done = set()
    if out_jsonl.exists():
        done = {(r["path"], r["page"]) for r in map(json.loads, out_jsonl.open(encoding="utf-8"))}
    lang_of = dict(files)
    jobs = [j for p, _ in files for j in sample_pages(p, per_file) if j not in done]
    got = [r for r in ex.map(thumb_job, jobs, chunksize=4) if r]
    with out_jsonl.open("a", encoding="utf-8") as f:
        for i in range(0, len(got), batch):
            part = got[i : i + batch]
            ims = [Image.open(io.BytesIO(r[2])).convert("RGB") for r in part]
            probs = clf.head.predict_proba(clf.embedder(ims))
            for r, p in zip(part, probs, strict=True):
                f.write(
                    json.dumps(
                        dict(
                            path=r[0],
                            page=r[1],
                            lang=lang_of[r[0]],
                            digital=r[3],
                            words=r[4],
                            probs={c: round(v, 4) for c, v in zip(clf.classes, p.tolist(), strict=True)},
                        ),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            f.flush()


def usable(row, typ):
    """True when the page is a `typ` page by the classifier's mass and its text is digital (or it has none)."""
    return sum(row["probs"][c] for c in C.accept[typ]) >= C.min_mass and (row["digital"] or not row["words"])


def folder_of(path):
    return Path(path).parent.name


def search(layouts, lang, needs, cache_dir, found_dir, clf=None, workers=10):
    """Looks in the documents of `lang` for pages of the types in `needs` ({type: pages wanted})"""
    cache = Path(cache_dir) / f"{lang}.jsonl"
    cache.parent.mkdir(parents=True, exist_ok=True)
    rank = {}
    for t in needs:
        for i, d in enumerate(C.search_folders[t]):
            rank[d] = min(rank.get(d, 99), i)
    by = sorted(files_of(layouts, {lang}).get(lang, []), key=lambda p: (rank.get(folder_of(p), 99), p))
    clf = clf or classifier()

    def chosen():
        out, taken = {}, {}
        for r in map(json.loads, cache.open(encoding="utf-8")) if cache.exists() else []:
            for t in needs:
                k = (r["path"], t)
                if usable(r, t) and len(out.setdefault(t, [])) < needs[t] and taken.get(k, 0) < C.per_file_per_type:
                    taken[k] = taken.get(k, 0) + 1
                    out[t].append(dict(r, type=t, scan=not r["words"]))
        return out

    got = {}
    with ProcessPoolExecutor(workers) as ex:
        for i in range(0, len(by), C.search_chunk):
            scan([(p, lang) for p in by[i : i + C.search_chunk]], cache, clf, ex)
            got = chosen()
            if all(len(got.get(t, [])) >= n for t, n in needs.items()):
                break
    Path(found_dir).mkdir(parents=True, exist_ok=True)
    with (Path(found_dir) / f"{lang}.jsonl").open("w", encoding="utf-8") as f:
        for rows in got.values():
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    cache.unlink(missing_ok=True)
    return {t: len(got.get(t, [])) for t in needs}
