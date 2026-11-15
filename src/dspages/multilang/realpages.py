"""Real pages for the language dataset"""

import hashlib
import json
import os
import random
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .config import LANGDS as C


def text_layer_ok(pg):
    """True when the page's text is real digital text"""
    tt = pg.get_texttrace()
    return not any(t["type"] == 3 or re.search(C.legacy_font, t["font"], re.I) for t in tt)


def parse_lang(name):
    m = re.search(r"[_.]([a-z]{2,3})\.(pdf|png|jpg|jpeg)$", name, re.I)
    lang = C.alias.get(m.group(1).lower(), m.group(1).lower()) if m else None
    return None if lang in C.skip_langs else lang


def candidates(layouts, langs, per=30, seed=0):
    """[(path, lang, type)]: up to `per` files per language and type, chosen at random."""
    by = {}
    for typ, folders in C.folders.items():
        for d in folders:
            for f in os.listdir(Path(layouts) / d):
                lang = parse_lang(f)
                if lang in langs and f.lower().endswith((".pdf", ".png", ".jpg", ".jpeg")):
                    by.setdefault((lang, typ), []).append(str(Path(layouts) / d / f))
    out = []
    for (lang, typ), fs in sorted(by.items()):
        random.Random(f"{seed}{lang}{typ}").shuffle(fs)
        out += [(f, lang, typ) for f in fs[:per]]
    return out


def _window_pdf(pg):
    import pymupdf

    bl = [b for b in pg.get_text("blocks") if b[6] == 0 and len(b[4].split()) >= 3]
    r = pg.rect
    if bl:
        cx = (min(b[0] for b in bl) + max(b[2] for b in bl)) / 2
        top = min(b[1] for b in bl)
    else:
        cx, top = (r.x0 + r.x1) / 2, r.y0
    x0 = min(max(cx - C.pt / 2, r.x0), max(r.x0, r.x1 - C.pt))
    y0 = min(max(top - 12, r.y0), max(r.y0, r.y1 - C.pt))
    clip = pymupdf.Rect(x0, y0, x0 + C.pt, y0 + C.pt) & r
    return clip


def make_row(job):
    """(path, lang, type, out_dir) -> row dict or None (see the module doc)."""
    path, lang, typ, out = job
    from PIL import Image

    from .langcheck import check_text

    try:
        pdf = path.lower().endswith(".pdf")
        row = dict(src=path, lang=lang, type=typ, scan=False)
        if pdf:
            import pymupdf

            with pymupdf.open(path) as doc:
                order = random.Random(path).sample(range(len(doc)), min(len(doc), 6))
                pick = None
                for i in order:
                    n = len(doc[i].get_text("words"))
                    if n >= 30 or typ == "handwritten":
                        pick = i
                        break
                if pick is None:
                    return None
                pg = doc[pick]
                if pg.rect.width > 900:
                    return None
                clip = _window_pdf(pg)
                digital = text_layer_ok(pg)
                text = pg.get_text("text", clip=clip) if digital else ""
                from hwsynth.scripts import text_script

                k = C.dpi / 72
                row["boxes"] = [
                    dict(
                        k="print",
                        b=[
                            round((v - o) * k)
                            for v, o in zip(ln["bbox"], (clip.x0, clip.y0, clip.x0, clip.y0), strict=True)
                        ],
                        script=text_script(t),
                    )
                    for b in pg.get_text("dict", clip=clip)["blocks"]
                    for ln in b.get("lines", [])
                    if digital and (t := "".join(sp["text"] for sp in ln["spans"]).strip())
                ]
                row["script"] = text_script(text[:400] or "a")
                row.update(page=pick)
                tz = pg.get_pixmap(
                    matrix=pymupdf.Matrix(
                        384 / max(pg.rect.width, pg.rect.height), 384 / max(pg.rect.width, pg.rect.height)
                    ),
                    colorspace=pymupdf.csRGB,
                )
                thumb = Image.frombytes("RGB", (tz.width, tz.height), tz.samples)
                pix = pg.get_pixmap(dpi=C.dpi, clip=clip, colorspace=pymupdf.csGRAY)
                im = Image.frombytes("L", (pix.width, pix.height), pix.samples)
        else:
            im0 = Image.open(path).convert("L")
            im0 = im0.resize((1240, round(im0.height * 1240 / im0.width)))
            im = im0.crop((108, 0, 108 + C.side, min(im0.height, C.side)))
            text = ""
            row.update(page=0, scan=True)
            thumb = im0.convert("RGB").copy()
            thumb.thumbnail((384, 384))
        if text.strip() and len(text.split()) >= 20:
            w = text.split()
            mid = max(0, len(w) // 2 - 30)
            status, det = check_text(
                " ".join(w[mid : mid + 60]) if " " in text[:300] else text[len(text) // 2 - 100 : len(text) // 2 + 100],
                lang,
            )
            row.update(lang_status=status, lang_detected=det)
            if status in ("mismatch", "short"):
                return None
            if status == "other":
                row["lang"] = det
        else:
            row.update(lang_status="unverified", lang_detected=None, scan=True)
        canvas = Image.new("L", (C.side, C.side), 255)
        canvas.paste(im, (0, 0))
        rid = hashlib.sha1(f"{path}{row['page']}".encode()).hexdigest()[:12]
        d = Path(out) / "real" / typ / row["lang"]
        d.mkdir(parents=True, exist_ok=True)
        canvas.save(d / f"{rid}.jpg", quality=88)
        (Path(out) / "thumb").mkdir(parents=True, exist_ok=True)
        thumb.save(Path(out) / "thumb" / f"{rid}.jpg", quality=90)
        row.update(id=rid, file=f"real/{typ}/{row['lang']}/{rid}.jpg", was_lang=lang)
        return row
    except Exception:
        if os.environ.get("HWDEBUG"):
            raise
        return None


def _run_chunk(fn, jobs, workers):
    """fn over jobs in one pool; if a worker dies on a page (a crash in the PDF library) the chunk is redone job by job, each in a process of its own, and the crashing job gives nothing."""
    from concurrent.futures.process import BrokenProcessPool

    try:
        with ProcessPoolExecutor(workers) as ex:
            return [r for r in ex.map(fn, jobs, chunksize=4) if r]
    except BrokenProcessPool:
        out = []
        for j in jobs:
            try:
                with ProcessPoolExecutor(1) as ex:
                    r = ex.submit(fn, j).result()
                    out += [r] if r else []
            except BrokenProcessPool:
                pass
        return out


def build_real(layouts, langs, out, per=30, workers=10, chunk=600):
    """Rows of the rechecked real pages (also written to out/real.jsonl); the pool is started once per `chunk` jobs."""
    os.environ["OMP_NUM_THREADS"] = "1"
    jobs = [(f, l, t, str(out)) for f, l, t in candidates(layouts, set(langs), per)]
    rows = []
    for i in range(0, len(jobs), chunk):
        rows += _run_chunk(make_row, jobs[i : i + chunk], workers)
    Path(out).mkdir(parents=True, exist_ok=True)
    with open(Path(out) / "real.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


def paragraph_pool(job):
    """(layouts, lang, out_dir, files) -> {role"""
    layouts, lang, out, nfiles = job
    import pymupdf

    from dspages.pagegen.source import scripts_of

    from .langcheck import check_text, clean_ok, repair
    from .translate import paragraphs

    fs = [
        str(Path(layouts) / d / f)
        for d in C.folders["text"]
        for f in os.listdir(Path(layouts) / d)
        if parse_lang(f) == lang and f.lower().endswith(".pdf")
    ]
    random.Random(lang).shuffle(fs)
    pool = {"p": [], "h": [], "li": []}
    for path in fs[: nfiles * 2]:
        if len(pool["p"]) >= 300:
            break
        try:
            with pymupdf.open(path) as doc:
                for i in random.Random(path).sample(range(len(doc)), min(len(doc), 4)):
                    pg = doc[i]
                    if pg.rect.width > 900 or len(pg.get_text("text")) < 400 or not text_layer_ok(pg):
                        continue
                    if "Arabic" in scripts_of(lang) and re.search("ا[إأآ]ل", pg.get_text("text")):
                        continue
                    ps = paragraphs(pg, 40)
                    body = " ".join(p["text"] for p in ps if p["role"] == "p")
                    if check_text(body[:700], lang)[0] not in ("ok", "short"):
                        continue
                    for p in ps:
                        t = p["text"] = repair(p["text"])
                        if not clean_ok(t, lang) or (
                            p["role"] == "p" and check_text(t, lang)[0] not in ("ok", "short")
                        ):
                            continue
                        n = len(t.split()) if " " in t[:200] else len(t) // 4
                        if (
                            p["role"] == "p"
                            and 10 <= n <= 120
                            and p["text"].rstrip()[-1:] in ".!?;:。։؟।"
                            and not t[:1].islower()
                        ):
                            pool["p"].append(p["text"])
                        elif p["role"] in ("h", "li") and 2 <= n <= 14:
                            pool[p["role"]].append(p["text"])
        except Exception:
            if os.environ.get("HWDEBUG"):
                raise
            continue
    d = Path(out) / "paras"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{lang}.json").write_text(json.dumps(pool, ensure_ascii=False))
    return lang, {k: len(v) for k, v in pool.items()}
