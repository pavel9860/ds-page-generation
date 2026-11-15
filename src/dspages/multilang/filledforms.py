"""Filled forms made, not found"""
import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path

from hwsynth.pagegen import forms
from hwsynth.pagegen.hand import writing_height
from PIL import Image

from .config import LANGDS as C
from .config import Config
from .textpool import load

CITY_NO = list(range(1, 31))


def spots(path, page, lang, seed=0, form=None):
    """(form, writing height, chars each spot holds, rng) of a form page; a spot holding fewer than 2 chars is not filled."""
    cfg = replace(Config(), long_side=1754)
    rng = random.Random(f"{seed}{path}{page}")
    form = form or forms.render_form(path, page, cfg, lang=lang)
    height = writing_height(form.prints, form.img.shape[0], rng)
    pool = load(cfg)
    sample = " ".join(random.Random(1).sample(pool.get(lang, ["x"]), min(6, len(pool.get(lang, ["x"])))))
    return form, height, forms.capacity(form, height, sample, cfg, rng), rng


def fillable(path, page, lang):
    """Whether prep_one shows a spot of the form page to be filled (one holding 2 chars or more); the handwriting width (capacity, the slow part) only
    when a spot's cells do not tell it."""
    form = forms.render_form(path, page, replace(Config(), long_side=1754), lang=lang)
    if not form.spots:
        return False
    if any(len(s.get("cells", ())) >= 2 for s in form.spots):
        return True
    return any(c >= 2 for c in spots(path, page, lang, form=form)[2])


def prep_one(job):
    """(path, page, lang, work_dir, seed) -> (batch item, meta)."""
    import cv2
    path, page, lang, work, seed = job
    form, height, caps, rng = spots(path, page, lang, seed)
    x0, y0 = 0, 0
    img = form.img.copy()
    shown = []
    for i, (sp, c) in enumerate(zip(form.spots, caps)):
        bx = sp["box"]
        if c < 2:
            continue
        shown.append(i)
        for cx0, cy0, cx1, cy1 in (sp["cells"] if "cells" in sp else [bx]):
            cv2.rectangle(img, (int(cx0 - x0), int(cy0 - y0)), (int(cx1 - x0), int(cy1 - y0)), (255, 0, 0), 2)
        cv2.putText(img, str(i), (int(bx[0] - x0) + 3, int(bx[3] - y0) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    fid = hashlib.sha1(f"{path}{page}".encode()).hexdigest()[:10]
    (Path(work) / "img").mkdir(parents=True, exist_ok=True)
    jpg = Path(work) / "img" / f"{fid}.jpg"
    Image.fromarray(img).save(jpg, quality=95)
    item = dict(form_id=fid, lang=lang, image=str(jpg),
                boxes=[dict(i=i, chars=int(caps[i]), h_lines=max(1, round((form.spots[i]["box"][3] - form.spots[i]["box"][1]) / height)),
                         label=form.spots[i]["label"])
                       for i in shown],
                seed=dict(gender=rng.choice(["woman", "man"]), age=rng.randint(19, 78), city_no=rng.choice(CITY_NO),
                          date=f"{rng.randint(2019, 2025)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}", name_no=rng.randint(4, 60)))
    return item, dict(fid=fid, path=path, page=page, lang=lang, height=height, chars=[int(c) for c in caps], win=[x0, y0], shown=shown)


def prepare(backgrounds, out, per_file=5, seed=0, workers=6):
    """backgrounds: [(path, page, lang)] of blank form pages."""
    from concurrent.futures import ProcessPoolExecutor
    work = Path(out) / "forms_work"
    work.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(workers) as ex:
        res = [r for r in ex.map(prep_one, [(p, g, lang, str(work), seed) for p, g, lang in backgrounds], chunksize=2) if r]
    items = [it for it, _ in res]
    meta = {m["fid"]: m for _, m in res}
    for n in range(0, len(items), per_file):
        (work / f"in_{n // per_file:04d}.json").write_text(json.dumps(items[n:n + per_file], ensure_ascii=False))
    (work / "meta.json").write_text(json.dumps(meta))
    return len(items)


def fill_all(out, variants, workers=10):
    """Fills every form of forms_work/meta.json that has values in out_NNNN.json {form_id"""
    from concurrent.futures import ProcessPoolExecutor
    work = Path(out) / "forms_work"
    meta = json.loads((work / "meta.json").read_text())
    vals = {}
    for f in sorted(work.glob("out_*.json")):
        vals.update(json.loads(f.read_text()))
    jobs = [(fid, m, vals[fid], str(out), v) for fid, m in meta.items() if fid in vals for v in range(variants.get(m["lang"], 1))]
    with ProcessPoolExecutor(workers) as ex:
        return [r for r in ex.map(make_filled, jobs, chunksize=2) if r]


def fill_window(fid, m, answers, v, path=None):
    """The whole page of form `fid` (meta m) filled with `answers` {box number"""
    import cv2
    cv2.setNumThreads(1)
    from .detlayouts import lines
    page = int(m["page"])
    cfg = replace(Config(), long_side=1754)
    rng = random.Random(f"{m['path']}{page}{m['lang']}{v}")
    form = forms.render_form(path or m["path"], page, cfg, lang=m["lang"])
    height = writing_height(form.prints, form.img.shape[0], random.Random(f"row{m['path']}{page}"))
    chars_of = m["chars"] if isinstance(m["chars"], list) else json.loads(m["chars"])
    values = [None] * len(form.spots)
    for k, text in answers.items():
        i = int(k)
        if i < len(values) and isinstance(text, str) and text.strip():
            t = text.strip()
            cap = chars_of[i] if i < len(chars_of) else None
            while cap and len(t) > cap:
                t = t.rsplit(" ", 1)[0] if " " in t else t[:cap]
            values[i] = [t, " ".join(t.split()[:max(1, round(0.6 * len(t.split())))])]
    img, hand_boxes, n, words = forms.fill(form, values, height, cfg, rng)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    k = C.side / max(gray.shape)
    gray = cv2.resize(gray, (round(gray.shape[1] * k), round(gray.shape[0] * k)), interpolation=cv2.INTER_AREA)
    scale = lambda bs: [[round(v * k) for v in b[:4]] for b in bs]
    return gray, scale(hand_boxes), scale(lines([tuple(b) + ("",) for b in form.prints])), n, words


def make_filled(job):
    """(form id, meta, answers {box number: text}, out_dir, variant) -> row dict or None"""
    fid, m, answers, out, v = job
    from hwsynth.scripts import text_script
    path, page, lang = m["path"], m["page"], m["lang"]
    res = fill_window(fid, m, answers, v)
    canvas, hw, pr, n, words = res
    scr = text_script(" ".join(t for t in answers.values() if isinstance(t, str))[:400] or "a")
    boxes = [dict(k="hand", b=b, script=scr) for b in hw] + [dict(k="print", b=b) for b in pr]
    d = Path(out) / "synth" / "filled" / lang
    d.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(d / f"{fid}_{v}.jpg", quality=88)
    return dict(id=f"{fid}_{v}", file=f"synth/filled/{lang}/{fid}_{v}.jpg", lang=lang, type="filled", src=path, page=page,
                spots=len(m["shown"]), filled=n, words=words, scan=False, lang_status="synthetic", boxes=boxes, script=scr)
