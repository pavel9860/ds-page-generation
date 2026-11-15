"""The language dataset: pages of every type labelled by language class (buckets.py)"""
import hashlib
import json
import os
import random
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from hwsynth.pagegen.handfonts import can_write

from .buckets import BUCKETS, bucket_of
from .config import LANGDS as C


def writable(lang):
    """True when synthetic handwriting is possible in the language's script(s) (handfonts.py)."""
    from hwsynth.pagegen.source import scripts_of
    return can_write(scripts_of(lang))


def plan(real_rows, n_per_class=300):
    """{(lang, type): quota} and the real rows chosen"""
    quota = {}
    for members in BUCKETS.values():
        langs = [m for m in members.split() if m != "dv" and m not in C.alias]
        if not langs:
            continue
        per = max(len(C.types) * C.min_per_type, n_per_class // len(langs))
        for lang in langs:
            share = dict(C.share)
            if not writable(lang):
                extra = share["handwritten"] + share["filled"]
                share = {t: (0.0 if t in C.hand else v / (1 - extra)) for t, v in share.items()}
            q = {t: (0 if t in C.hand and share[t] == 0 else max(C.min_per_type, round(share[t] * per))) for t in C.types}
            quota.update({(lang, t): n for t, n in q.items()})
    by = defaultdict(list)
    for r in real_rows:
        by[(r["lang"], r["type"])].append(r)
    chosen = []
    for (lang, t), n in quota.items():
        if t == "filled" or n == 0:
            continue
        rows = by.get((lang, t), [])
        digital = [r for r in rows if r.get("lang_status") in ("ok", "other") and not r.get("scan")]
        rows = digital or ([r for r in rows if r.get("scan")] if t in C.scans_ok else [])
        random.Random(f"{lang}{t}").shuffle(rows)
        chosen += rows[:max(0, int(n * C.real_share))]
    return quota, chosen


def class_of(lang, script=None):
    return bucket_of(lang, script)


def assemble(out, rows, ds):
    """Links the images of `rows` ({file, lang, type, src|id}) into ds/{train,val}/<class>/ (val: a fifth of the sources, by hash) and writes ds/manifest.jsonl."""
    out, ds = Path(out), Path(ds)
    n = defaultdict(int)
    with open(ds / "manifest.jsonl", "w", encoding="utf-8") as mf:
        for r in rows:
            key = r.get("src") or r["id"]
            split = r.get("split") or ("val" if int(hashlib.sha1(f"{key}".encode()).hexdigest(), 16) % 5 == 0 else "train")
            cls = class_of(r["lang"], r.get("script"))
            d = ds / split / cls
            d.mkdir(parents=True, exist_ok=True)
            tag = "synth" if r.get("lang_status") == "synthetic" else "real"
            dst = d / f"{r['lang']}_{r['type']}_{tag}_{r['id']}.jpg"
            if not dst.exists():
                shutil.copy(out / r["file"], dst)
            mf.write(json.dumps(dict(r, split=split, cls=cls), ensure_ascii=False) + "\n")
            n[split] += 1
    return dict(n)


def synth_jobs(quota, chosen_real, out, pools):
    """[(kind, lang, seed, out, pool, split)] of the synthetic pages still needed per (language, type)"""
    have = defaultdict(int)
    for r in chosen_real:
        have[(r["lang"], r["type"])] += 1
    jobs = []
    for (lang, t), n in sorted(quota.items()):
        if t == "filled" or n == 0:
            continue
        need = max(0, n - have[(lang, t)])
        n_val = round(0.2 * need) if need >= 5 else (1 if need >= 3 else 0)
        for split, count in (("train", need - n_val), ("val", n_val)):
            pool = pools.get((lang, split))
            if pool and len(pool["p"]) >= 2:
                jobs += [(t, lang, i, str(out), pool, split) for i in range(count)]
    return jobs


def run_synth(jobs, workers=11, chunk=3000):
    from .synthpages import make_synth
    rows = []
    os.environ["OMP_NUM_THREADS"] = "1"
    for i in range(0, len(jobs), chunk):
        with ProcessPoolExecutor(workers) as ex:
            rows += [r for r in ex.map(make_synth, jobs[i:i + chunk], chunksize=2) if r]
    return rows


def balance(rows, per_class):
    """Rows cut to at most per_class per class, equal per language inside a class (the languages in turn)."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[class_of(r["lang"], r.get("script"))][r["lang"]].append(r)
    out = []
    for cls, langs in by.items():
        queues = [list(v) for v in langs.values()]
        got = 0
        while got < per_class and any(queues):
            for q in queues:
                if q and got < per_class:
                    out.append(q.pop())
                    got += 1
    return out


def counts(rows):
    c = defaultdict(lambda: defaultdict(int))
    for r in rows:
        c[class_of(r["lang"], r.get("script"))][r["type"]] += 1
    return c


def report(rows):
    """Items per class and type, and the languages (or class-type cells) below C.min_per_type."""
    per_lang = defaultdict(lambda: defaultdict(int))
    for r in rows:
        per_lang[r["lang"]][r["type"]] += 1
    short = sorted((lang, t, per_lang[lang][t]) for members in BUCKETS.values() for lang in members.split() if lang != "dv" and lang not in C.alias
                   for t in C.types if per_lang[lang][t] < C.min_per_type and (t not in C.hand or writable(lang)))
    return dict(classes=len(counts(rows)), items=len(rows), short=short[:40], n_short=len(short))

