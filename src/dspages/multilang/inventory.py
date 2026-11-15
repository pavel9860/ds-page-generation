"""What the language dataset can be built from, per language"""

import json
import os
from collections import defaultdict
from pathlib import Path

from dspages.pagegen.handfonts import can_write
from dspages.pagegen.source import scripts_of

from .buckets import BUCKETS
from .config import LANGDS as C
from .langds import plan
from .realpages import parse_lang
from .synthpages import load_pools


def source_files(layouts):
    """{(lang, type): [digital pdf count, image count]} of the Layouts folders."""
    n = defaultdict(lambda: [0, 0])
    for typ, folders in C.folders.items():
        for d in folders:
            for f in os.listdir(Path(layouts) / d):
                lang = parse_lang(f)
                if lang:
                    n[(lang, typ)][0 if f.lower().endswith(".pdf") else 1] += 1
    return n


def languages():
    return sorted({m for ms in BUCKETS.values() for m in ms.split()} - {"dv"} - set(C.alias))


def inventory(layouts, out, work):
    """Rows {lang, bucket, files: {type"""
    files = source_files(layouts)
    bucket = {m: b for b, ms in BUCKETS.items() for m in ms.split()}
    quota, _ = plan([])
    rows = []
    for lang in languages():
        real = Path(out) / "paras" / f"{lang}.json"
        pool = load_pools(lang, out, work, None)
        rows.append(
            dict(
                lang=lang,
                bucket=bucket[lang],
                files={t: files.get((lang, t), [0, 0]) for t in C.folders},
                real_paras=len(json.loads(real.read_text()).get("p", [])) if real.exists() else 0,
                pool_p=len(pool["p"]),
                docs=len(list(Path(work).glob(f"out_{lang}_?.json"))),
                fields=min(len(pool["labels"]), len(pool["heads"]), len(pool["cells"])),
                writable=can_write(scripts_of(lang)),
                quota={t: quota.get((lang, t), 0) for t in C.types},
            )
        )
    return rows


def gaps(rows):
    """[(lang, what)]: a printed type with a quota and no source file of it (handwritten and filled pages are made here), a text pool under C.min_pool, fewer than 7 translated documents, no form words."""
    out = []
    for r in rows:
        for t in ("text", "table", "form", "flyer", "magazine"):
            if r["quota"][t] and not sum(r["files"][t]):
                out.append((r["lang"], f"no {t} source file"))
        if r["pool_p"] < C.min_pool:
            out.append((r["lang"], f"{r['pool_p']} paragraphs"))
        if r["docs"] < 7:
            out.append((r["lang"], f"{r['docs']} of 7 translated documents"))
        if not r["fields"]:
            out.append((r["lang"], "no form / table words"))
    return out


def report(rows):
    print(f"{'lang':5}{'bucket':14}" + "".join(f"{t[:5]:>8}" for t in C.folders) + "  real_p pool_p docs fields hand")
    for r in rows:
        cells = "".join(f"{r['files'][t][0]:>4}/{r['files'][t][1]:<3}" for t in C.folders)
        print(
            f"{r['lang']:5}{r['bucket'][:13]:14}{cells}  {r['real_paras']:>6} {r['pool_p']:>6} {r['docs']:>4} {r['fields']:>6} {'yes' if r['writable'] else 'no':>4}"
        )
    print("share of the plan:", {t: C.share[t] for t in C.types})
    for lang, what in gaps(rows):
        print("GAP", lang, what)


def needs(rows):
    """{lang: {type: (quota, real wanted, files of the type, real still missing, pages to render)}} from the typed folders only (no page-type check, no search of other folders); filled and handwritten pages are all rendered."""
    out = {}
    for r in rows:
        d = {}
        for t in C.types:
            q = r["quota"][t]
            if not q:
                continue
            real = int(q * C.real_share) if t in C.folders and t not in C.hand else 0
            have = sum(r["files"].get(t, [0, 0])) if t not in C.hand else 0
            d[t] = (q, real, have, max(0, real - have), q - min(real, have))
        out[r["lang"]] = d
    return out
