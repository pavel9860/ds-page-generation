"""More source forms for the languages short of forms
python scripts/translation/gt_sources.py [N_MORE_PER_SHORT_FORM]"""
import json
import sys

import pymupdf
from dspages.multilang import translate as T
from dspages.multilang.config import TRANSLATE
from dspages.multilang.paths import FILLING, TRANSLATION

RICH = TRANSLATE.rich
S = TRANSLATION / "source"


def main():
    spare = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    meta = json.loads((FILLING / "meta.json").read_text())
    st = json.loads((FILLING / "usable.json").read_text())
    short = json.loads((FILLING / "short.json").read_text())
    src = json.loads((S / "src_typed.json").read_text())
    have = {s[0] for s in src}
    jobs = {j[0]: j for j in json.loads((S / "jobs.json").read_text())}
    for fid, m in meta.items():
        if len(src) >= 80:
            break
        if fid in have or st.get(fid) != "ok" or m["lang"] not in RICH or "/generation/" in m["path"] or "/langds/tr" in m["path"]:
            continue
        with pymupdf.open(m["path"]) as doc:
            lines = [l["text"] for l in T.lines(doc[int(m["page"])])]
        if 12 <= len(lines) <= 45 and sum(map(len, lines)) <= 4500 and len(m["shown"]) >= 6:
            src.append([fid, m["lang"], lines])
    (S / "src_typed.json").write_text(json.dumps(src, ensure_ascii=False))
    extra = []
    for lang, n in short.items():
        used = set(jobs[lang][2]) if lang in jobs else set()
        free = [i for i in range(len(src)) if i not in used and src[i][1] != lang]
        extra.append([lang, jobs[lang][1] if lang in jobs else lang, free[len(free) // 3:][:3 * n + spare]])
    (S / "jobs_extra.json").write_text(json.dumps(extra))
    print(len(src), "sources;", sum(len(e[2]) for e in extra), "translations for", len(extra), "languages")


if __name__ == "__main__":
    main()
