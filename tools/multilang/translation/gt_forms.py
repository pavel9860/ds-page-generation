"""Source forms translated by Google (HTTP)
python scripts/translation/gt_forms.py SUFFIX"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor

from dspages.multilang import gtranslate
from dspages.multilang.config import TRANSLATE as T
from dspages.multilang.paths import TRANSLATION

D, OUT = TRANSLATION / "source", TRANSLATION / "google"


def translate_job(job, src):
    lang, tl, idxs = job
    return dict(lang=lang, tl=tl, items=[dict(key=src[i][0], tr=gtranslate.lines([s.strip() or "." for s in src[i][2]], tl, src[i][1])) for i in idxs])


def main():
    suffix = sys.argv[1]
    src = json.loads((D / "src_typed.json").read_text())
    jobs = [j for j in json.loads((D / "jobs_extra.json").read_text()) if not (OUT / f"gt_{j[0]}{suffix}.json").exists()]
    with ThreadPoolExecutor(T.workers) as ex:
        for job, res in zip(jobs, ex.map(lambda j: translate_job(j, src), jobs)):
            bad = sum(not it["tr"] or None in it["tr"] for it in res["items"])
            (OUT / f"gt_{job[0]}{suffix}.json").write_text(json.dumps(res, ensure_ascii=False))
            print(job[0], len(res["items"]), "forms,", bad, "with a failed line", flush=True)


if __name__ == "__main__":
    main()
