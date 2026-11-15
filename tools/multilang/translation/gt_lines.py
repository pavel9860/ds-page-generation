"""Bank lines for languages with too few
python scripts/translation/gt_lines.py [--lang xx] [--rounds 3]"""
import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor

from dspages.multilang import linebank as LB
from dspages.multilang.config import LINEBANK as C
from dspages.multilang.config import TRANSLATE as T
from dspages.multilang.gtranslate import google
from dspages.multilang.paths import LANGDS, TRANSLATION

PARAS = LANGDS / "paras"


def translated(lang):
    """{raw, w, used}; files of the first run have no `raw`, only the lines and the paragraphs used."""
    f = LB.TRANSLATED / f"{lang}.json"
    d = json.loads(f.read_text()) if f.exists() else {}
    return {"raw": d.get("raw", []), "w": d.get("w", []), "used": d.get("used", [])}


def missing(lang):
    """Lines a language still lacks, counting its corpus bank and the lines translated so far."""
    f = LB.BANK / f"{lang}.json"
    return max(0, C.min_lines - (len(json.loads(f.read_text())["w"]) if f.exists() else 0) - len(translated(lang)["w"]))


def sources(langs):
    """{language: [paragraph]} shuffled, without paragraphs already translated"""
    used = {p for l in langs for p in translated(l)["used"] + [r["src"] for r in translated(l)["raw"]]}
    out = {}
    for sl in list(T.rich) + [l for l in langs if l not in T.rich and not missing(l)]:
        f = PARAS / f"{sl}.json"
        ps = [" ".join(p.split()) for p in json.loads(f.read_text()).get("p", [])] if f.exists() else []
        ps = [p for p in ps if p not in used]
        if ps:
            random.Random(sl).shuffle(ps)
            out[sl] = ps
    return out


def plan(langs, src):
    """{target: [(source language, paragraph)]} whose clauses should cover what the target lacks"""
    k, out = dict.fromkeys(src, 0), {}
    for lang in sorted(l for l in langs if missing(l)):
        want, got, out[lang] = missing(lang) * T.margin, 0, []
        while got < want and any(k[sl] < len(src[sl]) for sl in src):
            for sl in src:
                if k[sl] < len(src[sl]) and got < want:
                    out[lang].append((sl, src[sl][k[sl]]))
                    got += len(LB.phrases(src[sl][k[sl]], sl))
                    k[sl] += 1
    return {l: ps for l, ps in out.items() if ps}


def translate_all(translate, jobs, workers=T.workers):
    """{target: [{from, src, tr}]} for jobs [(target, source language, paragraph)]"""
    with ThreadPoolExecutor(workers) as ex:
        texts = list(ex.map(lambda j: translate(*j), jobs))
    out = {}
    for (lang, sl, p), t in zip(jobs, texts):
        out.setdefault(lang, []).append({"from": sl, "src": p, "tr": t})
    return out


def lines_of(raw, lang):
    """The clauses of the translations, each once."""
    seen, out = set(), []
    for r in raw:
        for line in LB.phrases(r["tr"] or "", lang):
            k = LB.key(line)
            if k and k not in seen:
                seen.add(k)
                out.append(" ".join(line.split()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang")
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args()
    langs = sorted({r["lang"] for r in json.loads((LANGDS / "ds3k_plan.json").read_text())})
    codes = {x[0]: x[1] for x in json.loads((TRANSLATION / "source/jobs.json").read_text())}
    t = time.time()
    for rnd in range(1, a.rounds + 1):
        plans = {l: ps for l, ps in plan(langs, sources(langs)).items() if not a.lang or l == a.lang}
        if not plans:
            break
        jobs = [(l, sl, p) for l, ps in plans.items() for sl, p in ps]
        print(f"round {rnd}: {len(plans)} languages, {len(jobs)} paragraphs", flush=True)
        for lang, new in translate_all(lambda l, sl, p: google(p, codes.get(l, l)), jobs).items():
            raw = translated(lang)["raw"] + new
            LB.TRANSLATED.mkdir(parents=True, exist_ok=True)
            old = translated(lang)
            (LB.TRANSLATED / f"{lang}.json").write_text(json.dumps(dict(raw=raw, w=list(dict.fromkeys(old["w"] + lines_of(raw, lang))), used=old["used"]), ensure_ascii=False))
            print(f"[{time.time() - t:4.0f}s] {lang}: {len(new)} paragraphs, {sum(r['tr'] is None for r in new)} failed, missing {missing(lang)}", flush=True)
    print("still missing:", {l: missing(l) for l in langs if missing(l)})


if __name__ == "__main__":
    main()
