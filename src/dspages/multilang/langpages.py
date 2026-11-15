"""Page-level language dataset: candidate pages of the layout PDFs per language and type (text, tables, forms)."""

import json
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pymupdf
from hwsynth.scripts import text_script

from .bucketing import KIND
from .buckets import bucket_of
from .config import BATCH, BINS, BODY_PT, BUCKET_NEED, FLOOR, NEED, OVERSAMPLE, Config  # noqa: F401

TEXT_CATEGORIES = {"books_pdf", "scientific_paper", "other_docs", "exams", "bureaucracy"}
CJK = {"zh", "ja", "ko", "th", "lo", "km", "my", "bo", "dz"}


def page_type(category):
    return "text" if category in TEXT_CATEGORIES else category if category in ("tables", "forms") else None


def lang_of(path):
    """Language postfix of a layout file name: `<hash>_<kind>_<lang>.pdf`."""
    return Path(path).stem.rsplit("_", 1)[-1]


def detect_type(page):
    """'tables' (a grid of ruling lines), 'forms' (fill-in widgets or underscore/dotted blanks) or 'text'"""
    n = len(page.get_text("words"))
    if n < 30:
        return None
    text = page.get_text()
    if page.first_widget or len(re.findall(r"_{4,}|\.{6,}", text)) >= 6:
        return "forms"
    lines = sum(1 for d in page.get_drawings() for it in d["items"] if it[0] in ("l", "re"))
    return "tables" if lines >= 40 else "text"


def snippet(page, cjk):
    """About 1/10 of the page's text (30-100 words, 100-250 characters when written without spaces) from the middle of the page, where running text is rather than headers; None when the page has less than 30 words."""
    text = page.get_text("text", sort=True)
    if cjk:
        t = re.sub(r"\s+", "", text)
        k = min(250, max(100, len(t) // 10))
        return t[len(t) // 2 - k // 2 :][:k] if len(t) >= 100 else None
    w = text.split()
    k = min(100, max(30, len(w) // 10))
    return " ".join(w[len(w) // 2 - k // 2 :][:k]) if len(w) >= 30 else None


def pages_of(job):
    """Up to 3 candidate pages of one file"""
    rec, cjk = job
    out = []
    try:
        with pymupdf.open(rec["path"]) as doc:
            n = len(doc)
            for i in random.Random(rec["path"]).sample(range(n), min(n, 5)):
                t = detect_type(doc[i])
                s = t and snippet(doc[i], cjk)
                if s and len(out) < 3:
                    out.append(
                        dict(
                            lang=rec["language"],
                            type=t,
                            was_type=page_type(rec["category"]),
                            path=rec["path"],
                            page=i,
                            body_pt=rec["body_pt"],
                            snippet=s,
                        )
                    )
    except Exception:
        pass
    return out


def inventory(cfg: Config, manifest):
    """Records {path, language, category, body_pt} of the digital layout PDFs"""
    recs = {}
    for line in open(manifest, encoding="utf-8"):
        r = json.loads(line)
        if (
            r.get("digital")
            and r.get("body_pt")
            and Path(r["path"]).exists()
            and BODY_PT[0] <= r["body_pt"] <= BODY_PT[1]
        ):
            recs[r["path"]] = r
    known = {Path(p).name for p in recs}
    for d in cfg.bg_folders:
        for f in (cfg.layouts / d).glob("*.pdf"):
            if f.name not in known and page_type(d):
                recs[str(f)] = dict(path=str(f), language=lang_of(f), category=d, body_pt=11.0)
    return list(recs.values())


def candidates(cfg: Config, keys, manifest, out):
    """Writes candidate pages of the languages `keys` to `out` (jsonl)"""
    by = defaultdict(list)
    for r in inventory(cfg, manifest):
        t = page_type(r["category"])
        if t and r["language"] in keys:
            by[(r["language"], t)].append(r)
    jobs = []
    for (lang, t), recs in by.items():
        random.Random(f"{lang}{t}").shuffle(recs)
        jobs += [(r, lang in CJK) for r in recs[: NEED[t] * OVERSAMPLE]]
    with ProcessPoolExecutor(cfg.workers) as ex, open(out, "w") as f:
        for pages in ex.map(pages_of, jobs, chunksize=16):
            for p in pages:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
    return len(jobs)


ALIAS = {"kz": "kk", "tj": "tg", "fil": "tl", "nb": "no", "sh": "sr"}


def select(path, per_type=NEED, spare=1.5):
    """Candidates per language and type"""
    by = defaultdict(lambda: defaultdict(list))
    for line in open(path):
        p = json.loads(line)
        by[(p["lang"], p["type"])][next(i for i, b in enumerate(BINS) if p["body_pt"] < b)].append(p)
    out = []
    for (lang, t), bins in by.items():
        seen, picked = set(), []
        lists = [random.Random(f"{lang}{t}{i}").sample(v, len(v)) for i, v in sorted(bins.items())]
        while any(lists) and len(picked) < round(per_type[t] * spare):
            for v in lists:
                while v:
                    p = v.pop()
                    if p["path"] not in seen:
                        seen.add(p["path"])
                        picked.append(p)
                        break
        out += picked
    return out


PROMPT = """Each item is a text snippet of a document. Name the language it is written in as an ISO 639-1 code (639-3 when
there is no 639-1 code). Decide only from the words, spelling, letters and grammar of the snippet. Ignore the topic and any
country, city, person or organisation names in it: a Russian text about Belarus is Russian, a Spanish text about Peru is
Spanish. Close languages differ in letters and words (Belarusian: ў, і, ’, "што", "гэта"; Russian: ы, э, ё, щ, "что", "это";
Ukrainian: і, ї, є, ґ; Serbian vs Croatian: script and ije/e; Norwegian vs Danish; Indonesian vs Malay), check them. If
the snippet is not running text or is garbled, answer "und". Write {id: code} for every item."""


def cls(p):
    """Class (buckets.py) of a page: its language, and its script when the language is written in several."""
    return bucket_of(label(p), text_script(p["snippet"]) if p.get("snippet") else None)


def label(p):
    return ALIAS.get(p["lang"], p["lang"])


def _answers(out_dir, k):
    """{page index: code} of pass k."""
    got = {}
    for f in sorted(Path(out_dir).glob(f"out_P{k}*.json")):
        got.update({int(i): c for i, c in json.loads(f.read_text()).items()})
    return got


def verified(pages, out_dir, passes=2):
    """(kept, dropped)."""
    answers = [_answers(out_dir, k) for k in range(1, passes + 1)]
    kept, dropped = [], []
    for i, p in enumerate(pages):
        codes = [a[i] for a in answers if i in a]
        named = Counter(c for c in codes if c != "und")
        if label(p) in codes:
            kept.append(p)
        elif named and named.most_common(1)[0][1] >= 2:
            kept.append({**p, "lang": named.most_common(1)[0][0], "was": p["lang"]})
        else:
            dropped.append((p, codes))
    return kept, dropped


def assign(pages, need=None, floor=None):
    """Fills the classes (see buckets.py) with the verified pages"""
    need = need or BUCKET_NEED
    floor = FLOOR if floor is None else floor
    by = defaultdict(lambda: defaultdict(list))
    for p in pages:
        by[(cls(p), p["type"])][label(p)].append(p)
    kept, surplus, files = [], [], set()

    def take(q):
        while q:
            p = q.pop(0)
            if p["path"] not in files:
                files.add(p["path"])
                kept.append(p)
                return True
        return False

    for (b, t), langs in by.items():
        queues = [q for _, q in sorted(langs.items())]
        got = 0
        for q in queues:
            for _ in range(floor.get(t, 0)):
                got += take(q)
        while any(queues) and got < need.get(t, 0):
            for q in queues:
                if got < need[t]:
                    got += take(q)
        surplus += [p for q in queues for p in q]
    return kept, surplus


def rendered(pages, work, langs):
    """Translated, re-set pages (bucketing.py: png/<lang>_<n>.png, text, tables or forms by the source page n) for the languages still short of their floor of pages per type after `assign`: [{lang, type, path, render: True}]."""
    have = Counter((label(p), p["type"]) for p in pages)
    out = []
    for lang in langs:
        for t, k in FLOOR.items():
            files = [
                f for f in sorted(Path(work, "png").glob(f"{lang}_*.png")) if KIND[int(f.stem.rsplit("_", 1)[1])] == t
            ]
            out += [
                dict(lang=lang, type=t, path=str(f), page=0, render=True) for f in files[: max(0, k - have[(lang, t)])]
            ]
    return out
