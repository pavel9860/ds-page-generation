"""Google Translate in a headless browser (Brave through selenium), no tokens
python scripts/translation/gt_run.py [--workers 3] [--lang xx] [--extra]   (--extra: source/jobs_extra.json, files gt_<lang>x.json)"""

import json
import sys
import time
from multiprocessing import Pool
from urllib.parse import quote

from selenium import webdriver

from dspages.multilang.paths import TRANSLATION

D = TRANSLATION / "source"
OUT = TRANSLATION / "google"
SUF = sys.argv[sys.argv.index("--suffix") + 1] if "--suffix" in sys.argv else "x" if "--extra" in sys.argv else ""
BIN = "/snap/brave/current/opt/brave.com/brave/brave"
SET = """const ta=document.querySelector('textarea[aria-label="Source text"]');
const setv=Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype,'value').set; setv.call(ta,arguments[0]);
ta.dispatchEvent(new Event('input',{bubbles:true}));"""
GET = "const e=[...document.querySelectorAll('span[jsname=\"jqKxS\"]')].pop(); return e?e.innerText:'';"


def translate(d, lines):
    """list of translated lines (None when the translation has another number of lines than the source)."""
    text = "\n".join(lines)
    d.execute_script(SET, "")
    t0 = time.time()
    while d.execute_script(GET) and time.time() - t0 < 5:  # the last text's output gone: nothing stale is read
        time.sleep(0.1)
    d.execute_script(SET, text)
    last, stable, t0 = "", 0, time.time()
    while time.time() - t0 < 40:
        time.sleep(0.25)
        t = d.execute_script(GET)
        if t:
            stable = stable + 1 if t == last else 0
            last = t
            parts = t.split("\n")
            if (
                stable >= 12 or stable >= 4 and len(parts) == len(lines)
            ):  # 1 s unchanged with every line there, else 3 s (Google refines a long text in steps)
                return parts if len(parts) == len(lines) else None
    return None


def one(d, lines):
    r = translate(d, lines)
    if r is not None or len(lines) == 1:
        return r
    out = []
    for l in lines:
        x = translate(d, [l])
        out.append(x[0] if x else None)
    return out


def run_lang(args):
    lang, tl, idxs = args
    f = OUT / f"gt_{lang}{SUF}.json"
    src = json.loads((D / "src_typed.json").read_text())
    if f.exists():
        return lang, "skip"
    o = webdriver.ChromeOptions()
    o.binary_location = BIN
    for a in ("--headless=new", "--no-sandbox", f"--user-data-dir=/tmp/gtprofile_{lang}"):
        o.add_argument(a)
    d = webdriver.Chrome(options=o)
    try:
        todo = [(src[i][0], src[i][1], [s.strip() or "." for s in src[i][2]]) for i in idxs]
        res = {}
        for sl in sorted({t[1] for t in todo}):
            d.get(f"https://translate.google.com/?sl={sl}&tl={quote(tl)}&op=translate")
            time.sleep(3)
            mine = [(k, ls) for k, l2, ls in todo if l2 == sl]
            j = 0
            while j < len(mine):
                grp, ch = [mine[j]], len("\n".join(mine[j][1]))
                j += 1
                while j < len(mine) and ch + len("\n".join(mine[j][1])) < 4200:
                    grp.append(mine[j])
                    ch += len("\n".join(mine[j][1]))
                    j += 1
                allv = [x for _, ls in grp for x in ls]
                tr = translate(d, allv)
                if tr is not None:
                    p = 0
                    for key, ls in grp:
                        res[key] = tr[p : p + len(ls)]
                        p += len(ls)
                else:
                    for key, ls in grp:
                        res[key] = one(d, ls)
        items = [dict(key=k, tr=res[k]) for k, _, _ in todo]
        bad = sum(1 for it in items if not it["tr"] or None in it["tr"])
        OUT.mkdir(parents=True, exist_ok=True)
        if bad == 0:
            f.write_text(json.dumps(dict(lang=lang, tl=tl, items=items), ensure_ascii=False))
        else:
            (OUT / f"bad_{lang}{SUF}.json").write_text(
                json.dumps(dict(lang=lang, tl=tl, items=items), ensure_ascii=False)
            )
        return lang, "ok" if bad == 0 else f"{bad} forms with failed lines"
    except Exception as e:
        return lang, f"err {e!r}"[:150]
    finally:
        d.quit()


def main():
    w = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 3
    only = sys.argv[sys.argv.index("--lang") + 1] if "--lang" in sys.argv else None
    jobs = [
        j for j in json.loads((D / ("jobs_extra.json" if SUF else "jobs.json")).read_text()) if not only or j[0] == only
    ]
    with Pool(w) as p:
        for lang, st in p.imap_unordered(run_lang, jobs):
            print(lang, st, flush=True)


if __name__ == "__main__":
    main()
