"""Translated layouts for the languages short of layouts
python scripts/translation/gt_layouts.py prep LANG=N [LANG=N ...]   |   python scripts/translation/gt_layouts.py render"""
import json
import random
import sys
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).parent.parent / "review"))
import translate_layouts as TL
from dspages.hand.plan import EXPECT
from dspages.multilang import detlayouts as ds3k
from dspages.multilang import translate as T
from dspages.multilang.buckets import NAMES


def sources(n, exclude):
    """n clean text-like layouts near the page ratio that are not planned or used, shuffled."""
    flag = json.loads((TL.L / "layout_hw_flagged.json").read_text())
    rows = [json.loads(l) for l in open(TL.L / "det_layouts/layouts.jsonl")]
    good = [r for r in rows if r["type"] in ("text", "magazine", "flyer") and flag.get(r["id"], "clean") == "clean" and r["id"] not in exclude
            and r.get("text_ok", True) and len(r["words"]) > 80 and ds3k.layout_is_near_a4(r["file"])
            and (r["script"] == EXPECT.get(r["lang"], r["script"]) or r["lang"] in TL.TWO)]
    random.Random(7).shuffle(good)
    return good[:n]


def prep(need):
    assign = json.loads((TL.TR / "assign.json").read_text()) if (TL.TR / "assign.json").exists() else {}
    exclude = {p.get("id") for p in json.loads((TL.L / "ds3k_plan.json").read_text())} | {a["id"] for v in assign.values() for a in v}
    srcs = iter(sources(sum(need.values()), exclude))
    for lang, n in sorted(need.items()):
        d = TL.TR / lang
        d.mkdir(parents=True, exist_ok=True)
        meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
        part, items = len(list(d.glob("in_*.json"))), []
        for r in (next(srcs) for _ in range(n)):
            a = dict(id=r["id"], doc=r["doc"], path=r["src"], page=int(r["page"]), src_lang=r["lang"], type=r["type"])
            assign.setdefault(lang, []).append(a)
            with pymupdf.open(a["path"]) as doc:
                for b in T.paragraphs(doc[a["page"]]):
                    i = f"{len(meta) + len(items):05d}"
                    items.append(dict(id=i, language=NAMES.get(lang, lang), text=b["text"]))
                    meta[i] = dict(doc=a["doc"], bbox=b["bbox"], size=b["size"], bold=b["bold"])
        (d / f"in_{part:04d}.json").write_text(json.dumps(items, ensure_ascii=False))
        (d / "meta.json").write_text(json.dumps(meta))
        print(lang, NAMES.get(lang, lang), n, "layouts,", len(items), "paragraphs ->", d / f"in_{part:04d}.json")
    (TL.TR / "assign.json").write_text(json.dumps(assign, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "prep":
        prep({k: int(v) for k, v in (a.split("=") for a in sys.argv[2:])})
    else:
        TL.render()
