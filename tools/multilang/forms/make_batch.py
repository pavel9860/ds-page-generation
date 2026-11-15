"""A batch of forms for the agents with the printed label of every box (from the page text) and the language name"""

import json
import sys
from dataclasses import replace

from dspages.multilang.buckets import NAMES
from dspages.multilang.config import Config
from dspages.multilang.paths import FILLING
from dspages.pagegen import forms


def main():
    out, name, fids = FILLING / sys.argv[1], sys.argv[2], sys.argv[3:]
    out.mkdir(exist_ok=True)
    meta = json.loads((FILLING / "meta.json").read_text())
    have = {}
    for f in sorted(FILLING.glob("in_*.json")):
        have.update({i["form_id"]: i for i in json.loads(f.read_text())})
    cfg = replace(Config(), long_side=1754)
    items = []
    for fid in fids:
        m, it = meta[fid], dict(have[fid])
        spots = forms.render_form(m["path"], int(m["page"]), cfg).spots
        it["language"] = NAMES.get(m["lang"], m["lang"])
        it["boxes"] = [dict(b, label=spots[b["i"]]["label"] if b["i"] < len(spots) else "") for b in it["boxes"]]
        items.append(it)
    (out / f"in_{name}.json").write_text(json.dumps(items, ensure_ascii=False))
    print(len(items), "forms ->", out / f"in_{name}.json")


if __name__ == "__main__":
    main()
