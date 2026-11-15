"""Forms of generation/handwriting/forms that can be used (>= 3 answers; the language is the form's label) per language against the slots of ds3k_plan.json; writes handwriting/forms/usable.json {fid: status} and short.json {lang: forms missing}. python scripts/forms/forms_status.py
python scripts/forms/forms_status.py"""

import json
from collections import Counter

from dspages.multilang.paths import FILLING as W
from dspages.multilang.paths import LANGDS as D


def main():
    meta = json.loads((W / "meta.json").read_text())
    ans = {}
    for f in sorted(W.glob("out_*.json")):
        ans.update(json.loads(f.read_text()))
    status = {}
    for fid, m in meta.items():
        a = [v for v in ans.get(fid, {}).values() if isinstance(v, str) and v.strip()]
        if fid not in ans:
            status[fid] = "unanswered"
        elif len(a) < 3:
            status[fid] = "few answers"
        else:
            status[fid] = "ok"
    (W / "usable.json").write_text(json.dumps(status))
    plan = json.loads((D / "ds3k_plan.json").read_text())
    need = Counter(p["lang"] for p in plan if p["kind"] == "form")
    have = Counter(meta[f]["lang"] for f, s in status.items() if s == "ok")
    short = {l: need[l] - have[l] for l in need if need[l] > have[l]}
    (W / "short.json").write_text(json.dumps(short))
    print(Counter(s.split(":")[0] for s in status.values()), "\nshort", sum(short.values()), short)


if __name__ == "__main__":
    main()
