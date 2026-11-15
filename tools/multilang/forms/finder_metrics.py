"""Precision / recall of the field finder against the fillable (AcroForm) fields of PDF forms"""
import json
import sys
from collections import Counter
from dataclasses import replace

import pymupdf
from dspages.multilang.config import Config
from dspages.multilang.paths import FILLING
from hwsynth.pagegen import forms


def ov(a, b):
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    return max(w, 0) * max(h, 0) / max(min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])), 1)


def main():
    lo, hi = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 30)
    meta = json.loads((FILLING / "meta.json").read_text())
    cfg = replace(Config(), long_side=1754)
    ok, bad, hit, miss, n = Counter(), Counter(), 0, 0, 0
    for fid, v in meta.items():
        try:
            pg = pymupdf.open(v["path"])[int(v["page"])]
            ws = [w.rect for w in (pg.widgets() or []) if w.field_type_string in ("Text", "ComboBox") and not w.field_value]
        except Exception:
            continue
        if len(ws) < 3:
            continue
        n += 1
        if not lo <= n - 1 < hi:
            continue
        k = cfg.long_side / max(pg.rect.width, pg.rect.height)
        W = [(r.x0 * k, r.y0 * k, r.x1 * k, r.y1 * k) for r in ws]
        sp = forms.render_form(v["path"], int(v["page"]), cfg).spots
        for w in W:
            hit, miss = (hit + 1, miss) if any(ov(w, s["box"]) > 0.5 for s in sp) else (hit, miss + 1)
        for s in sp:
            (ok if any(ov(w, s["box"]) > 0.5 for w in W) else bad)[s.get("src", "?")] += 1
    tot = sum(ok.values()) + sum(bad.values())
    print(f"forms {lo}-{hi}: recall {100 * hit / max(hit + miss, 1):.0f}% ({hit} of {hit + miss}) | spots off a real field {100 * sum(bad.values()) / max(tot, 1):.0f}% ({sum(bad.values())} of {tot})")
    print("   by source: " + ", ".join(f"{k}: {bad[k]}/{ok[k] + bad[k]} off" for k in sorted(set(ok) | set(bad))))


if __name__ == "__main__":
    main()
