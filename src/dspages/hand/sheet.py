"""HTML sheet (every page next to its exact text, tables as tables) and a contact sheet of a set of generated pages.

python -m dspages.hand sheet DIR      # DIR/<i>.jpg, DIR/data/plan_samples.jsonl, DIR/data/meta_samples.jsonl
"""

import argparse
import html
import json

import cv2
import numpy as np

CSS = (
    "body{font:14px sans-serif;margin:16px;max-width:1300px}section{display:flex;gap:16px;margin-bottom:28px;border-bottom:1px solid #ccc;"
    "padding-bottom:16px}img{width:520px;height:auto;border:1px solid #aaa}.t{max-height:740px;overflow:auto}p{margin:4px 0}"
    "b{color:#a00;font-size:11px;margin-right:6px}td{font-size:12px}"
)


def item_html(it):
    if it["k"] == "table":
        return (
            "<table border=1 cellpadding=3 style='border-collapse:collapse'>"
            + "".join("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in r) + "</tr>" for r in it["rows"])
            + "</table>"
        )
    return f"<p><b>{it['k']}</b> {html.escape(it['t'])}</p>"


def read_jsonl(path):
    return {r["i"]: r for r in map(json.loads, open(path))}


def build(d):
    plan, meta = read_jsonl(f"{d}/data/plan_samples.jsonl"), read_jsonl(f"{d}/data/meta_samples.jsonl")
    order = sorted(plan, key=lambda i: (plan[i]["fmt"], plan[i]["dense"], plan[i]["lang"]))
    rows = []
    for i in order:
        r, m = plan[i], meta[i]
        mode = "table" if r["table"] else "dense" if r["dense"] else "normal"
        rows.append(
            f"<section><img src='{i:05d}.jpg'><div><h3>#{i} {m['fmt']} · {mode} · {r['lang']} · {'OCR' if r.get('ocr') else r['kind']} · {r['type']} · "
            f"{r['style']} · {r['pen']} · {r['paper']} · xh {m['xh_mm']}mm · tallest glyph {m['tallest_mm']}mm · {m['lines']}/{m['lines_want']} lines</h3>"
            f"<div class='t'>{''.join(item_html(it) for it in r['items'])}</div></div></section>"
        )
    open(f"{d}/samples.html", "w").write(f"<!doctype html><meta charset=utf-8><style>{CSS}</style>" + "".join(rows))
    ims = [cv2.resize(cv2.imread(f"{d}/{i:05d}.jpg"), (256, 360), interpolation=cv2.INTER_AREA) for i in order]
    ims += [np.full_like(ims[0], 255)] * (-len(ims) % 5)
    cv2.imwrite(f"{d}/contact_sheet.jpg", np.vstack([np.hstack(ims[k : k + 5]) for k in range(0, len(ims), 5)]))
    return len(order)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dir")
    print(build(ap.parse_args(argv).dir), "pages")


if __name__ == "__main__":
    main()
