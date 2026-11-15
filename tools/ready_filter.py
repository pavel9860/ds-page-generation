"""Drop from the labelling page (tests/legacy_label/tables.json, original kept as tables_all.json) every glyph of a font that has a published table (route is not "labelled"). Tables left empty disappear.
Also used by export_labelled.py: a published table wins over a label.   python tools/ready_filter.py"""

import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from dspages import paths as _P  # noqa: E402
from dspages.layout import legacy_tables as T

L = f"{_P.ROOT}/tests/legacy_label/"


def ready(font, ch):
    r = T.route(font)
    if not r or r[0] == "labelled":
        return False
    try:
        out = T._decode(font, ch)
    except Exception:
        return False
    return bool(out) and out.strip() != "" and out != ch


def exists(font):
    """True when a published table is routed for the font: its glyphs leave the labelling page whole."""
    r = T.route(font)
    return bool(r) and r[0] != "labelled"


if __name__ == "__main__":
    src = L + "tables_all.json"
    if not os.path.exists(src):
        shutil.copy(L + "tables.json", src)
    tabs, kept, dropped = [], 0, 0
    for t in json.load(open(src, encoding="utf-8")):
        items = [i for i in t["items"] if not exists(i["font"])]
        dropped += len(t["items"]) - len(items)
        kept += len(items)
        print(t["id"], len(t["items"]), "->", len(items)) if len(items) != len(t["items"]) else None
        if items:
            tabs.append({**t, "items": items})
    json.dump(tabs, open(L + "tables.json", "w", encoding="utf-8"), ensure_ascii=False)
    print("tables", len(tabs), "glyphs kept", kept, "dropped", dropped)
