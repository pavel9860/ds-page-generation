"""Which legacy fonts of the whole corpus have a decoding table: every font family of Layouts/archive/info/legacy/families.json (script -> family -> code -> text counts, from the
corpus pages) routed through dspages.layout.legacy_tables.route. Writes tables/COVERAGE.json and tables/COVERAGE.md, missing ones first by use.
python tools/tables_coverage.py"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from dspages.layout import legacy_tables as T

GROUPS = (
    ("published", "published table"),
    ("labelled", "hand-labelled / guessed table"),
    ("ocr", "read from the rendered lines (tools/ocr_tables.py)"),
    ("symbol", "symbol font, no text to recover"),
    ("unicode", "own Unicode (ToUnicode right, see UNICODE_OK.json)"),
    (None, "NO table"),
)


def load(name, default):
    path = os.path.join(T.TABLES, name)
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else default


def kind_of(font, hand, unicode_ok):
    """published | labelled | ocr | unicode | symbol | None for a font family."""
    r = T.route(font)
    if r is None:
        return "unicode" if font in unicode_ok["fonts"] else "symbol" if font in unicode_ok.get("symbol", {}) else None
    if r[0] != "labelled":
        return "published"
    return "labelled" if font in hand or font not in T._ocr() else "ocr"


def rows():
    hand, unicode_ok = load("labelled.json", {}), load("UNICODE_OK.json", {"fonts": {}})
    out = []
    for script, fams in load("../families.json", {}).items():
        for f, codes in fams.items():
            kind = kind_of(f, hand, unicode_ok)
            out.append(
                {
                    "script": script,
                    "family": f,
                    "uses": sum(sum(v.values()) for v in codes.values()),
                    "codes": len(codes),
                    "table": kind,
                    "route": (T.route(f) or [None])[0],
                }
            )
    return sorted(out, key=lambda r: (r["table"] is not None, -r["uses"]))


def write_markdown(rows_):
    total = sum(r["uses"] for r in rows_) or 1
    with open(os.path.join(T.TABLES, "COVERAGE.md"), "w", encoding="utf-8") as f:
        f.write("# Table coverage of the corpus fonts\n\n")
        for kind, title in GROUPS:
            g = [r for r in rows_ if r["table"] == kind]
            f.write(
                f"- {title}: {len(g)} families, {100 * sum(r['uses'] for r in g) / total:.1f} % of the glyph uses\n"
            )
        for title, kind in (("Missing (by use)", None), ("Hand-labelled only", "labelled")):
            f.write(f"\n## {title}\n\n| script | family | uses | codes |\n|---|---|---|---|\n")
            f.writelines(
                f"| {r['script']} | {r['family']} | {r['uses']} | {r['codes']} |\n" for r in rows_ if r["table"] == kind
            )


if __name__ == "__main__":
    rows_ = rows()
    json.dump(rows_, open(os.path.join(T.TABLES, "COVERAGE.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    write_markdown(rows_)
    missing = [r for r in rows_ if r["table"] is None]
    print(
        len(rows_),
        "families; missing",
        len(missing),
        f"({100 * sum(r['uses'] for r in missing) / (sum(r['uses'] for r in rows_) or 1):.1f} % of uses)",
    )
