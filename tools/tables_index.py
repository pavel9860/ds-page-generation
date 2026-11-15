"""One index of every decoding table dspages.layout.legacy_tables can route: tables/INDEX.json and tables/INDEX.md.
Per kind: where the table comes from; per font family key: the kind, the table, how many codes it has (labelled: by hand / guessed).
python tools/tables_index.py"""

import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from dspages.layout import legacy_tables as T
from dspages.layout import padma

SOURCE = {
    "padma": "Padma (Mozilla Public License), encodings/*.js: Telugu, Devanagari, Tamil, Bengali, Malayalam, Gurmukhi newspaper and DTP fonts",
    "preeti": "Preeti-family rules (Nepali / Hindi legacy fonts), tables/preeti.json",
    "ml": "Malayalam ML-TT / ML-Revathi family maps, tables/ml/",
    "nudi": "Nudi / Baraha Kannada, tables/nudi.json",
    "bijoy": "Bijoy / SutonnyMJ Bangla, tables/bijoy.json",
    "geet": "Geetanjali Bangla, tables/geetanjali.json",
    "pun": "Punjabi converters (Anmol Lipi, Asees, Chatrik, Joy, Satluj ...), tables/punjabi/",
    "chain": "ordered replace chains: Open-SL Sinhala, OdiaWikimedia/Converter (GPL-3), tables/chains.json",
    "tbl": "UTFC Tibetan .tbl (GPL-3), tables/tibetan/",
    "map": "plain key -> letter maps (Georgian Nusx, Tibetan Chogyal, Win Innwa Myanmar from python-myanmar), tables/charmaps.json",
    "cp": "a Windows / Mac code page",
    "priya": "Priyaanka Telugu, tables/priyaanka.json",
    "labelled": "read glyph by glyph from the pictures (tests/legacy_label, tools/export_labelled.py), tables/labelled.json",
}
fonts = defaultdict(dict)


def put(key, kind, name):
    fonts[key].setdefault(kind, name)


for k in padma.faces():
    put(T.key(k), "padma", k)
for k in T._preeti():
    put(k, "preeti", "preeti")
for k in T._ml():
    put(k, "ml", T._ml()[k])
for k in T._aliases():
    kind, name = T._aliases()[k]
    put(k, kind, name)
for stem in T._pun_files():
    if stem not in ("groups", "tight_groups", "unicode"):
        put(T.key(stem), "pun", stem)
for k in json.load(open(os.path.join(T.TABLES, "chains.json"))):
    put(T.key(k), "chain", k)
lab, src = T._labelled(), T._label_source()
for k, m in lab.items():
    c = Counter(src.get(k, {}).values())
    fonts[k]["labelled"] = {"codes": len(m), "by_hand": c["label"], "guessed": len(m) - c["label"]}
out = {"kinds": SOURCE, "fonts": {k: v for k, v in sorted(fonts.items())}}
json.dump(out, open(os.path.join(T.TABLES, "INDEX.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
kinds = Counter(next(iter(v)) for v in fonts.values())
with open(os.path.join(T.TABLES, "INDEX.md"), "w", encoding="utf-8") as f:
    f.write(
        "# Legacy font decoding tables\n\nLookup order in `legacy_tables.route`: published table first; a hand label or guess only fills codes the table leaves as typed.\n\n## Table kinds\n\n"
    )
    for k, v in SOURCE.items():
        f.write(f"- **{k}** ({sum(1 for x in fonts.values() if k in x)} font keys): {v}\n")
    f.write("\n## Font keys\n\n| font key | tables |\n|---|---|\n")
    for k, v in sorted(fonts.items()):
        f.write(
            f"| {k} | "
            + "; ".join(
                f"{a}:{b if not isinstance(b, dict) else str(b['by_hand']) + ' by hand + ' + str(b['guessed']) + ' guessed'}"
                for a, b in v.items()
            )
            + " |\n"
        )
print(len(fonts), "font keys", dict(kinds), "->", T.TABLES + "/INDEX.md")
