"""Published legacy-font tables from the downloads (legacy/downloads/) into tables/, read as data (nothing of a download runs):
- tables/legacylipi.json from the legacylipi mappings (indic-legacy-pdf, MIT, a fork of biswasbiplob/legacylipi): Devanagari legacy fonts (DVB-TT / DVBW-TT Surekh,
Chanakya, Walkman Chanakya, Shree-Dev, Shree-Lipi, Shusha, APS-DV, Kruti Dev). The dict literals of its mapping modules are read as data (ast.literal_eval,
nothing of the download runs): {table: {code or code sequence: text}}, merged in the order the module merges them.
- tables/payyans.json from libindic unicode-conversion-maps (Payyans maps, GPLv3: Bengali Banglapedia, Malayalam ML-TT, Tamil TM-TT Valluvar, Shree-Lipi ...):
  {map: {code or code sequence: text}}
python tools/import_tables.py"""

import ast
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from dspages.layout import legacy_tables as T  # noqa: E402

DOWN = os.path.join(os.path.dirname(T.TABLES), "downloads")
SRC = os.path.join(DOWN, "indic-legacy-pdf")
PARTS = ("MAPPINGS", "LIGATURES", "HALF_FORMS")  # MappingTable.all_mappings: mappings, then ligatures, then half forms
DVB = ("CONSONANTS", "MATRAS", "SPECIAL", "CONJUNCTS", "ADDITIONAL")  # dvb_tt.py's own DVB_TT_COMPLETE_MAPPING


def literals(path):
    """{NAME: dict} of the module's top-level dict literals."""
    out = {}
    for node in ast.parse(open(path, encoding="utf-8").read()).body:
        target = (
            node.targets[0]
            if isinstance(node, ast.Assign)
            else node.target
            if isinstance(node, ast.AnnAssign)
            else None
        )
        if isinstance(target, ast.Name) and isinstance(node.value, ast.Dict):
            try:
                out[target.id] = ast.literal_eval(node.value)
            except ValueError:  # a dict of objects (BUILTIN_MAPPINGS): not a table
                continue
    return out


def payyans():
    """tables/payyans.json: each libindic map, `code=text` lines ('#' comments; the BOM of a first line dropped)."""
    mdir = os.path.join(DOWN, "unicode-conversion-maps", "maps")
    out = {}
    for f in sorted(os.listdir(mdir)):
        m = {}
        for line in open(os.path.join(mdir, f), encoding="utf-8-sig"):
            line = line.rstrip("\n").lstrip("\ufeff")
            if line.startswith("#") or "=" not in line[1:]:
                continue
            k = line.index("=", 1)  # a code may itself be "=": the separator is the first one after it
            if line[:k]:
                m[line[:k]] = line[k + 1 :]
        out[T.key(os.path.splitext(f)[0])] = m
    json.dump(out, open(os.path.join(T.TABLES, "payyans.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    shutil.copy(
        os.path.join(DOWN, "unicode-conversion-maps", "README.md"),
        os.path.join(T.TABLES, "LICENSES", "libindic-payyans-GPL3.txt"),
    )
    print({k: len(v) for k, v in out.items()})


def main():
    payyans()
    src = SRC
    mdir = os.path.join(src, "src", "legacylipi", "mappings")
    tables = {}
    for f in sorted(os.listdir(mdir)):
        if not f.endswith(".py") or f == "__init__.py":
            continue
        lit = literals(os.path.join(mdir, f))
        stems = sorted(
            {
                k.rsplit("_", 1)[0] if k.endswith(("_MAPPINGS", "_LIGATURES")) else k[: -len("_HALF_FORMS")]
                for k in lit
                if k.endswith(tuple("_" + p for p in PARTS))
            }
        )
        for stem in stems:
            tables[stem.lower()] = {c: t for p in PARTS for c, t in lit.get(f"{stem}_{p}", {}).items()}
        if f == "dvb_tt.py":
            tables["dvb_tt"] = {c: t for p in DVB for c, t in lit[f"DVB_TT_{p}"].items()}
    tables = {k: v for k, v in tables.items() if v}
    json.dump(
        tables, open(os.path.join(T.TABLES, "legacylipi.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0
    )
    shutil.copy(os.path.join(src, "NOTICE.md"), os.path.join(T.TABLES, "LICENSES", "legacylipi-MIT.txt"))
    print({k: len(v) for k, v in tables.items()})


if __name__ == "__main__":
    main()
