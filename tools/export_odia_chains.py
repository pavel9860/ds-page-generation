"""Odia converters of OdiaWikimedia/Converter (GPL-3: Akruti-Sarala, ORTT-Sarala, Aprant, Shreelipi; the HTML pages are read as data, no script runs) -> tables/chains.json.
Each page holds an ordered array of (legacy, Unicode) pairs, each applied to all its occurrences, then regex replacements that move vowel signs and the reph.

  python tools/export_odia_chains.py DIR     # DIR = a clone of the repository"""
import json
import re
import sys
from pathlib import Path

from dspages.layout.legacy import CACHE

FILES = {"akruti_sarala": "Akruti-Sarala - Unicode Converter.htm", "orttsarala": "ORTT-Sarala - Unicode Converter.htm",
         "aprant": "Aprant - Unicode Converter.htm", "shreelipi_odia": "Shreelipi - Unicode Odia Converter.htm"}
STR = re.compile(r'"((?:[^"\\]|\\.)*)"')
POST = re.compile(r'modified_substring\s*=\s*modified_substring\.replace\(\s*/((?:[^/\\\n]|\\.)+)/g\s*,\s*"((?:[^"\\]|\\.)*)"\s*\)')


def js_string(s):
    return s.encode().decode("unicode_escape").encode("latin-1").decode() if "\\" in s and re.search(r"\\u|\\x", s) else s.replace('\\"', '"')


def chain(text):
    a = re.search(r"var \w+ = new Array\(", text)
    b = re.search(r"function Replace_|var text_array_length", text[a.end():])
    body = text[a.end():a.end() + b.start()]
    body = body[:body.rindex(")")]
    lines = [re.sub(r"\s//.*$", "", ln) if not ln.lstrip().startswith("//") else "" for ln in body.splitlines()]
    items = [js_string(m) for ln in lines for m in STR.findall(ln)]
    out = [[re.escape(x), y.replace("\\", "\\\\")] for x, y in zip(items[0::2], items[1::2])]
    post = "\n".join(ln for ln in text[a.end() + b.start():].splitlines() if not ln.lstrip().startswith("//"))
    for rx, rep in POST.findall(post):
        out.append([rx, re.sub(r"\$(\d)", r"\\\1", rep)])
    return out


def main():
    d = Path(sys.argv[1])
    f = Path(CACHE) / "tables" / "chains.json"
    chains = json.loads(f.read_text(encoding="utf-8"))
    for name, fn in FILES.items():
        chains[name] = chain((d / fn).read_text(encoding="utf-8", errors="replace"))
        print(name, len(chains[name]), "rules")
    f.write_text(json.dumps(chains, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
