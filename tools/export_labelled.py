"""Export the glyph labels of tests/legacy_label as the decoding table used by dspages.layout.legacy_tables.decode:
  Layouts/archive/info/legacy/tables/labelled.json         {font family key: {typed char: Unicode text}}
  Layouts/archive/info/legacy/tables/labelled_source.json  {font family key: {typed char: "label" | "guess:<where from>"}}
"label" = answered by hand (labels.jsonl, status ok, last answer); "guess" = a glyph nobody answered: the best candidate of the labelling page (checked words,
phrase-OCR statistics, then the server's shape / line / statistics list, the first one that belongs to the table's script).
The server list needs the labelling server (tests/legacy_label/server.py, port 8770); without it only the checked words and the OCR statistics are used.
python tools/export_labelled.py"""

import json
import re
import sys
import unicodedata
import urllib.parse
import urllib.request

import regex

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from ready_filter import ready

from dspages import paths as _P  # noqa: E402

L = f"{_P.ROOT}/tests/legacy_label/"
T = f"{_P.ROOT}/Layouts/archive/info/legacy/tables/"
key = lambda name: re.sub(r"[^a-z0-9]", "", name.split("+")[-1].lower())
SCRIPT = {"Oriya": "Oriya"}
lab = {}
for line in open(L + "labels.jsonl", encoding="utf-8"):
    r = json.loads(line)
    lab[r["id"]] = r
words = json.load(open(L + "words_cands.json", encoding="utf-8"))
ocr = json.load(open(L + "ocr_cands.json", encoding="utf-8"))


def server_list(iid):
    try:
        return json.load(
            urllib.request.urlopen("http://127.0.0.1:8770/cands?id=" + urllib.parse.quote(iid) + "&k=60", timeout=20)
        )
    except Exception:
        return []


def guess(iid, script):
    ok = regex.compile(rf"^[\p{{Script={SCRIPT.get(script, script)}}}\p{{Script=Common}}\p{{Script=Inherited}}]+$")
    for src, lst in (
        ("words", words.get(iid, [])),
        ("ocr", [c for c in ocr.get(iid, []) if len(c.strip()) == len(c) and len(c) <= 3]),
        ("shape", server_list(iid)),
    ):
        for c in lst:
            if c and ok.match(c):
                return unicodedata.normalize("NFC", c), src
    return None, None


res, srcs = {}, {}
n_label = n_guess = 0
for t in json.load(open(L + "tables_all.json", encoding="utf-8")):
    fams = {key(i["font"]) for i in t["items"]} | set(t["families"])
    for i in t["items"]:
        r = lab.get(i["id"])
        c = chr(int(i["id"].split("|")[1], 16))
        if ready(i["font"], c):  # a published table decodes it: it wins over any label
            continue
        if r and r["status"] == "ok" and r["text"]:
            txt, src = unicodedata.normalize("NFC", r["text"]), "label"
            n_label += 1
        elif r and r["status"] in ("none", "blank"):
            continue
        else:
            txt, src = guess(i["id"], t["script"])
            if not txt:
                continue
            src = "guess:" + src
            n_guess += 1
        for f in fams | {key(i["font"])}:
            if (
                f in res and c in res[f] and not (f == key(i["font"]))
            ):  # an answer of the glyph's own font beats the table-wide default
                continue
            res.setdefault(f, {})[c] = txt
            srcs.setdefault(f, {})[c] = src
res = {f: dict(sorted(m.items())) for f, m in sorted(res.items())}
srcs = {f: {c: srcs[f][c] for c in m} for f, m in res.items()}
json.dump(res, open(T + "labelled.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
json.dump(srcs, open(T + "labelled_source.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
print(len(res), "font families;", n_label, "labelled glyphs,", n_guess, "guessed ->", T + "labelled.json")
