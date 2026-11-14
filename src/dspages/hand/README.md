# dspages.hand: handwritten A4 and notebook pages

Real digital-born text with its structure (headings, paragraphs, lists, tables, blank lines, columns), written by
[hwsynth](../../../../handwriting) writers (24 types, 10 styles, 17 pens) on blank / lined / squared / dotted / legal / Seyès / margin / spiral paper.

```bash
pip install -e ".[hand]"                         # hwsynth, tesserocr, regex; Tesseract models in $DSPAGES_TESSDATA (rare scripts only)
python -m dspages.hand texts  --out texts.jsonl  # prose pages, books, papers of the corpus (digital text layer, page type by embeddings)
python -m dspages.hand edu    --out edu.jsonl    # long lecture / textbook passages (dense pages)
python -m dspages.hand tables --out tables.jsonl # real tables
python -m dspages.hand plan   --texts texts.jsonl edu.jsonl tables.jsonl --out plan.jsonl --n 10000
python -m dspages.hand check  plan.jsonl         # shares of every factor, Cramer's V between factors
python -m dspages.hand run    --plan plan.jsonl --out OUT --workers 6        # or tools/kaggle_handpages for Kaggle CPU shards
python -m dspages.hand labelled --out DIR        # handwritten versions of labelled pages, next to the originals
python -m dspages.hand rare   --out DIR          # A4 pages of rare scripts: real pages read by OCR with their structure
python -m dspages.hand sheet  DIR                # HTML sheet and contact sheet of samples
```

| module | role |
|---|---|
| `config` | paths (`DSPAGES_UVTM`, `DSPAGES_PAGETYPE_LABELS`, `DSPAGES_TESSDATA`) |
| `corpus`, `langs` | catalogue queries (documents, languages, page types); script of a language, text plausibility |
| `texts`, `edu`, `tables_src` | passage extraction from digital PDFs: every line of the page kept (h / p / li with level / l single line / v / table), wrapped lines joined, page furniture dropped; blank lines, alignment and columns by `structure` |
| `pagekind` | plain / poems / formatted from the text layer only (line geometry, fonts, sizes, bullets, rules): ~8 ms a page; tools/pagekind_sources.py + tools/pagekind_eval.py |
| `ocr`, `structure` | a page read by Tesseract with positions; blocks to items with gaps, heading sizes and columns |
| `plan` | exact shares, independent factors, passages matched to the capacity of a page |
| `paper`, `page`, `tables`, `util` | papers (formats A4 / A5 / B5), layout and drawing of a page, hand-drawn tables, text helpers |
| `run`, `sheet`, `check`, `labelled`, `rare` | generation, review sheets, plan checks, sample sets |

`page.make(plan, items)` -> `(rgb uint8 1440 x 1024, meta)`; the meta has every word box in px, the paper, the fonts and the size.

Factors of a plan (exact shares, paired by independent shuffles): language (English 40 %, other shares by script group, equal inside a group),
writer type, style, pen, writer (type individual or fitted real writer), sheet format, mode (normal / dense / table), paper, fill, x-height,
source kind. Limits: x-height at most 3.4 mm, tallest word 15 mm, lines follow the ruling, lines on free paper wander together with
their own variation. Languages without a usable text layer (ur ps ku ckb sd) have no pages until a source is added.
Split train / validation by `doc`: pages of one source document share it.
