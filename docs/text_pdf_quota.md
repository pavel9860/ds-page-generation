| # | Genre / facet | Weight | Target pages | Era | Layout | Format | Candidate source(s) |
|---|---|---:|---:|---|---|---|---|
| 1 | Books — non-technical (prose, poetry, illustrated; tag era old/modern, illustrated yes/no) | 7.5% | 37,500 | half historical (pre-2000), half other (modern/illustrated) | single-column, some inline images | PDF primary (scan or born-digital); plain text only to patch font/OCR gaps, never standalone | Gutenberg catalog, storytracer/US-PD-Books, Common Corpus, CC-licensed modern fiction |
| 2 | Books — technical (physics, math, chemistry, computer science, electronics, table-heavy reference) | 7.5% | 37,500 | modern | single/multi-column + figures/equations/code | PDF primary; plain text only to patch font/OCR gaps | OpenStax, Open Textbook Library, Wikibooks |
| 3 | Scientific papers | 2% | 10,000 | modern | 1-column and 2-column, both | native PDF | arXiv, peS2o, OpenAlex |
| 4 | Correspondence / letters | 1% | 5,000 | historical + modern | letter/email layout | scanned PDF strongly| Gutenberg letters (as scans), modern email/chat source |
| 5 | Newspapers / magazines | 1% | 5,000 | old + modern | multi-column | PDF preferred; OCR text optional, not required | archive.org, Common Corpus PD newspapers |
| 6 | Invoice / receipt | 10% | 50,000 | modern | form/table | scanned PDF strongly preferred, not JSON | CORD-v2, SROIE, Kaggle |
| 7 | Checks (bank/financial instruments) | 4% | 20,000 | modern | fixed-field form layout, partly handwritten | scanned PDF | not yet sourced |
| 8 | Business form / memo / letter | 6% | 30,000 | mixed | memo/letter/form layout | scanned PDF strongly preferred, not plain text | Gutenberg letters (as scans), DocLayNet |
| 9 | Legal — case law / statute | 3% | 15,000 | modern | dense clauses + footnotes | scanned PDF strongly preferred, not plain text | Common Corpus (Court Listener, Caselaw Access Project, Eurlex, French Open Data) |
| 10 | Legal — scanned forms / contracts | 5% | 25,000 | modern | form/table | scanned PDF; OCR/layout metadata a bonus, not required | DocLayNet |
| 11 | Healthcare / medical forms | 5% | 25,000 | modern | fixed-field form, partly handwritten | scanned PDF/images | not yet sourced |
| 12 | Government / tax forms | 3% | 15,000 | modern | fixed-field form | scanned PDF | not yet sourced |
| 13 | Manuals / instructions | 2% | 10,000 | modern | numbered steps + diagrams | scanned PDF | DocLayNet |
| 14 | Patent | 2% | 15,000 | modern | numbered claims + figure refs | scanned PDF | DocLayNet, Common Corpus (USPTO) |
| 15 | Table-heavy record | 4% | 20,000 | modern | dense tabular | scanned PDF/images } DocLayNet, born-digital spreadsheets rendered to PDF |
| 16 | Exam / worksheet (education, partly hand-filled) | 4% | 20,000 | modern | Q&A / fill-in-the-blank, partly handwritten | scanned PDF | AI2 ARC, SciQ |
| 17 | Data ingestion — scanned handwritten tables | 1.5% | 7,500 | modern | handwritten tabular grid | scanned PDF | not yet sourced |
| 18 | Reference / encyclopedia (script floor prioritized over the 65% English default) | 2% | 10,000 | modern | prose + infobox | PDF preferred  | Wikipedia |
| 19 | Dictionary (headword/definition layout; script floor prioritized) | 0.5% | 2,500 | modern | dense columnar, symbol-heavy | PDF preferred | Wiktionary |
| 20 | Code listing  | 1% | 5000 | modern |  Formatted text, native files  |
| — | **Text/PDF subtotal** | **72%** | **360,000** | | | | |
| — | Images (scans, blueprints, sketches, handwriting) — separate phase | 28% | 140,000 | | | | |

## Target language/script distribution (applies across the whole corpus; default for every row above unless noted otherwise)

| Script/family | Languages (examples) | Target share |
|---|---|---:|
| Latin — English | en | 65% |
| Latin — other | fr, de, es, pt, it, nl, pl, sv, fi, hu, cs, ro | 17% |
| CJK | zh, ja, ko | 5% |
| Cyrillic | ru, uk, bg, sr | 3% |
| Arabic script | ar, fa, ur | 3% |
| Devanagari / Indic | hi, bn, ta, te | 3% |
| Southeast Asian | th, vi, km, my | 1.5% |
| Greek | el | 0.5% |
| Hebrew | he | 0.5% |
| Other / long tail | hy, ka, am, mn, bo, and remaining scripts | 1.5% |
