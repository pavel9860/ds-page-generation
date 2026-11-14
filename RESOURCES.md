# Resources

All the tables, dictionaries, models, libraries and data the code of `pages_generation` (`dspages`, `tools/`) and `handwriting_detection` (`hwdet`) uses.

**Before writing a new checker, table, reader or loader, look here first.** Each job below has one implementation; use it.

## One implementation per job

| job | use | not |
|---|---|---|
| text of a PDF page (lines, words, paragraphs, legacy fonts decoded) | `dspages.hand.texts.page_lines` / `page_words` / `page_paragraphs` / `blocks_of`, read inside `legacy.language(lang)` | `page.get_text(...)` of your own, `hwdet.translate.lines` / `paragraphs` (these only forward to `texts`) |
| garbled (pairs) | `rare.translated` / `seed_texts.form`: more than `rare.LOST` (50 %) of the blocks' `<p id>` tags lost in the whole-page translation (`seed_texts.whole`, `keyed`) | |
| does a page's text layer read clean | `dspages.hand.rare.page_blocks` -> `how`; garbled = `dspages.ingest.lang.garbled` (Google Translate through the browser, source auto-detected: a word count (ICU word breaks) outside 0.5-2x the input's; no answer raises). The only garbled test. | dictionaries, word lists, LID thresholds |
| how much a text reads as a language | `legacy.lexicon(lang)` (wordfreq Zipf, else the words of `PARAS`) and `legacy_fit.char_lm(lang)` (character trigram, Witten-Bell on `PARAS`) | |
| script of a character or text | `hwsynth.scripts.script` / `text_script` | |
| letters with their marks (regex) | `regex` `(?:\p{L}\p{M}*)`, grapheme clusters `\X` | `re` `[^\W\d_]` (it cuts Indic words at every vowel sign) |
| line breaking | `dspages.layout.linebreak.breaks` (ICU UAX #14) | |
| legacy-font decode | `legacy_tables.decode` / `run` / `decoders()`, chosen per document and font by `legacy.table_for`, else the shape fit `legacy.font_tables` | |
| Kaggle API | `dspages.kaggle_api.api()` | |
| farthest-first pick | `dspages.ingest.litefeat.farthest_first` | |
| D-optimal pick | `tools/shift_test.py d_optimal` | |
| page skew | `hwdet.detds.skew_angle(rng, table)` | |
| translation (Google) | `hwdet.gtranslate.lines` (HTTP); `tools/seed_texts.py` swaps in the browser (`browser_lines`, Selenium + Brave) | |
| handwriting fonts per script (text pages and forms) | `hwdet.pagegen.handfonts.Restricted` over the writer's index (`REALISTIC`: the curated realistic fonts of each script); `shift_test.writable` uses the same | hwsynth's whole handwriting list, a second font policy |
| form fields (spots, labels, capacity) | `hwdet.pagegen.forms.render_form(..., lang=)`, `hwdet.filledforms.spots` / `fillable` | |
| form answers | `hwdet.formfill.fill_form` (kind from the English label `kind_of`; Faker per kind; line bank for free text) | |

## Legacy-font tables

`$LAYOUTS/archive/info/legacy/tables/` (`legacy_tables.TABLES`). See also its `README.md`.

| file / dir | what | kind key in `decoders()` |
|---|---|---|
| `padma/` | Padma converters (JS, run by `py_mini_racer`): Devanagari, Bengali, Gujarati, Gurmukhi, Oriya, Tamil, Telugu, Kannada, Malayalam fonts | `padma:<face>` |
| `preeti.json` | Nepali Preeti family | `preeti:` |
| `ml/` | Malayalam ML-TT and similar | `ml:` |
| `punjabi/` | Gurmukhi (Chatrik, Asees, Awaze ...) | `pun:` |
| `nudi.json` | Kannada Nudi | `nudi:` |
| `bijoy.json` | Bengali Bijoy / SutonnyMJ | `bijoy:` |
| `geetanjali.json` | Geetanjali | `geet:` |
| `priyaanka.json` | Telugu Priyaanka (glyph names) | `priya:` |
| `chains.json` | multi-step tables (Akruti, FM Abaya ...) | `chain:` |
| `charmaps.json` | single-character maps (Tibetan Chogyal, Georgian ...) | `map:` |
| `tibetan/*2Uni.tbl` | Tibetan converters | `tbl:` |
| `khmer/`, `sinhala/` | Khmer and Sinhala tables | |
| `khmer/conversiondata.db` | KhmerConverterPHP (GPL2): 782 Khmer legacy fonts (Limon, ABC, Mondulkiri ...), 368 distinct tables; routed by font name | `khm:<n>` |
| glibc iconv `ICONV` | ARMSCII-8, GEORGIAN-ACADEMY, GEORGIAN-PS, TSCII, KOI8-T / -U / -RU, TIS-620 (the C library's tables; Python lacks them) | `iconv:<name>` |
| `khmer/unicode_limon.csv`, `sinhala/fm_abhaya.json` | not wired in (Limon is in the Khmer database; FM Abhaya to be checked against `chain:fm_abaya`) | |
| Python codecs `CODEPAGES`, `MULTIBYTE` | cp125x, mac_*, koi8_r, cp866, cp874; big5, gbk, shift_jis, euc_* ... | `cp:` / `mb:` |
| `labelled.json`, `labelled_source.json` | glyphs labelled one by one (by hand or a guess, with the source per code) | `labelled:` |
| `ocr.json` | tables read from pictures by `tools/ocr_tables.py`: `_source` is `lines` (fit) or `sheet` (two readings; provisional) | merged into `labelled:` |
| `aliases.json` | a font that uses a published table under another name | |
| `UNICODE_OK.json` | real-Unicode fonts and symbol fonts (no table) | |
| `AUDIT.json` | legacy fonts with no table (`tools/legacy_audit.py`, by family name) | |
| `COVERAGE.*`, `INDEX.*`, `CHECK.json` | coverage (`tables_coverage.py`, `tables_index.py`), own-table check (`table_check.py`) | |
| `export/` | every table as mapping files (`export_tables.py`) | |
| `paths_read.json` | readings of text drawn as glyph outlines (`legacy.path_lines`) | |

Other files under `legacy/`:

| path | what |
|---|---|
| `docs/<stem>.pkl` | the decode decisions of each document: `table_for` per font (valid only while `tables_sig` matches the current table set; otherwise decided again), shape tables |
| `ref_<script>.npz` | glyph-shape references |
| `families.json` | learned shape tables per family |
| `audit/` | `tools/table_audit.py`: every legacy font of the corpus through the real reader (`scan.parquet`, `decode.parquet`, `tables.parquet`) |

### Table tools, and what each one covers

| tool | covers |
|---|---|
| `tools/table_audit.py` | **the one audit**: every font of the non-Latin documents (anonymous names too) and every table, through the real reader, judged by invalid clusters; writes `AUDIT.json` |
| `tools/ocr_tables.py` | builds the tables (aliases, ocr, unicode) |
| `_aside/legacy_audit.py`, `_aside/table_check.py` | replaced by `table_audit.py` |
| `tests/test_legacy_fit.py`, `tests/test_decoding.py`, `tests/test_reader.py` | tests |

### Known weak spots

- Labels from glyph sheets: about 80 % right. Example: `dvbwttsurekhen` gives invalid Devanagari (`Ÿ`→`त` should be `त्`, `ü`→`ं`).
- `legacy_evidence.typed_codes` takes the alphabetically first lines, not a spread of the font's text.
- `table_for` judges candidates with `lexicon`, which is the 112-word `PARAS` list for languages wordfreq lacks (hi has wordfreq; mr, ne, bn and others do not).

## Language data

| resource | where | used by |
|---|---|---|
| wordfreq (42 languages: ar bg bn ... hi ... ru ta uk ur zh; no mr ne gu pa te kn ml si th ka hy) | pip | `legacy.lexicon`, `word_share`, `texts`, `langcheck` |
| real paragraphs per language (small: hi 31 paragraphs, bn 6) | `$UVTM/handwriting/handwriting_detection/langds/paras/<lang>.json` (`legacy.PARAS`) | `char_lm`, `lexicon` fallback, `langs._paras` |
| ICU `libicuuc.so.78` | system | `layout.linebreak` |
| language ID: `langid` (`langs`, `langcheck`), `langdetect` (`langcheck`), `lingua` (`hwdet.dettext`; not installed in this venv), `langstack.joblib` | pip / `archive/info` | |
| `ftfy` | pip | `langcheck` (mojibake) |
| Faker (40.x) | pip | `hwdet.dettext` / `formfill` (names, dates, addresses per locale) |
| line bank | `hwdet.linebank` | free-text form answers |

## Fonts and shaping

| resource | where | used by |
|---|---|---|
| Noto (`/usr/share/fonts/truetype/noto`, `NotoSansCJK-Regular.ttc`) | system | `legacy` references |
| updated Noto Thai | `~/.fonts/noto-thai` | |
| `~/.fonts`, `/usr/share/fonts`, `/usr/local/share/fonts` | | font sets (`refont.families`: the newest `head.fontRevision` per family wins; `RENAMED`) |
| font index of `hwsynth` | `~/.cache/hwsynth/fonts` | `hwsynth.writer._index`, `translate.fonts_for` |
| handwriting fonts | `hwsynth` | handwriting synthesis |
| reference fonts (full files of fonts PDFs embed with wrong text maps) | `$UVTM/fonts_ref` (`DSPAGES_FONTS_REF`): Khmer OS (Debian fonts-khmeros, GPL), Lohit (Debian fonts-lohit-*, OFL), Nikosh, Kalpurush, SolaimanLipi (hmoazzem/bangla-fonts); sources in `legacy/downloads/fonts_ref_src`. Microsoft fonts (Mangal, Nirmala UI, Kokila, Iskoola Pota ...) are not redistributable: only from a licensed Windows | `layout.reffonts` |
| `~/.cache/dspages/fonts.pkl` | | `refont` font cache |
| fontTools (`head`, cmap, glyph outlines), uharfbuzz (shaping) | pip | `refont`, `reffonts`, `refext`, `langs` |
| PyMuPDF `<1.28` (1.28 crashes with a double free on a broken structure tree) | pip | everywhere |

## Models

| model | where | used by |
|---|---|---|
| YOLO handwriting / text detector `yolo26s-ds3k3250v3.pt`, `hwdet-r4-s.pt` | `handwriting_detection/models` | `hw_det`, `kaggle_topup`, `lens` |
| DocLayNet YOLOv11n ONNX | `~/.cache/dspages/yolov11n-doclaynet.onnx` (onnxruntime) | `layout.pictures` |
| ResNet-18 ImageNet | `~/.cache/torch/hub/checkpoints` | |
| SigLIP 2 embedding (on Kaggle, 2 x T4) | `corpus_collecting/pagetype/embed_kaggle.py` | `ds100k`, `kaggle_topup`, `shift_test embed` |
| transformers models | | `diverse_ds`, `maps3d_emb` |
| XGBoost | | `fillpic.detector`, `pagekind_eval`, `maps3d_*` |
| Tesseract (tesserocr) | system | `legacy_evidence`, `hand.ocr` |

## Corpus and work data (`$UVTM` = `/run/media/p/D/ML_DS/UVTM`, `$LAYOUTS` = `$UVTM/Layouts`)

| data | what |
|---|---|
| `$LAYOUTS/manifest.sqlite` | the corpus manifest, with these tables: `files` (hash, path, ext), `plan` (doc, lang, type, pages ...), `fonts` (doc, font family key), `fonts_done`, `font_lang`, `docs`, `feats`, `labels`, `lang_*`, `scan_pct`, `hw_det`, `jpeg_grid` ... |
| `$LAYOUTS/forms`, `forms_clean`, `archive/...` | the documents |
| `$LAYOUTS/archive/info/` | per-tool outputs, including `form_fields.parquet` (one `fillable` result per form page), `ds100k/`, `pages_*`, `legacy/` |
| `$LAYOUTS/archive/info/ds100k/shift/pairs/` | `shift_test pairs` (`PAIRS_DIR` overrides it): `layouts.csv`, `img/`, `tr/`, `forms_tr/`, `forms_*/forms_work` |
| `$UVTM/tests/` | sample and check outputs (`pairs_sample100`, `forms_audit`, `legacy_label` ...) |
| `$UVTM/handwriting/` | hwdet work (`HWDET_WORK`), `langds/` |
| `corpus_collecting/` | crawl, `pagetype` (thumbs, `pack_npz`, embedding), `kindnet_data` |

## Environment

- Python: `pages_generation/.venv` (3.14). `handwriting_detection/.venv` (3.12) is a different venv.
- Tools that use `hwdet` need `PYTHONPATH=src:../handwriting_detection/src:tools:../corpus_collecting`.
- Variables:
  - `DSPAGES_LAYOUTS`, `DSPAGES_UVTM`
  - `HWDET_UVTM`, `HWDET_WORK`, `HWDET_DATA`, `HWDET_LAYOUTS`, `HWDET_GENERATION`, `HWDET_RESULTS`
  - `PAIRS_DIR`, `OMP_NUM_THREADS`, `HWDEBUG`
