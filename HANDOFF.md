# Hand-off: seed pages for the Lens search (2026-11-09)

Goal: 200 hw pages, 200 filled forms, 20 notes in non-Latin scripts, seeds chosen to fill gaps / extend the border of the hw / filled classes in the 16-d head space, equal quota per language (45), translation where a language has no layout near a gap. Nothing is committed.

## State
- Seeds: `ds100k/seeds/{hw,filled,notes}.csv` (tools/seed_pick.py; `src` set = layout translated into `lang`; 199 / 198 / 20 are translated, because no layout of any language lands within TOL=0.3 of a gap: median 0.63; decided: go with translations).
- Run dir `ds100k/seeds/run` (tools/seed_run.sh OUT [forms|hw|prev]): texts translated (hw 109 of 200, notes 15 of 20 usable; the 36 lost not investigated), hw+notes pages 124 generated and embedded (`emb_hw.npz`), 664 earlier translated pages embedded (`emb_translated.npz`), forms line running since 12:50 (browser translation -> `work/*.pdf` -> prepare -> `filled/` -> `emb_filled.npz`; 74 of 200 form pdfs were done, the rest resumes).
- Translation = Google Translate in headless Brave via selenium (installed in .venv); the HTTP endpoint is blocked (429) after my 6-worker run. Keep Google workers at 2 in all.

## Next
1. Wait for `run/emb_filled.npz`; then check the gap fit: distance of `Z` of the embedded pages (hw, filled, translated) to the targets (`seed_pick.targets`) against the landing used for the selection; refine `ds100k/shift/translate.json` (only 30 pages, 2 per script) with the new pairs.
2. Find why 36 of 220 seed texts failed (rare.translated needs 80% of lines translated; blocks / form lines).
3. Then `lens seeds` (dspages.acquire) with the page dirs `run/hw`, `run/filled`.

## Code touched this session (all uncommitted)
- layout/legacy.py (glyph audit of Unicode Indic fonts adopted only when the page reads better, stray / PUA / Mac punctuation, cached route, fast fit), legacy_tables.py, hand/texts.py (follow, layout, native, Arabic lam-alef, glue), hand/plan.py (`--as-is`), hand/rare.py (`write`, `translated(…, sl)`), tools/seed_pick.py, seed_texts.py, seed_run.sh, shift_test.py (`translate`, `embed` ops).
- Data: tables/aliases.json lost 12 wrong Tamil aliases (Bamini family, they are not Amudham); it was restored once by something, check it still has no `akarathi`.

## Known open reading defects
Telugu Rachana (unrouted) / Gidugu (scrambled map), Myanmar, Odia `ª`, Kannada `ø`, Persian wrong glyphs, Hindi Kokila; accepted for now.

## Rules from the user
Short answers; never poll manually (use Monitor); reuse existing tools / results and ask before implementing new ones; keep the codebase compact; no `pkill -f`; test on small samples before full runs.
