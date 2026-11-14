#!/bin/bash
# The seeds of tools/seed_pick.py to pages and their embeddings, three lines in parallel (a browser Google Translate worker each for texts and forms, 2 cores for the
# generators, 2 for every embedding):
#   texts -> hw / notes pages -> their embedding     forms -> filled forms -> their embedding     the earlier translated pages -> their embedding
#   bash tools/seed_run.sh OUT [forms|hw|prev]      one line only (FORM_WORKERS=2: the forms line takes the Google worker of the finished texts line)
cd "$(dirname "$0")/.."
OUT=$1
SEEDS=/run/media/p/D/ML_DS/UVTM/Layouts/archive/info/ds100k/seeds/${SEED_SET-v2}
export PYTHONPATH=src:../handwriting_detection/src THREADS=2
PY=.venv/bin/python
EMBED="$PY tools/shift_test.py embed"
hw() {
  $PY tools/seed_texts.py texts "$OUT/work" --workers 1 > "$OUT/texts.log" 2>&1
  $PY -m dspages.hand plan --texts $SEEDS/hw.jsonl $SEEDS/notes.jsonl --out "$OUT/plan.jsonl" --as-is > "$OUT/plan.log" 2>&1
  $PY -m dspages.hand run --plan "$OUT/plan.jsonl" --out "$OUT/hw" --workers 2 > "$OUT/hw.log" 2>&1
  $EMBED "$OUT/emb_hw.npz" "$OUT/hw" > "$OUT/emb_hw.log" 2>&1
}
forms() {
  $PY tools/seed_texts.py forms "$OUT/work" --workers ${FORM_WORKERS:-1} > "$OUT/forms.log" 2>&1
  $PY ../handwriting_detection/scripts/forms/fill_cpu.py "$OUT/work/forms_work" "$OUT/filled" 400 --workers 2 > "$OUT/filled.log" 2>&1
  $EMBED "$OUT/emb_filled.npz" "$OUT/filled" > "$OUT/emb_filled.log" 2>&1
}
prev() {
  [ -f "$OUT/emb_translated.npz" ] || $EMBED "$OUT/emb_translated.npz" /run/media/p/D/ML_DS/UVTM/generation/translation/pages > "$OUT/emb_translated.log" 2>&1
}
if [ -n "$2" ]; then $2; else hw & forms & prev & wait; fi
