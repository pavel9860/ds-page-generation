# Tasks (delete a task when it is done)

## Triplets (tools/triplets.py, manifest m3 = the first 600-page manifest with its garbled pages replaced)
- Overlaps left: widened single-line blocks over a neighbour (cases 8, 9, 38, 40, 55): limit the widening by every block of the line band.
- Frames filled short (stream ended): the spread of lines is skipped when a box lies in the frame; look at 080 (frame 1 at 77%).
- Underlines and rules drawn inside a frame stay where the source had them while its text is set again (025).
- Devanagari of some PDFs comes out with its vowel signs apart (025: "म ुंबई").

## Code
- Tests: cover layout/template.py further (obstacles, pictures, the reflow of sections, column balance).
- Manifest: the listing has to be rebuilt for the new language groups and the new folders of LayoutCfg.sources
  (python -m dspages.prep.files, then the manifest).

## Pipeline (do not touch geometry solve_strip / solve_sag)
- Geometry reads the noise scale the layout of the same process left (deep creases through layout.paper.smooth_noise):
  `all` and `geometry` give different surfaces for one index. Pass the scale explicitly.
- Refactor, simplify, consolidate the pipeline (generate, render, paper effects).
- 200 UV samples (flat, warped, dewarped, UV map, 3D map) with the new layout and the noise scale: python -m dspages all, then tools/gallery.py.
