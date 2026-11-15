# pagegen: page generation

`dspages.pagegen` writes handwriting onto pages. It depends only on hwsynth (handwriting lines) and pymupdf; no other dspages module. Its numbers are `pagegen.config.GenCfg`.

| module | what |
|---|---|
| `config.py` | `PageCfg`/`PAGE` (page size, margins, handwriting height share), `GenCfg` (pen, scan look, form finder, boxes) |
| `canvas.py` | `Canvas` (paper, ruling), `Hand` (writer, ink strips), `new_writer`, `line_boxes`, font helpers (`pick_fonts`, `strip_missing`) |
| `handfonts.py` | which fonts may write a script (`can_write`, `Restricted`) |
| `paper.py` | `page_segments` (print boxes, form fields of a PDF page), `free_mask`, `blend`, `degrade` (scan look) |
| `layouts.py`, `page.py` | the handwritten page layouts (`LAYOUTS`) and `handwritten_page(layout, pool, cfg, rng)` |
| `forms.py` | `render_form`, `capacity`, `fill`: fields of a printed form filled by one writer at one size |
| `hand.py`, `source.py` | `FontHand`/`writing_height` for forms and notes; text blocks of real documents by kind |

The text pool of a page is a dict `{"heading": [...], "para": [...], "item": [...]}` (a layout may also use `"formula"`).
`handwritten_page` returns `None` when the page is rejected (too little text fits); the caller draws another seed.

```bash
python tools/generate_page.py out.png 1 plain      # layouts: sorted(pagegen.layouts.LAYOUTS)
```
Printed-page notes (what a reader writes between the lines) need the text pool and line bank of the project and live in the consumer (`dspages.multilang.notes`).
