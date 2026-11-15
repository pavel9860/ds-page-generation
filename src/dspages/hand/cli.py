"""python -m dspages.hand <command> [args]

texts    passages from digital PDFs (prose pages, books, papers)         edu     long lecture / textbook passages
tables   real tables of digital PDFs                                      plan    the plan of the pages (exact shares, independent factors)
check    balance and leakage of a plan                                    run     the pages of a plan shard (jpg + meta), resumable
labelled handwritten versions of labelled pages (pagetype labels.json)    rare    A4 pages of rare scripts from OCR'd real pages
sheet    HTML sheet and contact sheet of a directory of samples
"""

import importlib
import sys

COMMANDS = {
    "texts": "texts",
    "edu": "edu",
    "tables": "tables_src",
    "plan": "plan",
    "check": "check",
    "run": "run",
    "labelled": "labelled",
    "rare": "rare",
    "sheet": "sheet",
}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in COMMANDS:
        sys.exit(__doc__)
    importlib.import_module(f".{COMMANDS[argv[0]]}", __package__).main(argv[1:])
