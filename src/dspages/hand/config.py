"""Paths of the handwritten-pages pipeline. Every one can be set by an environment variable; nothing else in the package names a path."""
import os
from dataclasses import dataclass


def _env(name, default):
    return os.environ.get(name, default)


@dataclass(frozen=True)
class HandPaths:
    uvtm: str = _env("DSPAGES_UVTM", "/run/media/p/D/ML_DS/UVTM")
    labels: str = _env("DSPAGES_PAGETYPE_LABELS", "/run/media/p/D/BUISNESS/corpus_collecting/pagetype/data/labels.json")

    @property
    def layouts(self):
        return f"{self.uvtm}/Layouts"

    @property
    def manifest(self):
        """SQLite manifest of the corpus: plan (documents), lang_res (languages), old_meta (topics)."""
        return f"{self.layouts}/manifest.sqlite"

    @property
    def doc_types(self):
        """Page type of every document from the embedding classifier (jsonl: path, label, p)."""
        return f"{self.uvtm}/emb_out/doc_types.jsonl"

    @property
    def tessdata(self):
        return _env("DSPAGES_TESSDATA", f"{self.layouts}/archive/tessdata")


PATHS = HandPaths()
