from dataclasses import replace
from pathlib import Path

import pytest

from dspages.config import get_preset

DATA = Path(__file__).parent / "data"


def small(name):
    """The preset on a coarse grid with few blend slices: same physics, fast."""
    P = get_preset(name)
    g = replace(P.geometry, mesh=(64, 90), n_slices=3, strip_segments=100)
    return replace(P, geometry=g)


def fake_manifest():
    """Pages of every script group (raster and restyled pages, as the manifest writes them after its checks)."""
    langs = ["en", "eu", "cyr", "zh", "el", "unknown"]
    out = [dict(source_path=f"x{i}.png", page_index=0, kind="raster", language=langs[i % 6], category=None)
           for i in range(60)]
    out += [dict(source_path=f"b{i % 3}.pdf", page_index=i, kind="refont", language=langs[i % 5], category="books_pdf") for i in range(60)]
    return out


@pytest.fixture(scope="session")
def full_small():
    return small("full")


@pytest.fixture(scope="session")
def demo():
    return get_preset("demo")


@pytest.fixture(scope="session")
def manifest(full_small):
    from dspages.layout.sources import load_manifest
    if not Path(full_small.paths.manifest).exists():
        pytest.skip("manifest not available")
    m = load_manifest(full_small.paths.manifest)
    if not all("window" in e or "scale" in e for e in m[:100]):
        pytest.skip("manifest of an older format: rebuild it")
    return m


@pytest.fixture(scope="session")
def fonts(full_small):
    from dspages.layout.refont import font_set
    return font_set(full_small.paths.font_dirs)
