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
    return load_manifest(full_small.paths.manifest)


@pytest.fixture(scope="session")
def fonts(full_small):
    from dspages.layout.text import find_fonts
    return find_fonts(full_small.paths.font_dirs)
