"""Every multilang module imports and every driver compiles (a renamed constant breaks here, not in a long build)."""

import importlib
import pkgutil
import py_compile
from pathlib import Path

import pytest

import dspages.multilang as M

TOOLS = Path(__file__).parents[2] / "tools" / "multilang"


@pytest.mark.parametrize("name", sorted(m.name for m in pkgutil.iter_modules(M.__path__)))
def test_module_imports(name):
    importlib.import_module(f"dspages.multilang.{name}")


@pytest.mark.parametrize("script", sorted(TOOLS.rglob("*.py")), ids=lambda p: p.name)
def test_tool_compiles(script):
    py_compile.compile(str(script), doraise=True)


def test_generator_floor_matches_the_page_check():
    """Writing smaller than the x-height floor gives words under the page check (14 px median at 7 px per mm of x-height, measured)."""
    from dspages.hand import util
    from dspages.multilang.handpages import MIN_WORD_PX

    assert util.MIN_XH * 7.0 >= 1.15 * MIN_WORD_PX
