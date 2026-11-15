"""Data locations: DSPAGES_* variables first, legacy names second, a .env file only where the environment is silent."""

import importlib

import pytest

import dspages.paths as P

VARS = (
    "DSPAGES_ROOT",
    "DSPAGES_UVTM",
    "MULTILANG_ROOT",
    "HWDET_UVTM",
    "DSPAGES_CORPUS",
    "DSPAGES_LAYOUTS",
    "DSPAGES_ENV",
)


@pytest.fixture
def fresh(monkeypatch, tmp_path):
    for k in VARS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DSPAGES_ENV", str(tmp_path / "none.env"))
    yield lambda: importlib.reload(P)
    monkeypatch.undo()
    importlib.reload(P)


def test_defaults_sit_under_data(fresh):
    p = fresh()
    assert p.ROOT == P.Path("data") and p.LAYOUTS == p.ROOT / "Layouts" and p.INFO == p.LAYOUTS / "archive" / "info"


def test_new_variable_wins_over_legacy(fresh, monkeypatch):
    monkeypatch.setenv("HWDET_UVTM", "/legacy")
    assert fresh().ROOT == P.Path("/legacy")
    monkeypatch.setenv("DSPAGES_ROOT", "/new")
    p = fresh()
    assert p.ROOT == P.Path("/new") and p.CORPUS == P.Path("/new/corpus_collecting")


def test_env_file_fills_only_unset_variables(fresh, monkeypatch, tmp_path):
    (tmp_path / "my.env").write_text('# comment\nDSPAGES_ROOT="/from_file"\nDSPAGES_CORPUS=/corpus_file\n')
    monkeypatch.setenv("DSPAGES_ENV", str(tmp_path / "my.env"))
    monkeypatch.setenv("DSPAGES_CORPUS", "/corpus_env")
    p = fresh()
    assert p.ROOT == P.Path("/from_file") and p.CORPUS == P.Path("/corpus_env")
