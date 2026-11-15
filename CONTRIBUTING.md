# Contributing

```bash
uv venv && uv pip install -e ".[dev,multilang,pagegen]" ruff pre-commit
pre-commit install
ruff check . && ruff format --check . && pytest -q -m "not slow"
```

- Layout of the code: `src/dspages/<package>/`; every tunable number in that package's `config.py`, every data location
  in `dspages/paths.py` (environment variables / `.env`). No absolute paths in code.
- Handwriting lines come from [hwsynth](https://github.com/pavel9860/realistic_handwriting_153_scripts); this
  library imports nothing else of the family.
- Tests stay fast and skip when the corpus is absent; long ones are marked `slow`.
- Bump `version` in `pyproject.toml` and add a `CHANGELOG.md` entry; tag `vX.Y.Z`.

By contributing you agree that your contribution is licensed under Apache-2.0.
