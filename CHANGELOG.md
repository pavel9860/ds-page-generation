# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions: [SemVer](https://semver.org/).

## 0.3.0 (2026-11-15)
### Changed
- One place for data locations: `dspages/paths.py` (environment variables `DSPAGES_ROOT`, `DSPAGES_CORPUS`,
  `DSPAGES_LAYOUTS`, `DSPAGES_HWDET_WEIGHTS`, or a `.env` file); no absolute paths in code, tools or tests.
- Tunable numbers moved from module bodies into the `config.py` of each package.
- Code formatted with ruff (PEP 8, line length 120) and linted in CI.
- `multilang` extra declares `lingua-language-detector` (used by `dettext`).
### Removed
- Dead code: the round-3 page `Composer` and bank builders of `multilang`, unused helpers of `layout`, `hand`, `fillpic`.
- `tools/hwgen.py` (pre-hwsynth copy of the handwriting generator), `tools/prepreg.py` (imported a removed module),
  `tools/kaggle_hwdet*` (detector training lives in hwdet).

## 0.2.1 (2026-11-15)
- `dspages.pagegen` (handwritten pages, filled forms) moved here from hwsynth.

## 0.2.0 (2026-11-15)
- `dspages.multilang` moved here from hwdet; Apache-2.0.
