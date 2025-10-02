# Copilot Instructions for the PYTHON repo

## Architecture snapshot
- Package code lives in `src/sandbox/`; expose utilities via `src/sandbox/__init__.py` when you want them importable as `sandbox.<name>`.
- Keep math helpers pure and side-effect free so they stay notebook-friendly and easy to test.
- Tests mirror modules under `tests/` (e.g., `tests/test_mathutils.py` covers `src/sandbox/mathutils.py`). Add new test files alongside the feature you extend.
- Notebooks under `notebooks/calc/` are exploratory; keep production-ready logic in `src/` and import it from notebooks instead of redefining.

## Tooling & quality gates
- Target Python 3.10; local dev typically uses `conda activate py310` before installing.
- Install dependencies with `pip install -e .` so edits in `src/` are immediately importable.
- Respect formatter defaults: Black + Ruff both use 100-character lines and double quotes (see `pyproject.toml`).
- CI enforces `ruff check`, `black --check`, `mypy`, and `pytest -q`. Run them locally in that order to match pipeline failures.
- Pytest runs with `--testmon` to re-test only impacted files; you can force a clean slate by deleting `.testmondata` if selective runs get confused.

## Coding conventions
- Prefer small, pure functions with explicit type hints (`mypy` runs in loose mode but missing annotations on public APIs will trigger Ruff `ANN` checks except for `self`/`cls`).
- Place shared constants or helpers in existing modules before adding new top-level packages; keep the `sandbox` namespace flat until there's real structure.
- Use type-stable math operations—tests expect deterministic floats, so avoid global state or random seeds.
- When adding notebook utilities, expose a thin wrapper in `src/sandbox` and keep notebook cells for demonstrations only.

## Workflow tips
- After adding a module, export it in `src/sandbox/__init__.py` so `from sandbox import <symbol>` works inside notebooks and tests.
- For new CLI or script entrypoints, stage them under `src/` and add a smoke test in `tests/` to keep coverage meaningful.
- Data files in `data/` are treated as read-only fixtures; prefer synthetic datasets inside tests over touching these files.
- Update `README.md` if you meaningfully change the high-level purpose or tooling (keeps Codespaces/RTX notes accurate).

## CI nuances
- Pull requests from forks skip the autofix loop in `.github/workflows/ci.yml`; keep diffs minimal so checks stay green without it.
- If CI fails on formatting or linting, run the same commands locally and commit the fixes—CI uses the latest `pip install --upgrade pip` so be ready for tooling updates.
