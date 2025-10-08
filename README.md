# PYTHON

Monorepo for notebooks, reusable Python packages, dashboards, coursework, and documentation.

- Local dev on RTX 4090 (CUDA).
- Dev Container / Codespaces for reproducible environments.
- CI: ruff · black · mypy · pytest (`--testmon`) · TruffleHog · CodeQL.
- Deep review every 2h (skips when no recent changes).

## Repository layout

```
apps/        # Deployable/demo apps (Panel, Streamlit, MU coursework, future projects)
packages/    # Shared Python packages (e.g., sandbox, jdw-core)
docs/        # LaTeX/Quarto documentation
data/        # Versioned datasets and fixtures
notebooks/   # Exploratory notebooks not tied to an app
reports/     # Generated figures and reports
tests/       # Shared test suite targeting packages
config files # Ruff, MyPy, pytest, coverage, etc. live at the repo root
```

Install an app or package in editable mode:

```powershell
pip install -e apps\\<name>
# or
pip install -e packages\\<name>
```

## Quickstart

```powershell
conda activate py310
pip install -e packages\\sandbox  # core utilities
jupyter lab
```
