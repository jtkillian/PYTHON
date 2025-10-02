# PYTHON

Public mono-repo for interactive notebooks (Calc & tutoring), reusable Python modules, tests, and papers (LaTeX).

- Local dev on RTX 4090 (CUDA).
- Dev Container / Codespaces for pristine, shareable environments.
- CI: ruff + black + mypy + pytest (testmon; changed-only) • TruffleHog (secrets) • CodeQL (public).
- Deep review every 2h (skips if no recent changes).

## Quickstart
```bash
conda activate py310
jupyter lab
