# PRISM Build & Operations Guide

## Installation

1. Ensure [uv](https://github.com/astral-sh/uv) is available in your environment.
2. Install required Python packages:

   ```bash
   uv pip install -r apps/prism/requirements.txt
   ```

3. Optionally install Playwright browsers if screenshots are desired:

   ```bash
   uv run playwright install chromium
   ```

4. Copy `.env.example` to `.env` inside `apps/prism/` and adjust configuration caps or defaults.

5. Initialize directories and the SQLite schema via PowerShell:

   ```powershell
   pwsh apps/prism/scripts/post_install.ps1
   ```

   The script creates `data/prism.db`, ensures `output/` and `data/artifacts/` exist, and verifies SQLite connectivity.

## Running Services

- **API**: `uv run uvicorn apps.prism.services.api.app:app --reload --port 8000`
- **UI**: `uv run streamlit run apps/prism/services/ui/app.py`

The Streamlit UI communicates with the FastAPI backend using the host/port specified in the environment variables (`API_URL`).

## Project Layout

- `services/api`: FastAPI application, pydantic schemas, orchestration pipeline, storage layer, collector runners, and normalizers.
- `services/ui`: Streamlit interface with reusable components for highlights, tables, graphs, and maps.
- `models`: placeholder folders for future ONNX models. Drop `model.onnx` into `models/face` to automatically enable face-matching toggles.
- `data`: SQLite database and saved raw artifacts (gitignored).
- `output`: Per-person report folders containing `Findings.md`, `Findings.json`, `graph.gexf`, `map.geojson`, and associated artifacts.

## Extending Collectors

Add new collectors under `services/api/runners/` and corresponding normalizers under `services/api/normalizers/`. Export the normalized schema using the helper dataclasses from `models.py`. Update the orchestrator in `app.py` to register the new module and add UI toggles in `services/ui/app.py`.

## Integration Hooks

`services/api/app.py` defines stubs for the forthcoming ORACLE, AEGIS, and CITADEL integrations. Implementations should maintain the same function signatures, returning awaitables where appropriate.

## Testing & Development Notes

- The backend uses SQLAlchemy with an async engine for compatibility with FastAPI.
- Configuration caps are loaded via Pydantic settings; environment overrides are respected.
- All collectors are optional and default to disabled heavy operations (screenshots, HTML archiving, paid APIs).
- Summaries are produced in both markdown and JSON to support downstream ingestion.

