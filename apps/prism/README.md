# PRISM OSINT Platform

PRISM is a modular, free-first OSINT collection and analysis stack that ships with a FastAPI backend and a Streamlit-powered digest UI. The system focuses on turning disparate open-source results into actionable human-friendly digests and graphs while remaining extendable for future integrations with AEGIS, ORACLE, and CITADEL.

## Features

- FastAPI orchestration API with modular collectors
- Streamlit UI that surfaces top highlights first
- SQLite storage with normalized schema for persons, artifacts, highlights, and provenance
- Free collectors: Sherlock, PhoneInfoga, Wayback Machine, and lightweight web search
- Automatic generation of per-person report folders containing markdown and JSON digests, graph exports, and saved artifacts
- Model-ready folder structure for optional ONNX face matching models
- Toggle-driven runs with sensible defaults for heavy features disabled

## Getting Started

1. Install dependencies with `uv pip install -r apps/prism/requirements.txt`.
2. Copy `apps/prism/.env.example` to `.env` and adjust limits or feature defaults as needed.
3. Execute the post-install script to initialize directories and the SQLite database: `pwsh apps/prism/scripts/post_install.ps1`.
4. Launch the FastAPI service: `uv run uvicorn apps.prism.services.api.app:app --reload --port 8000`.
5. Launch the Streamlit UI: `uv run streamlit run apps/prism/services/ui/app.py`.
6. Submit a person of interest via the UI, choose the modules to run, and collect results.

See `INSTRUCTIONS.md` for architecture details and extension guidance.
