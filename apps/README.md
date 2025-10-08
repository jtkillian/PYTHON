# Applications

This monorepo groups deployable or exploratory applications under `apps/`:

| App | Description |
| --- | --- |
| `panel/` | Panel demos and dashboards. |
| `streamlit/` | Streamlit demo gallery. |
| `mu/` | Coursework notebooks and lab materials. |
| `aegis/`, `atlas/`, `citadel/`, `echo/`, `insight/`, `oracle/`, `prism/`, `pulse/` | Reserved for future projects. |

Each app can define its own `pyproject.toml` for dependencies and tooling. Install an app in editable mode with:

```powershell
pip install -e apps/<app-name>
```
