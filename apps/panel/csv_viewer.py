import io
from typing import cast

import hvplot.pandas  # noqa: F401  # registers the hvplot accessor
import pandas as pd
import panel as pn


pn.extension()
uploader = pn.widgets.FileInput(accept=".csv")
plot_kind = pn.widgets.Select(name="Chart", options=["line", "scatter", "hist"])


def make_view() -> object:
    raw_value = cast(bytes | None, uploader.value)
    if not raw_value:
        return pn.pane.Markdown("### Upload a CSV to begin")

    try:
        df = pd.read_csv(io.BytesIO(raw_value))
    except Exception as exc:  # pragma: no cover - surfaced in UI
        return pn.pane.Alert(f"Could not parse CSV: {exc}", alert_type="danger")

    numeric_cols = df.select_dtypes("number").columns.tolist()

    if plot_kind.value == "line":
        if not numeric_cols:
            return pn.pane.Markdown("Need at least one numeric column for line charts.")
        return df[numeric_cols].hvplot.line()

    if plot_kind.value == "scatter":
        if len(numeric_cols) >= 2:
            return df.hvplot.scatter(x=numeric_cols[0], y=numeric_cols[1])
        return pn.pane.Markdown("Need at least two numeric columns for scatter plots.")

    if numeric_cols:
        return df[numeric_cols[0]].hvplot.hist()
    return pn.pane.Markdown("Need at least one numeric column for a histogram.")


view = pn.bind(make_view)
pn.Column("# Panel CSV Viewer", uploader, plot_kind, view).servable()

# python -m panel serve apps\panel\demo_quick.py --show --autoreload
