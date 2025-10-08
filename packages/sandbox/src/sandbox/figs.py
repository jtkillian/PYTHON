from __future__ import annotations

import contextlib
import glob
import pathlib
from typing import Any

import matplotlib.pyplot as plt


# Lazy imports for Plotly
try:
    import plotly.graph_objects as go
    from plotly.io import write_image
except Exception:
    go = None
    write_image = None  # type: ignore[assignment]


def _fig_dir_for(notebook_name: str) -> pathlib.Path:
    """
    notebook_name: e.g. "calc/01_tangent" or "00_welcome"
    Output:        quarto/build/<subdirs>/<name>/figures
    """
    nb = pathlib.Path(notebook_name)
    sub = nb.parent
    stem = nb.stem
    if str(sub) == ".":
        outdir = pathlib.Path("quarto") / "build" / stem / "figures"
    else:
        outdir = pathlib.Path("quarto") / "build" / sub / stem / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def _next_number_by_scan(notebook_name: str) -> int:
    """Find the next figure number by scanning existing files for this notebook."""
    outdir = _fig_dir_for(notebook_name)
    stem = pathlib.Path(notebook_name).stem
    # Match "stem figN.pdf" or "stem figN.png"
    matches = glob.glob(str(outdir / f"{stem} fig*.pdf")) + glob.glob(
        str(outdir / f"{stem} fig*.png")
    )
    nums = []
    for m in matches:
        name = pathlib.Path(m).stem  # "stem figN"
        parts = name.replace(stem, "").strip().split()
        if len(parts) == 2 and parts[0] == "fig":
            with contextlib.suppress(ValueError):
                nums.append(int(parts[1]))
        elif parts and parts[0].startswith("fig"):
            with contextlib.suppress(ValueError):
                nums.append(int(parts[0][3:]))
    return (max(nums) + 1) if nums else 1


def _base_path(notebook_name: str, label: str | None) -> pathlib.Path:
    outdir = _fig_dir_for(notebook_name)
    stem = pathlib.Path(notebook_name).stem
    if label:
        return outdir / f"{stem} {label}"
    else:
        n = _next_number_by_scan(notebook_name)
        return outdir / f"{stem} fig{n}"


def save_best(
    notebook_name: str,
    fig: Any | None = None,
    label: str | None = None,
    plotly_png_scale: float = 3.0,
) -> pathlib.Path:
    """
    Export a static snapshot:
      - Plotly with 3D traces -> high-DPI PNG
      - Plotly without 3D     -> PDF
      - Matplotlib            -> PDF
    Uses the specific object you pass (Figure, FigureWidget, Axes, or ipympl canvas).
    Falls back to pyplot's current figure only if `fig` is None.
    """

    base = _base_path(notebook_name, label)  # keep your existing helper

    # ------------ Plotly path (robust) ------------
    is_plotly = False
    plotly_types: tuple[type[Any], ...] = ()
    try:
        import plotly.graph_objects as go

        plotly_types = (go.Figure,)
        figure_widget = getattr(go, "FigureWidget", None)
        if isinstance(figure_widget, type):
            plotly_types = plotly_types + (figure_widget,)
        if fig is not None:
            is_plotly = isinstance(fig, plotly_types)
    except Exception:
        go = None  # not installed / unavailable

    if fig is not None and is_plotly:
        # Detect 3D-ish traces
        def _has_3d(f) -> bool:
            data = getattr(f, "data", ()) or ()
            for tr in data:
                t = getattr(tr, "type", "") or ""
                t = t.lower()
                if (
                    t in {"surface", "mesh3d", "isosurface", "volume", "cone", "streamtube"}
                    or "3d" in t
                ):
                    return True
            return False

        has_3d = _has_3d(fig)
        out = base.with_suffix(".png" if has_3d else ".pdf")
        # Prefer the object's own writer; fallback to pio.write_image
        try:
            fig.write_image(str(out), scale=(plotly_png_scale if has_3d else 2.0))
        except Exception:
            import plotly.io as pio

            pio.write_image(fig, str(out), scale=(plotly_png_scale if has_3d else 2.0))
        return out

    # ------------ Matplotlib path (object-oriented) ------------
    # Normalize fig-like things (Figure, Axes, ipympl canvas) to a Figure
    mpl_fig: Any | None = None
    if fig is not None:
        try:
            # If it's already a Figure
            if hasattr(fig, "savefig"):
                mpl_fig = fig
            # ipympl canvas or Artist with .figure
            elif hasattr(fig, "figure") and hasattr(fig.figure, "savefig"):
                mpl_fig = fig.figure
            # Axes -> parent Figure
            elif hasattr(fig, "get_figure"):
                maybe = fig.get_figure()
                if hasattr(maybe, "savefig"):
                    mpl_fig = maybe
        except Exception:
            mpl_fig = None

    if mpl_fig is not None:
        out = base.with_suffix(".pdf")
        mpl_fig.savefig(out, format="pdf", bbox_inches="tight")
        return out

    # Fallback: save pyplot's CURRENT figure only if nothing was passed

    out = base.with_suffix(".pdf")
    plt.gcf().savefig(out, format="pdf", bbox_inches="tight")
    return out


def save_buttons(notebook_name: str, fig: Any | None = None) -> None:
    """
    Single button: 'Save Best (PDF/PNG)' with tactile feedback and no message cutoff.
    - If fig is Plotly, uses it (captures current camera for FigureWidget)
    - Else saves current Matplotlib figure
    """
    import threading
    import time
    import traceback

    import ipywidgets as w
    from IPython.display import display

    # --- lightweight CSS for tactile feel (press + gap) ---
    css = w.HTML(
        """
    <style>
      .sb-row { gap: 8px; }
      .sb-btn  { transition: transform 60ms ease, filter 120ms ease; }
      .sb-btn:active { transform: translateY(1px) scale(0.995); filter: brightness(0.96); }
      .sb-msg code { white-space: pre-wrap; word-break: break-all; }
    </style>
    """
    )

    # --- button: auto width, but never too small; fixed height; no flex-grow ---
    btn = w.Button(
        description="Save Best (PDF/PNG)",
        tooltip="Export optimal snapshot",
        layout=w.Layout(
            width="auto",  # expand to fit text
            min_width="200px",  # ensure it's wide enough
            height="36px",
            flex="0 0 auto",  # don't steal space from the message
            padding="0 12px",
        ),
        button_style="primary",
    )
    btn.add_class("sb-btn")

    # message area on its own line to avoid truncation
    msg = w.HTML("", layout=w.Layout(margin="4px 0 0 0"))
    msg.add_class("sb-msg")

    # assemble: button row above, message below
    row = w.HBox([btn], layout=w.Layout(justify_content="flex-start"))
    row.add_class("sb-row")
    box = w.VBox([css, row, msg])

    def _reset_btn(after=0.6, *, desc="Save Best (PDF/PNG)", style="primary") -> None:
        def _do_reset() -> None:
            time.sleep(after)
            btn.button_style = style
            btn.description = desc
            btn.disabled = False

        threading.Thread(target=_do_reset, daemon=True).start()

    def _on_click(_) -> None:
        btn.disabled = True
        btn.button_style = "warning"
        btn.description = "Saving…"
        try:
            p = save_best(notebook_name, fig=fig, label=None)
            msg.value = f"✅ Saved: <code>{p}</code>"
            btn.button_style = "success"
            btn.description = "Saved ✓"
            _reset_btn()
        except Exception as e:
            # Show error, then go back to ready state
            tb = traceback.format_exc()
            msg.value = (
                f"❌ <b>Error:</b> {e}<br>"
                "<details><summary>Traceback</summary>"
                f"<pre>{tb}</pre></details>"
            )
            btn.button_style = "danger"
            btn.description = "Failed"
            _reset_btn(after=1.0)

    btn.on_click(_on_click)
    display(box)
