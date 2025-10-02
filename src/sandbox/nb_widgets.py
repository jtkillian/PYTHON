from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np

# Optional SymPy import (works even if SymPy isn't installed)
try:
    import sympy as sp  # type: ignore
except Exception:  # SymPy missing or not importable
    sp = None  # type: ignore

if TYPE_CHECKING:
    import ipywidgets as ipywidgets
    import sympy as sympy_types  # type: ignore

    SymExprType = sympy_types.Expr
else:
    SymExprType = object  # fallback type

SymOrCallable: TypeAlias = SymExprType | Callable[[np.ndarray], np.ndarray]


# ---------- Ensure "<repo>/src" on sys.path so imports work from any notebook ----------
def _ensure_src_on_sys_path() -> None:
    here = Path.cwd()
    for p in (here, *here.parents):
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            src = p / "src"
            ssrc = str(src)
            if ssrc not in sys.path:
                sys.path.insert(0, ssrc)
            return


_ensure_src_on_sys_path()


# ---------- Notebook setup (safe in/without IPython) ----------
def notebook_setup() -> None:
    """Set up nice defaults: matplotlib (widget if available), Plotly inline, SymPy pretty."""
    # Matplotlib: prefer interactive widget backend if ipympl is present
    try:
        import ipympl  # noqa: F401
        from IPython import get_ipython  # type: ignore

        shell = get_ipython()
        if shell is not None:
            shell.run_line_magic("matplotlib", "widget")
    except Exception:
        # fallback to static inline
        try:
            from IPython import get_ipython  # type: ignore

            shell = get_ipython()
            if shell is not None:
                shell.run_line_magic("matplotlib", "inline")
        except Exception:
            pass

    # Plotly inline
    try:
        import plotly.io as pio  # type: ignore

        if not pio.renderers.default:
            pio.renderers.default = "vscode"  # or "notebook_connected"
    except Exception:
        pass

    # SymPy pretty
    try:
        import sympy as _sp  # type: ignore

        _sp.init_printing(use_latex="mathjax")
    except Exception:
        pass

    print("Notebook ready: matplotlib (widget/inline), plotly inline, sympy pretty printing.")


# ---------- Utilities: coerce to numeric f and f' ----------
def _coerce_univariate(sym_or_callable: SymOrCallable) -> tuple[
    Callable[[np.ndarray], np.ndarray],
    Callable[[np.ndarray], np.ndarray],
]:
    """
    Accept a SymPy expression in x OR a Python callable f(x).
    Return (f, fprime) as numpy-callables.
    """
    if sp is not None and isinstance(sym_or_callable, sp.Expr):  # type: ignore[arg-type]
        xs = list(sym_or_callable.free_symbols)  # type: ignore[attr-defined]
        if len(xs) != 1:
            raise ValueError("SymPy expression must have exactly one symbol (e.g., x).")
        sym_x = xs[0]
        f_expr = sym_or_callable  # type: ignore[assignment]
        fprime_expr = sp.diff(f_expr, sym_x)  # type: ignore[misc]

    f_sym = sp.lambdify(sym_x, f_expr, modules="numpy")  # type: ignore[misc]
    fprime_sym = sp.lambdify(sym_x, fprime_expr, modules="numpy")  # type: ignore[misc]
    return f_sym, fprime_sym

    if not callable(sym_or_callable):
        raise TypeError("Expected a SymPy expression or a Python callable f(x).")

    def f_callable(x: np.ndarray) -> np.ndarray:
        return np.asarray(sym_or_callable(x), dtype=float)  # type: ignore[misc]

    def fprime_callable(x: np.ndarray) -> np.ndarray:
        # central difference with adaptive step
        h = 1e-5 * (1.0 + np.abs(x))
        return (f_callable(x + h) - f_callable(x - h)) / (2.0 * h)

    return f_callable, fprime_callable


# ---------- Tangent line widget (single output, smooth updates) ----------
def tangent_widget(
    func: SymOrCallable,
    a_init: float = 1.0,
    xmin: float = -4.0,
    xmax: float = 4.0,
    step: float = 0.05,
    n: int = 400,
) -> ipywidgets.Widget:
    """
    Interactive tangent visualizer for y = f(x).
    Returns a single VBox (slider + plot) and NEVER calls display() internally.
    Uses plt.ioff() so Matplotlib cannot auto-render a second figure.
    Works with both ipympl (widget) and inline backends.
    """
    import ipywidgets as w
    import matplotlib
    import matplotlib.pyplot as plt
    from IPython.display import display  # type: ignore

    f, fprime = _coerce_univariate(func)

    # Domain & samples
    x0: float = float(xmin)
    x1: float = float(xmax)
    nn: int = int(n)
    x = np.linspace(x0, x1, num=nn)
    y = f(x)

    # Create figure with auto-render disabled
    with plt.ioff():
        fig, ax = plt.subplots(figsize=(6, 4))
        (line_f,) = ax.plot(x, y, label="f(x)")
        (line_tan,) = ax.plot([], [], "--", label="Tangent", linewidth=2.0)
        (pt,) = ax.plot([], [], "o", color="black")
        ax.set_xlim(x0, x1)
        yspan = float(np.nanmax(y) - np.nanmin(y))
        ypad = 0.1 * (yspan + 1e-9)
        ax.set_ylim(float(np.nanmin(y) - ypad), float(np.nanmax(y) + ypad))
        ax.grid(True)
        ax.legend()

    # Decide rendering body depending on backend
    backend = matplotlib.get_backend().lower()
    is_widget = "widget" in backend  # ipympl (live canvas)

    if is_widget:
        body = fig.canvas  # live canvas widget
    else:
        out = w.Output()
        # Render once into the Output (still inside notebook, but not auto)
        with out:
            display(fig)
        body = out  # we'll re-populate on updates

    def update(a: float) -> None:
        aa = float(a)
        fa = float(np.asarray(f(np.array(aa))))
        fpa = float(np.asarray(fprime(np.array(aa))))
        y_tan = fa + fpa * (x - aa)
        line_tan.set_data(x, y_tan)
        pt.set_data([aa], [fa])
        ax.set_title(f"Tangent at a={aa:.2f}")
        if is_widget:
            fig.canvas.draw_idle()
        else:
            # Re-draw into the Output container
            with body:  # type: ignore[assignment]
                body.clear_output(wait=True)  # type: ignore[attr-defined]
                display(fig)

    slider = w.FloatSlider(
        value=float(a_init),
        min=x0 + 1e-6,
        max=x1 - 1e-6,
        step=float(step),
        description="a",
        continuous_update=True,
    )
    slider.observe(lambda _chg: update(float(slider.value)), names="value")
    update(float(a_init))  # initial draw

    ui = w.VBox([slider, body])
    return ui  # return (no display) so the cell renders exactly once


# ---------- 3D surface widget (FigureWidget live; static fallback) ----------
def surface3d_widget(
    func: Callable[..., np.ndarray],
    x_range: tuple[float, float] = (-np.pi, np.pi),
    y_range: tuple[float, float] = (-np.pi, np.pi),
    grid: int = 100,
    params: dict[str, tuple[float, float, float, float]] | None = None,
) -> None:
    """
    Interactive 3D surface for z = func(x, y, **params).
    Uses Plotly FigureWidget for live updates (requires ipywidgets + anywidget).
    Falls back to a static Figure if FigureWidget is unavailable.
    """
    import ipywidgets as w
    import plotly.io as pio  # type: ignore
    from IPython.display import display  # type: ignore

    # Domain/grid
    x0: float = float(x_range[0])
    x1: float = float(x_range[1])
    y0: float = float(y_range[0])
    y1: float = float(y_range[1])
    nn: int = int(grid)
    x = np.linspace(x0, x1, num=nn)
    y = np.linspace(y0, y1, num=nn)
    x_grid, y_grid = np.meshgrid(x, y)

    params = params or {}
    current: dict[str, float] = {k: float(v[0]) for k, v in params.items()}

    try:
        import plotly.graph_objects as go  # type: ignore

        if not pio.renderers.default:
            pio.renderers.default = "vscode"

        # Confirm FigureWidget availability (requires anywidget)
        _ = go.FigureWidget  # type: ignore[attr-defined]

        z_grid = func(x_grid, y_grid, **current)
        fig = go.FigureWidget(data=[go.Surface(z=z_grid, x=x_grid, y=y_grid)])
        fig.update_layout(
            width=720, height=520, margin=dict(l=0, r=0, t=30, b=0), title=_surface_title(current)
        )

        sliders: list[w.FloatSlider] = []
        for name, (val, vmin, vmax, step) in list(params.items())[:2]:
            sliders.append(
                w.FloatSlider(
                    value=float(val),
                    min=float(vmin),
                    max=float(vmax),
                    step=float(step),
                    description=name,
                    continuous_update=True,
                )
            )

        def on_change(_chg: dict[str, Any]) -> None:
            for s in sliders:
                current[s.description] = float(s.value)
            updated_z = func(x_grid, y_grid, **current)
            with fig.batch_update():
                fig.data[0].z = updated_z
                fig.update_layout(title=_surface_title(current))

        for s in sliders:
            s.observe(on_change, names="value")

        box = w.VBox([*sliders, fig]) if sliders else fig
        display(box)
        return None

    except Exception:
        # Fallback: static Figure
        import plotly.graph_objects as go  # type: ignore

        if not pio.renderers.default:
            pio.renderers.default = "vscode"

        z_grid = func(x_grid, y_grid, **current)
        fig = go.Figure(data=[go.Surface(z=z_grid, x=x_grid, y=y_grid)])
        fig.update_layout(
            width=720,
            height=520,
            margin=dict(l=0, r=0, t=30, b=0),
            title=_surface_title(current) + "  (static fallback)",
        )
        fig.show()
        print("Note: install anywidget for live updates:  pip install anywidget")
        return None


def _surface_title(current: dict[str, Any]) -> str:
    if not current:
        return "z = f(x, y)"
    parts = [f"{k}={float(v):.2f}" for k, v in current.items()]
    return "z = f(x, y)  |  " + ", ".join(parts)
