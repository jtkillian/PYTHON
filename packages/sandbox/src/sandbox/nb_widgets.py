from __future__ import annotations

import contextlib
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import numpy as np


# Optional SymPy import (works even if SymPy isn't installed)
try:
    import sympy as sp  # type: ignore
except Exception:  # SymPy missing or not importable
    sp = None  # type: ignore

NumericFunc = Callable[[np.ndarray], np.ndarray]

if TYPE_CHECKING:
    from ipywidgets import Widget as IpyWidget
    from sympy import Expr as SymExpr

    SymOrCallable: TypeAlias = SymExpr | NumericFunc
else:
    IpyWidget = Any
    SymExpr = Any
    SymOrCallable = NumericFunc | Any


# ---------- Ensure "<repo>/src" on sys.path so imports work from any notebook ----------
def _ensure_repo_paths_on_sys_path() -> None:
    here = Path.cwd()
    for p in (here, *here.parents):
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            candidates: list[Path] = []

            packages_dir = p / "packages"
            if packages_dir.exists():
                for subdir in packages_dir.iterdir():
                    src_dir = subdir / "src"
                    if src_dir.exists():
                        candidates.append(src_dir)

            legacy_src = p / "src"
            if legacy_src.exists():
                candidates.append(legacy_src)

            for path in candidates:
                s_path = str(path)
                if s_path not in sys.path:
                    sys.path.insert(0, s_path)
            return


_ensure_repo_paths_on_sys_path()


# ---------- Notebook setup (safe in/without IPython) ----------
def notebook_setup() -> None:
    """
    Configure sensible defaults for notebooks in VS Code:
    - Plotly renderer set to 'vscode' (fallback to 'notebook_connected')
    - Enable ipywidgets extensions
    - (Optional) Matplotlib inline
    """
    try:
        from plotly.io import renderers

        try:
            renderers.default = "vscode"
        except Exception:
            renderers.default = "notebook_connected"
    except Exception:
        pass

    # Widgets: VS Code handles frontend; ensure ipywidgets is importable
    try:
        import ipywidgets as _  # noqa: F401
    except Exception:
        pass

    # Matplotlib inline
    with contextlib.suppress(Exception):
        from IPython import get_ipython  # type: ignore

        get_ipython().run_line_magic("matplotlib", "inline")  # type: ignore[name-defined]


# ---------- Utilities: coerce to numeric f and f' ----------
def _coerce_univariate(sym_or_callable: SymOrCallable) -> tuple[NumericFunc, NumericFunc]:
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
    func,
    a_init: float = 1.0,
    xmin: float = -4.0,
    xmax: float = 4.0,
    step: float = 0.05,
    n: int = 400,
    *,
    width: int = 720,  # <- NEW: fixed width in px
    height: int = 480,  # <- NEW: fixed height in px
    dpi: int = 100,  # <- NEW: used for matplotlib sizing in inline mode
):
    """
    Interactive tangent visualizer for y = f(x).
    Returns VBox([slider, body]) with a fixed pixel width/height (no autoscale).
    - ipympl backend: uses live canvas, sized via widget layout
    - inline backend: updates a single PNG Image widget (no flicker)
    """
    from io import BytesIO
    from typing import Any

    import ipywidgets as w
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    # Resolve f and f'
    try:
        f, fprime = _coerce_univariate(func)  # your helper
    except NameError:
        f, fprime = func, lambda x: (func(x + 1e-6) - func(x - 1e-6)) / (2e-6)

    # Domain & samples
    x0, x1 = float(xmin), float(xmax)
    nn = int(n)
    x = np.linspace(x0, x1, nn)
    y = f(x)

    # Create figure at the requested pixel size
    w_in, h_in = width / dpi, height / dpi
    with plt.ioff():
        fig, ax = plt.subplots(figsize=(w_in, h_in), dpi=dpi)
        (line_f,) = ax.plot(x, y, label="f(x)")
        (line_tan,) = ax.plot([], [], "--", label="Tangent", linewidth=2.0)
        (pt,) = ax.plot([], [], "o", color="black")
        ax.set_xlim(x0, x1)
        yspan = float(np.nanmax(y) - np.nanmin(y))
        ypad = 0.1 * (yspan + 1e-9)
        ax.set_ylim(float(np.nanmin(y) - ypad), float(np.nanmax(y) + ypad))
        ax.grid(True)
        ax.legend()

    # Backend switch
    is_widget = "widget" in matplotlib.get_backend().lower()

    if is_widget:
        # ipympl: live canvas — set widget layout to fixed size
        body = cast(Any, fig.canvas)
        body.layout.width = f"{width}px"
        body.layout.height = f"{height}px"
        img = None
    else:
        # inline: update a single PNG image (no reflow/scroll)
        img = w.Image(format="png", layout=w.Layout(width=f"{width}px"))
        body = img

        def _render_png() -> None:
            buf = BytesIO()
            # ensure the figure stays at requested size
            fig.set_size_inches(w_in, h_in, forward=True)
            fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
            buf.seek(0)
            img.value = buf.read()

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
            _render_png()

    # Slider sized to the same fixed width
    slider = w.FloatSlider(
        value=float(a_init),
        min=x0 + 1e-6,
        max=x1 - 1e-6,
        step=float(step),
        description="a",
        continuous_update=True,  # smooth only on ipympl
        layout=w.Layout(width="100%"),
    )
    slider.observe(lambda _chg: update(float(slider.value)), names="value")
    update(float(a_init))  # initial draw

    # Fixed-width container so nothing stretches
    ui = w.VBox([slider, body], layout=w.Layout(width=f"{width}px"))
    return ui


# ---------- 3D surface widget (FigureWidget live; static fallback) ----------
import math
from collections.abc import Callable
from typing import Any


def surface3d_widget_v2(
    func: str | Callable[..., np.ndarray],
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    grid: int = 100,
    params: dict[str, tuple[float, float, float, float]] | None = None,
    *,
    height: int = 520,
    width: int | None = None,
    autosize: bool = False,
    aspectmode: str = "cube",
    lock_yaw: bool = True,
    z_eye: float = 1.5,
):
    import ipywidgets as w
    import numpy as np
    import plotly.graph_objects as go

    # ---------------- 4 fun built-ins + defaults ----------------
    def peaks_like(X, Y):
        return (
            (3 * (1 - X) ** 2) * np.exp(-(X**2) - (Y + 1) ** 2)
            - 10 * (X / 5 - X**3 - Y**5) * np.exp(-(X**2) - Y**2)
            - (1 / 3) * np.exp(-((X + 1) ** 2) - Y**2)
        )

    def volcano_ripples(X, Y, k1=4.0, k2=8.0, decay=0.5):
        R = np.sqrt(X**2 + Y**2)
        return np.exp(-decay * R**2) * (np.cos(k1 * R) + 0.5 * np.cos(k2 * R))

    def swirl_ramp(X, Y, freq=5.0, ramp=1.0, fade=0.12):
        R = np.sqrt(X**2 + Y**2)
        TH = np.arctan2(Y, X)
        return ramp * TH + 0.20 * np.sin(freq * R) * np.exp(-fade * R**2)

    def monkey_saddle_wave(X, Y, scale=5.0, fade=0.10):
        R2 = X**2 + Y**2
        base = (X**3 - 3 * X * Y**2) / scale
        return base * np.exp(-fade * R2)

    builtins: dict[str, dict[str, Any]] = {
        "peaks": {
            "fn": peaks_like,
            "xyrange": (-3.0, 3.0, -3.0, 3.0),
            "params": {"gain": (1.0, 0.2, 2.0, 0.05)},
            "wrap": lambda z, p: p.get("gain", 1.0) * z,
            "title": "peaks",
        },
        "volcano": {
            "fn": volcano_ripples,
            "xyrange": (-3.0, 3.0, -3.0, 3.0),
            "params": {
                "k1": (4.0, 2.0, 8.0, 0.2),
                "k2": (8.0, 4.0, 16.0, 0.2),
                "decay": (0.5, 0.1, 1.2, 0.02),
                "gain": (1.0, 0.2, 2.0, 0.05),
            },
            "wrap": lambda z, p: p.get("gain", 1.0) * z,
            "title": "volcano",
        },
        "swirl": {
            "fn": swirl_ramp,
            "xyrange": (-3.0, 3.0, -3.0, 3.0),
            "params": {
                "freq": (5.0, 1.0, 12.0, 0.2),
                "ramp": (0.9, 0.0, 2.0, 0.05),
                "fade": (0.12, 0.02, 0.40, 0.01),
                "gain": (1.0, 0.2, 2.0, 0.05),
            },
            "wrap": lambda z, p: p.get("gain", 1.0) * z,
            "title": "swirl",
        },
        "monkey": {
            "fn": monkey_saddle_wave,
            "xyrange": (-2.5, 2.5, -2.5, 2.5),
            "params": {
                "scale": (6.0, 2.0, 12.0, 0.2),
                "fade": (0.12, 0.02, 0.40, 0.01),
                "gain": (1.0, 0.2, 2.0, 0.05),
            },
            "wrap": lambda z, p: p.get("gain", 1.0) * z,
            "title": "monkey",
        },
    }

    # Resolve function + defaults
    title_suffix = ""
    if isinstance(func, str):
        key = func.lower().strip()
        if key not in builtins:
            raise ValueError(f"Unknown built-in '{func}'. Use one of: {list(builtins.keys())}")
        f = builtins[key]["fn"]
        x0d, x1d, y0d, y1d = builtins[key]["xyrange"]
        if x_range is None:
            x_range = (x0d, x1d)
        if y_range is None:
            y_range = (y0d, y1d)
        if params is None:
            params = dict(builtins[key]["params"])
        wrap = builtins[key]["wrap"]
        title_suffix = f" · {builtins[key]['title']}"
    else:
        f = func
        if x_range is None:
            x_range = (-math.pi, math.pi)
        if y_range is None:
            y_range = (-math.pi, math.pi)
        if params is None:
            params = {"gain": (1.0, 0.2, 2.0, 0.05)}

        def wrap(z, p):
            return p.get("gain", 1.0) * z

    # Grid
    x0, x1 = map(float, x_range)
    y0, y1 = map(float, y_range)
    nn = int(grid)
    x = np.linspace(x0, x1, nn)
    y = np.linspace(y0, y1, nn)
    xg, yg = np.meshgrid(x, y)

    # Params -> state
    current = {k: float(v[0]) for k, v in params.items()}
    # Compute initial Z
    args0 = {k: current[k] for k in params if k != "gain"}
    z0 = wrap(f(xg, yg, **args0), current)

    # FigureWidget
    fig = go.FigureWidget([go.Surface(z=z0, x=xg, y=yg, colorscale="Viridis")])
    fig.update_layout(
        height=height,
        width=width,
        autosize=autosize,
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(
            aspectmode=aspectmode,
            camera=dict(up=dict(x=0, y=0, z=1), eye=dict(x=1.5, y=1.5, z=float(z_eye))),
            dragmode="turntable" if lock_yaw else "orbit",
        ),
        title=_surface_title(current) + title_suffix,
        uirevision="keep-camera",
    )

    # Sliders for ALL params
    sliders: list[w.FloatSlider] = []
    for name, (val, vmin, vmax, step) in params.items():
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

    def on_change(_chg) -> None:
        for s in sliders:
            current[s.description] = float(s.value)
        args = {k: current[k] for k in params if k != "gain"}
        z = wrap(f(xg, yg, **args), current)
        with fig.batch_update():
            fig.data[0].z = z
            fig.update_layout(title=_surface_title(current) + title_suffix)

    for s in sliders:
        s.observe(on_change, names="value")

    return w.VBox([*sliders, fig]) if sliders else w.VBox([fig])


# Helper used in the title
def _surface_title(current: dict[str, float]) -> str:
    if not current:
        return "z = f(x, y)"
    parts = [f"{k}={float(v):.2f}" for k, v in current.items()]
    return "z = f(x, y)  |  " + ", ".join(parts)
