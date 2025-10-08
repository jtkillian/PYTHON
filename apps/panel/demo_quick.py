# apps/panel/demo_quick.py
import hvplot.pandas  # noqa: F401
import pandas as pd
import panel as pn


pn.extension()  # default Bokeh backend

df = pd.DataFrame({"t": range(100), "v": [i**0.5 for i in range(100)]})
w = pn.widgets.IntSlider(name="Window", start=5, end=30, value=10)


def _plot(win: int) -> object:
    rolling = df.rolling(win, min_periods=1).mean()
    return rolling.hvplot.line(x="t", y="v", title=f"Rolling mean (w={win})")


view = pn.bind(_plot, w)  # cleaner than @pn.depends for this case
pn.Column("# Panel Rolling Mean", w, view).servable()

# python -m panel serve apps\panel\demo_plotly.py --show --autoreload
