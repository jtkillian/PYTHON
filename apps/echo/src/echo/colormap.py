"""Colormap presets for the Echo UI."""

from __future__ import annotations

from matplotlib import cm

DEFAULT_COLORMAP = cm.get_cmap("magma")
HIGH_CONTRAST = cm.get_cmap("viridis")

__all__ = ["DEFAULT_COLORMAP", "HIGH_CONTRAST"]
