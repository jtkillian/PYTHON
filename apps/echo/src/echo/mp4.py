"""MP4 export utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import imageio
import numpy as np

from .ui_fourpane import FourPaneUI
from .utils import log_info


def iter_frames(duration: float, fps: int) -> Iterable[float]:
    frame_count = max(1, int(np.ceil(duration * fps)))
    for i in range(frame_count):
        yield min(duration, i / fps)


def render_mp4(ui: FourPaneUI, out_path: Path, fps: int = 30, codec: str = "h264", fast: bool = False) -> None:
    """Render an MP4 by sampling frames from the provided UI."""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fast:
        ui.fidelity = "Performance"
    else:
        ui.fidelity = "Hi-Fi"

    log_info(f"Rendering MP4 → {out_path} @ {fps} FPS ({ui.fidelity})")
    writer = imageio.get_writer(out_path, fps=fps, codec=codec)
    try:
        for t in iter_frames(ui.state.duration, fps):
            frame = ui.render_frame(t)
            writer.append_data(frame)
    finally:
        writer.close()
    log_info("MP4 render complete")
