"""MP4 export utilities."""

from __future__ import annotations

from pathlib import Path

import imageio
import numpy as np

from .ui_fourpane import FourPaneUI
from .utils import log_info


def render_mp4(ui: FourPaneUI, out_path: Path, fps: int = 30, codec: str = "h264", fast: bool = False) -> None:
    """Render an MP4 by sampling frames from the provided UI."""

    out_path = Path(out_path)
    if out_path.suffix.lower() == ".mo4":
        log_info(f"Correcting output suffix from .mo4 to .mp4 for {out_path.name}")
        out_path = out_path.with_suffix(".mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fast:
        ui.fidelity = "Performance"
    else:
        ui.fidelity = "Hi-Fi"

    frame_total = max(1, int(np.ceil(ui.state.duration * fps)))
    log_info(f"Rendering MP4 → {out_path} @ {fps} FPS ({ui.fidelity}, {frame_total} frames)")
    writer = imageio.get_writer(out_path, fps=fps, codec=codec)
    try:
        for index in range(frame_total):
            t = min(ui.state.duration, index / fps)
            frame = ui.render_frame(t)
            writer.append_data(frame)
            if frame_total >= 20:
                step = max(1, frame_total // 10)
                if (index + 1) % step == 0 or index == frame_total - 1:
                    log_info(f"Rendered {index + 1}/{frame_total} frames")
    finally:
        writer.close()
    log_info("MP4 render complete")
