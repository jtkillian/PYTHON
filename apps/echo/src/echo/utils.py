"""Utility helpers shared across the Echo application."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np


try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover - torch optional
    torch = None  # type: ignore


def log_info(message: str) -> None:
    """Simple logging helper."""

    print(f"[echo] {message}")


def db_clip(db: np.ndarray, floor: float, ceiling: float) -> np.ndarray:
    return np.clip(db, floor, ceiling)


def ensure_numpy(value: np.ndarray | Iterable[float]) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def downsample_waveform(
    waveform: np.ndarray,
    samplerate: int,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a decimated waveform and matching time axis.

    The Matplotlib waveform pane only needs a limited number of points to look
    smooth. Rendering millions of samples introduces large memory churn during
    interactive updates and MP4 rendering.  This helper keeps the displayed
    waveform lightweight while preserving the time alignment with the original
    audio.
    """

    waveform = np.asarray(waveform, dtype=np.float32)
    total_samples = waveform.size
    if total_samples == 0:
        empty = np.array([], dtype=np.float32)
        return empty, empty

    if total_samples <= max_points:
        sample_idx = np.arange(total_samples, dtype=np.float32)
    else:
        sample_idx = np.linspace(0, total_samples - 1, num=max_points, dtype=np.float32)
        waveform = np.interp(sample_idx, np.arange(total_samples, dtype=np.float32), waveform)

    times = sample_idx / float(samplerate)
    duration = (total_samples - 1) / float(samplerate)
    if times.size:
        times[-1] = max(times[-1], duration)
    return waveform.astype(np.float32, copy=False), times.astype(np.float32, copy=False)


def time_to_text(seconds: float) -> str:
    minutes, sec = divmod(max(0.0, float(seconds)), 60.0)
    hours, minutes = divmod(minutes, 60.0)
    if hours:
        return f"{int(hours):02d}:{int(minutes):02d}:{sec:05.2f}"
    return f"{int(minutes):02d}:{sec:05.2f}"


def device_summary() -> str:
    if torch is None:
        return "Torch unavailable"
    if torch.cuda.is_available():  # pragma: no cover - GPU specific
        return f"CUDA:{torch.cuda.get_device_name(0)}"
    return "CPU"


def hash_config(file_path: Path, config_data: dict[str, object]) -> str:
    """Create a deterministic hash for cache keys."""

    stat = file_path.stat()
    payload = {
        "file": str(file_path.resolve()),
        "mtime": stat.st_mtime_ns,
        "size": stat.st_size,
        "config": config_data,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def hud_lines(
    gate_width: float,
    gate_rate: float,
    fidelity: str,
    fine_step: float,
) -> str:
    lines = [
        "Controls:",
        "←/→ : ±5s",
        f",/. : ±{fine_step:.1f}s",
        f"[/] : gate {gate_width:.1f}s",
        f"Q/E : rate {gate_rate:.2f}×",
        f"P   : {fidelity}",
        "Click fine box to toggle",
    ]
    return "\n".join(lines)
