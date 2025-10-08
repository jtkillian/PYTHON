"""Audio loading utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np


try:  # pragma: no cover - optional dependency
    import soundfile as sf
except ImportError:  # pragma: no cover - fallback when soundfile missing
    sf = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import librosa
except ImportError:  # pragma: no cover - fallback when librosa missing
    librosa = None  # type: ignore

from .config import EchoConfig


def load_audio(path: Path, config: EchoConfig) -> tuple[np.ndarray, int]:
    """Load audio from *path* using the supplied configuration.

    Parameters
    ----------
    path:
        WAV file path.
    config:
        Application configuration controlling sample rate and mono fold-down.
    """

    path = Path(path)
    if not path.exists():  # pragma: no cover - defensive
        raise FileNotFoundError(path)

    target_sr = config.sr
    mono = config.mono

    if librosa is not None:
        data, sr = librosa.load(path, sr=target_sr, mono=mono)
        if data.ndim == 1:
            return data.astype(np.float32), int(sr)
        return data.astype(np.float32), int(sr)

    # Librosa missing; use soundfile directly
    if sf is None:
        raise RuntimeError("soundfile is required when librosa is unavailable")
    data, sr = sf.read(path, always_2d=True)
    if target_sr is not None and target_sr != sr:
        raise RuntimeError("Resampling requires librosa; install it to change --sr")
    data = data.mean(axis=1) if mono else data.T
    return data.astype(np.float32), int(sr)
