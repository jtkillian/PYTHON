"""Spectrogram computation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover - fallback
    torch = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import librosa
except Exception:  # pragma: no cover - fallback
    librosa = None  # type: ignore

from .config import EchoConfig
from .utils import db_clip, ensure_numpy, log_info


class SpectrogramResult(Dict[str, np.ndarray]):
    """Dictionary subclass storing spectrogram data."""


def _torch_stft(audio: np.ndarray, sr: int, config: EchoConfig) -> SpectrogramResult:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    waveform = torch.from_numpy(audio)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    waveform = waveform.to(device)

    win_length = config.win_length or config.n_fft
    window = torch.hann_window(win_length, device=device) if config.window == "hann" else None

    spec = torch.stft(
        waveform,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=win_length,
        window=window,
        center=config.center,
        return_complex=True,
    )

    magnitude = torch.abs(spec).cpu().numpy()
    freqs = np.linspace(0, sr / 2, magnitude.shape[-2])
    times = np.arange(magnitude.shape[-1]) * config.hop_length / sr

    db = 20 * np.log10(np.maximum(magnitude, 1e-12))
    db -= db.max()
    db = db_clip(db, config.db_floor, config.db_ceiling)

    log_info(f"STFT computed with torch on {device} for {audio.shape[-1]} samples")

    return SpectrogramResult(
        magnitude=magnitude,
        db=db,
        times=times,
        freqs=freqs,
        sr=np.array([sr], dtype=np.int32),
    )


def _librosa_stft(audio: np.ndarray, sr: int, config: EchoConfig) -> SpectrogramResult:
    if librosa is None:
        raise RuntimeError("librosa is required for CPU STFT computation")

    spec = librosa.stft(
        audio,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        window=config.window,
        center=config.center,
    )
    magnitude = np.abs(spec)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=config.n_fft)
    times = librosa.frames_to_time(np.arange(magnitude.shape[-1]), sr=sr, hop_length=config.hop_length)

    db = 20 * np.log10(np.maximum(magnitude, 1e-12))
    db -= db.max()
    db = db_clip(db, config.db_floor, config.db_ceiling)

    log_info("STFT computed with librosa (CPU path)")

    return SpectrogramResult(
        magnitude=magnitude,
        db=db,
        times=times,
        freqs=freqs,
        sr=np.array([sr], dtype=np.int32),
    )


def compute_spectrogram(audio: np.ndarray, sr: int, config: EchoConfig) -> SpectrogramResult:
    """Compute the spectrogram, preferring the GPU implementation."""

    if torch is not None and torch.cuda.is_available():  # pragma: no cover - GPU specific
        return _torch_stft(audio, sr, config)
    return _librosa_stft(audio, sr, config)


def save_spectrogram(result: SpectrogramResult, cache_path: Path) -> None:
    """Persist the spectrogram result to disk."""

    arrays = {k: ensure_numpy(v) for k, v in result.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(cache_path, **arrays)


def load_spectrogram(cache_path: Path) -> SpectrogramResult:
    with np.load(cache_path) as data:
        result = {k: data[k] for k in data.files}
    return SpectrogramResult(result)
