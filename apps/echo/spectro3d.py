"""Typer CLI entrypoint for the Echo 3D spectrogram application."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
import numpy as np

APP_ROOT = Path(__file__).resolve().parent
SRC_ROOT = APP_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from echo import EchoConfig, load_audio, compute_spectrogram  # noqa: E402
from echo.cache import cache_paths, load_cache, save_cache  # noqa: E402
from echo.mp4 import render_mp4  # noqa: E402
from echo.playback import AudioPlayback, PlaybackState  # noqa: E402
from echo.ui_fourpane import FourPaneUI, UIState  # noqa: E402
from echo.utils import device_summary, log_info  # noqa: E402

app = typer.Typer(add_completion=False)


def apply_overrides(config: EchoConfig, **kwargs) -> EchoConfig:
    overrides = {k: v for k, v in kwargs.items() if v is not None}
    config.update_from_cli(overrides)
    return config


def prepare_state(path: Path, config: EchoConfig) -> tuple[UIState, np.ndarray, int]:
    audio, sr = load_audio(path, config)
    cache_npz, cache_meta = cache_paths(APP_ROOT, path, config)
    cached = load_cache(cache_npz, cache_meta)
    if cached is None:
        log_info(f"Computing spectrogram ({device_summary()})")
        result = compute_spectrogram(audio, sr, config)
        meta = {
            "config": config.as_dict(),
            "samplerate": sr,
            "path": str(path),
        }
        save_cache(cache_npz, cache_meta, result, meta)
    else:
        result = cached

    db = result["db"]
    if db.ndim == 3:
        db = db[0]
    times = result["times"]
    freqs = result["freqs"]
    duration = max(times[-1], audio.shape[-1] / sr) if times.size else audio.shape[-1] / sr
    waveform = audio if audio.ndim == 1 else audio[0]
    waveform = np.asarray(waveform, dtype=np.float32)
    state = UIState(times=times, freqs=freqs, db=db, waveform=waveform, sr=sr, duration=float(duration))
    return state, audio, sr


def build_config(**kwargs) -> EchoConfig:
    config = EchoConfig()
    return apply_overrides(config, **kwargs)


@app.command()
def demo(
    wav_path: Path = typer.Argument(..., exists=True, readable=True),
    mp4: Optional[Path] = typer.Option(None, help="Optional MP4 export path"),
    fps: int = typer.Option(30, help="Output FPS"),
    fast: bool = typer.Option(False, help="Render MP4 with performance LoD"),
    n_fft: Optional[int] = typer.Option(None, "--n-fft", help="FFT size"),
    hop_length: Optional[int] = typer.Option(None, "--hop-length", help="Hop length"),
    win_length: Optional[int] = typer.Option(None, "--win-length", help="Window length"),
    sr: Optional[int] = typer.Option(None, "--sr", help="Target sample rate"),
    db_floor: Optional[float] = typer.Option(None, "--db-floor", help="dB floor"),
    db_ceiling: Optional[float] = typer.Option(None, "--db-ceiling", help="dB ceiling"),
    gate_width_s: Optional[float] = typer.Option(None, "--gate-width", help="Initial gate width in seconds"),
    gate_rate: Optional[float] = typer.Option(None, "--gate-rate", help="Gate follow rate"),
    mono: bool = typer.Option(True, "--mono/--stereo", help="Mix to mono"),
    enable_playback: bool = typer.Option(True, "--playback/--no-playback", help="Enable audio playback"),
) -> None:
    """Precompute, open the UI, and optionally export an MP4."""

    config = build_config(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        sr=sr,
        db_floor=db_floor,
        db_ceiling=db_ceiling,
        gate_width_s=gate_width_s,
        gate_rate=gate_rate,
        mono=mono,
        enable_playback=enable_playback,
    )
    state, audio, sr = prepare_state(wav_path, config)
    playback = None
    if config.enable_playback:
        playback = AudioPlayback(PlaybackState(audio=np.atleast_2d(audio), samplerate=sr))
    ui = FourPaneUI(state=state, config=config, playback=playback)
    if mp4 is not None:
        render_mp4(FourPaneUI(state=state, config=config, headless=True), mp4, fps=fps, fast=fast)
    ui.show()


@app.command()
def view(
    wav_path: Path = typer.Argument(..., exists=True, readable=True),
    n_fft: Optional[int] = typer.Option(None, "--n-fft", help="FFT size"),
    hop_length: Optional[int] = typer.Option(None, "--hop-length", help="Hop length"),
    win_length: Optional[int] = typer.Option(None, "--win-length", help="Window length"),
    sr: Optional[int] = typer.Option(None, "--sr", help="Target sample rate"),
    db_floor: Optional[float] = typer.Option(None, "--db-floor", help="dB floor"),
    db_ceiling: Optional[float] = typer.Option(None, "--db-ceiling", help="dB ceiling"),
    gate_width_s: Optional[float] = typer.Option(None, "--gate-width", help="Initial gate width in seconds"),
    gate_rate: Optional[float] = typer.Option(None, "--gate-rate", help="Gate follow rate"),
    mono: bool = typer.Option(True, "--mono/--stereo", help="Mix to mono"),
    enable_playback: bool = typer.Option(True, "--playback/--no-playback", help="Enable audio playback"),
) -> None:
    """Precompute and open the interactive UI."""

    config = build_config(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        sr=sr,
        db_floor=db_floor,
        db_ceiling=db_ceiling,
        gate_width_s=gate_width_s,
        gate_rate=gate_rate,
        mono=mono,
        enable_playback=enable_playback,
    )
    state, audio, sr = prepare_state(wav_path, config)
    playback = None
    if config.enable_playback:
        playback = AudioPlayback(PlaybackState(audio=np.atleast_2d(audio), samplerate=sr))
    ui = FourPaneUI(state=state, config=config, playback=playback)
    ui.show()


@app.command("render-mp4")
def render_mp4_cmd(
    wav_path: Path = typer.Argument(..., exists=True, readable=True),
    out: Path = typer.Option(..., help="Output MP4 path"),
    fps: int = typer.Option(30, help="Frames per second"),
    codec: str = typer.Option("h264", help="FFmpeg codec"),
    fast: bool = typer.Option(False, help="Use performance LoD"),
    n_fft: Optional[int] = typer.Option(None, "--n-fft", help="FFT size"),
    hop_length: Optional[int] = typer.Option(None, "--hop-length", help="Hop length"),
    win_length: Optional[int] = typer.Option(None, "--win-length", help="Window length"),
    sr: Optional[int] = typer.Option(None, "--sr", help="Target sample rate"),
    db_floor: Optional[float] = typer.Option(None, "--db-floor", help="dB floor"),
    db_ceiling: Optional[float] = typer.Option(None, "--db-ceiling", help="dB ceiling"),
    gate_width_s: Optional[float] = typer.Option(None, "--gate-width", help="Initial gate width in seconds"),
    gate_rate: Optional[float] = typer.Option(None, "--gate-rate", help="Gate follow rate"),
    mono: bool = typer.Option(True, "--mono/--stereo", help="Mix to mono"),
    enable_playback: bool = typer.Option(True, "--playback/--no-playback", help="Enable audio playback"),
) -> None:
    """Render an MP4 without opening the UI."""

    config = build_config(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        sr=sr,
        db_floor=db_floor,
        db_ceiling=db_ceiling,
        gate_width_s=gate_width_s,
        gate_rate=gate_rate,
        mono=mono,
        enable_playback=enable_playback,
    )
    state, audio, sr = prepare_state(wav_path, config)
    ui = FourPaneUI(state=state, config=config, headless=True)
    render_mp4(ui, out, fps=fps, codec=codec, fast=fast)


@app.command()
def play(
    wav_path: Path = typer.Argument(..., exists=True, readable=True),
    n_fft: Optional[int] = typer.Option(None, "--n-fft", help="FFT size"),
    hop_length: Optional[int] = typer.Option(None, "--hop-length", help="Hop length"),
    win_length: Optional[int] = typer.Option(None, "--win-length", help="Window length"),
    sr: Optional[int] = typer.Option(None, "--sr", help="Target sample rate"),
    db_floor: Optional[float] = typer.Option(None, "--db-floor", help="dB floor"),
    db_ceiling: Optional[float] = typer.Option(None, "--db-ceiling", help="dB ceiling"),
    gate_width_s: Optional[float] = typer.Option(None, "--gate-width", help="Initial gate width in seconds"),
    gate_rate: Optional[float] = typer.Option(None, "--gate-rate", help="Gate follow rate"),
    mono: bool = typer.Option(True, "--mono/--stereo", help="Mix to mono"),
    enable_playback: bool = typer.Option(True, "--playback/--no-playback", help="Enable audio playback"),
) -> None:
    """Start playback with the 4-pane UI."""

    config = build_config(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        sr=sr,
        db_floor=db_floor,
        db_ceiling=db_ceiling,
        gate_width_s=gate_width_s,
        gate_rate=gate_rate,
        mono=mono,
        enable_playback=enable_playback,
    )
    state, audio, sr = prepare_state(wav_path, config)
    playback = AudioPlayback(PlaybackState(audio=np.atleast_2d(audio), samplerate=sr))
    playback.play()
    ui = FourPaneUI(state=state, config=config, playback=playback)
    ui.show()


if __name__ == "__main__":
    app()
