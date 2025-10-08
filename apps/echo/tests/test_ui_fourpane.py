import os

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
from matplotlib import pyplot as plt

from echo.config import EchoConfig
from echo.ui_fourpane import FourPaneUI, UIState, _time_window_slice  # type: ignore[import]


def test_time_window_slice_respects_start_end() -> None:
    times = np.linspace(0, 9, 10)
    window = _time_window_slice(times, 2.0, 4.0)
    np.testing.assert_allclose(times[window], np.array([2.0, 3.0, 4.0]))


def test_time_window_slice_clamps_to_bounds() -> None:
    times = np.linspace(0, 9, 10)
    window = _time_window_slice(times, -5.0, 100.0)
    np.testing.assert_allclose(times[window], times)


def test_time_window_slice_returns_single_index_when_needed() -> None:
    times = np.linspace(0, 9, 10)
    window = _time_window_slice(times, 9.5, 9.5)
    np.testing.assert_allclose(times[window], np.array([9.0]))


def test_time_window_slice_empty_times() -> None:
    times = np.array([])
    window = _time_window_slice(times, 0.0, 1.0)
    assert window == slice(0, 0)


def test_update_3d_reuses_surface_collection() -> None:
    times = np.linspace(0.0, 4.0, 64, dtype=np.float32)
    freqs = np.linspace(20.0, 8000.0, 33, dtype=np.float32)
    db = np.linspace(-80.0, 0.0, times.size * freqs.size, dtype=np.float32).reshape(freqs.size, times.size)
    wave_times = np.linspace(0.0, 4.0, 128, dtype=np.float32)
    waveform = np.sin(wave_times * 2.0 * np.pi * 0.5).astype(np.float32)
    state = UIState(
        times=times,
        freqs=freqs,
        db=db,
        waveform=waveform,
        wave_times=wave_times,
        sr=48000,
        duration=float(times[-1]),
    )
    config = EchoConfig()
    ui = FourPaneUI(state=state, config=config, headless=True)
    try:
        surface = ui._surface  # type: ignore[attr-defined]
        assert surface is not None
        ui.set_time(0.5)
        assert ui._surface is surface  # type: ignore[attr-defined]
        ui.set_time(1.5)
        assert ui._surface is surface  # type: ignore[attr-defined]
        assert ui._surface_polys is not None  # type: ignore[attr-defined]
        assert ui._surface_polys.dtype == np.float32  # type: ignore[attr-defined]
    finally:
        plt.close(ui.figure)
