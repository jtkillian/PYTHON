import numpy as np
from echo.ui_fourpane import _time_window_slice  # type: ignore[import]


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
