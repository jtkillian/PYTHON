import numpy as np
from echo.utils import downsample_waveform


def test_downsample_waveform_limits_point_count() -> None:
    samplerate = 44100
    samples = np.linspace(-1.0, 1.0, samplerate * 2)
    down, times = downsample_waveform(samples, samplerate=samplerate, max_points=1000)
    assert down.shape == (1000,)
    assert times.shape == (1000,)
    np.testing.assert_allclose(times[-1], (samples.size - 1) / samplerate, rtol=1e-6)


def test_downsample_waveform_handles_empty_input() -> None:
    down, times = downsample_waveform(np.array([]), samplerate=44100, max_points=1000)
    assert down.size == 0
    assert times.size == 0
