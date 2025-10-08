"""Audio playback helpers using sounddevice."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Optional

import numpy as np

try:  # pragma: no cover - optional dependency
    import sounddevice as sd
except Exception:  # pragma: no cover - fallback when sounddevice missing
    sd = None  # type: ignore

from .utils import log_info


@dataclass
class PlaybackState:
    audio: np.ndarray
    samplerate: int
    position: float = 0.0
    playing: bool = False


class AudioPlayback:
    """Lightweight wrapper around ``sounddevice`` output streams."""

    def __init__(self, state: PlaybackState):
        state.audio = np.atleast_2d(np.asarray(state.audio, dtype=np.float32))
        self.state = state
        self._stream: Optional[sd.OutputStream] = None if sd else None
        self._lock = Lock()

    def _callback(self, outdata, frames, time, status):  # pragma: no cover - realtime code
        if status:
            log_info(f"Playback status: {status}")
        with self._lock:
            if not self.state.playing:
                outdata[:] = 0
                return
            start = int(self.state.position * self.state.samplerate)
            end = start + frames
            samples = self.state.audio[..., start:end]
            if samples.shape[-1] < frames:
                samples = np.pad(samples, ((0, 0), (0, frames - samples.shape[-1])), mode="constant")
                self.state.playing = False
                self.state.position = self.duration
            else:
                self.state.position = end / self.state.samplerate
            outdata[:] = samples.T.reshape(frames, -1)

    def ensure_stream(self) -> None:
        if sd is None:
            return
        if self._stream is None:
            channels = 1 if self.state.audio.ndim == 1 else self.state.audio.shape[0]
            self._stream = sd.OutputStream(
                samplerate=self.state.samplerate,
                channels=channels,
                callback=self._callback,
                dtype="float32",
            )
            self._stream.start()
            log_info("Audio playback stream started")

    def play(self) -> None:
        if sd is None:
            return
        with self._lock:
            self.state.playing = True
        self.ensure_stream()

    def pause(self) -> None:
        with self._lock:
            self.state.playing = False

    def toggle(self) -> None:
        if self.state.playing:
            self.pause()
        else:
            self.play()

    def seek(self, seconds: float) -> None:
        with self._lock:
            self.state.position = max(0.0, min(seconds, self.duration))

    @property
    def duration(self) -> float:
        return self.state.audio.shape[-1] / self.state.samplerate

    def shutdown(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            log_info("Audio playback stream stopped")
