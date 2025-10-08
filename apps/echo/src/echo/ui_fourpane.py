"""Single window four-pane Matplotlib UI for Echo."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider


if TYPE_CHECKING:
    from matplotlib.backend_bases import CloseEvent, KeyEvent, MouseEvent
    from mpl_toolkits.mplot3d import Axes3D

from .colormap import DEFAULT_COLORMAP
from .config import EchoConfig
from .playback import AudioPlayback
from .utils import hud_lines, log_info


@dataclass
class UIState:
    times: np.ndarray
    freqs: np.ndarray
    db: np.ndarray
    waveform: np.ndarray
    wave_times: np.ndarray
    sr: int
    duration: float


def _time_window_slice(times: np.ndarray, start: float, end: float) -> slice:
    if times.size == 0:
        return slice(0, 0)
    left = int(np.searchsorted(times, start, side="left"))
    right = int(np.searchsorted(times, end, side="right"))
    if left == right:
        left = max(0, min(left, times.size - 1))
        right = min(times.size, left + 1)
    return slice(left, right)


class FourPaneUI:
    """Encapsulates the Matplotlib figure and interactions."""

    def __init__(
        self,
        state: UIState,
        config: EchoConfig,
        playback: AudioPlayback | None = None,
        headless: bool = False,
    ) -> None:
        self.state = state
        self.config = config
        self.playback = playback
        self.headless = headless

        self.current_time = 0.0
        self.gate_width = config.gate_width_s
        self.gate_rate = config.gate_rate
        self.fidelity = "Performance"
        self.fine_step = config.fine_step
        self.fine_step_alt = config.fine_step_alt
        self.gate_center = 0.0

        self.figure = plt.figure(figsize=(12, 8))
        manager = getattr(self.figure.canvas, "manager", None)
        if manager is not None and not headless:
            manager.set_window_title("Echo — 3D Spectrogram")
        gs = GridSpec(2, 2, figure=self.figure)

        self.ax3d = cast("Axes3D", self.figure.add_subplot(gs[0, 0], projection="3d"))
        self.ax_spec = self.figure.add_subplot(gs[0, 1])
        self.ax_wave = self.figure.add_subplot(gs[1, 0])
        self.ax_slice = self.figure.add_subplot(gs[1, 1])
        slider_ax = self.figure.add_axes([0.1, 0.04, 0.6, 0.03])

        self.slider = Slider(slider_ax, "Time", 0.0, max(0.01, self.state.duration), valinit=0.0)
        self.slider.on_changed(self._on_slider)

        self.hud = self.figure.text(
            0.78,
            0.02,
            "",
            ha="left",
            va="bottom",
            fontsize=9,
            family="monospace",
            color="white",
            bbox=dict(facecolor="black", alpha=0.5, edgecolor="white"),
        )
        self.fine_label = self.figure.text(
            0.7,
            0.02,
            "",
            ha="left",
            va="bottom",
            fontsize=10,
            family="monospace",
            color="yellow",
            bbox=dict(facecolor="black", alpha=0.4, edgecolor="yellow"),
        )

        self._init_panes()
        self._connect_events()
        self.update_all()

    # ------------------------------------------------------------------
    def _init_panes(self) -> None:
        db = self.state.db
        times = self.state.times
        freqs = self.state.freqs

        self.ax3d.set_xlabel("Time [s]")
        self.ax3d.set_ylabel("Freq [Hz]")
        self.ax3d.set_zlabel("Level [dB]")

        self.spec_img = self.ax_spec.imshow(
            db,
            aspect="auto",
            origin="lower",
            extent=[times[0], times[-1], freqs[0], freqs[-1]],
            cmap=DEFAULT_COLORMAP,
        )
        self.ax_spec.set_xlabel("Time [s]")
        self.ax_spec.set_ylabel("Freq [Hz]")
        self.figure.colorbar(self.spec_img, ax=self.ax_spec, shrink=0.8, pad=0.02)

        (self.wave_line,) = self.ax_wave.plot(self.state.wave_times, self.state.waveform, color="#3fa7d6")
        self.ax_wave.set_xlabel("Time [s]")
        self.ax_wave.set_ylabel("Amplitude")

        (self.slice_line,) = self.ax_slice.plot(freqs, db[:, 0], color="#ffcc00")
        self.ax_slice.set_xlabel("Freq [Hz]")
        self.ax_slice.set_ylabel("Level [dB]")

        self.spec_cursor = self.ax_spec.axvline(0.0, color="white", lw=1.0)
        self.wave_cursor = self.ax_wave.axvline(0.0, color="red", lw=1.0)

    def _connect_events(self) -> None:
        if self.headless:
            return
        self.figure.canvas.mpl_connect("key_press_event", self._on_key)
        self.figure.canvas.mpl_connect("button_press_event", self._on_click)
        self.figure.canvas.mpl_connect("close_event", self._on_close)
        if self.playback is not None:
            timer = self.figure.canvas.new_timer(interval=50)
            timer.add_callback(self._tick)
            timer.start()
            self._timer = timer
        else:
            self._timer = None

    # ------------------------------------------------------------------
    def _tick(self) -> None:
        if self.playback is None:
            return
        pos = self.playback.state.position
        if self.playback.state.playing:
            self.set_time(pos)

    def _on_close(self, event: CloseEvent) -> None:  # pragma: no cover - GUI callback
        if self.playback is not None:
            self.playback.shutdown()
        if hasattr(self, "_timer") and self._timer is not None:
            self._timer.stop()

    def _on_slider(self, value: float) -> None:
        self.set_time(float(value))

    def _on_click(self, event: MouseEvent) -> None:  # pragma: no cover - GUI callback
        if event.inaxes in (self.ax_spec, self.ax_wave) and event.xdata is not None:
            self.set_time(float(event.xdata))
            if self.playback is not None:
                self.playback.seek(self.current_time)
        elif self.fine_label.contains(event)[0]:
            self.toggle_fine_step()

    def _on_key(self, event: KeyEvent) -> None:  # pragma: no cover - GUI callback
        key = event.key or ""
        actions: dict[str, Callable[[], None]] = {
            "left": lambda: self.seek_relative(-5.0),
            "right": lambda: self.seek_relative(5.0),
            ",": lambda: self.seek_relative(-self.fine_step),
            ".": lambda: self.seek_relative(self.fine_step),
            "[": lambda: self.adjust_gate(-1.0),
            "]": lambda: self.adjust_gate(1.0),
            "q": lambda: self.adjust_rate(-0.25),
            "e": lambda: self.adjust_rate(0.25),
            "p": self.toggle_fidelity,
        }
        action = actions.get(key)
        if action is not None:
            action()
            self.update_all()
            return
        if key == " " and self.playback is not None:
            self.playback.toggle()
            self.update_all()

    # ------------------------------------------------------------------
    def seek_relative(self, delta: float) -> None:
        self.set_time(self.current_time + delta)
        if self.playback is not None:
            self.playback.seek(self.current_time)

    def set_time(self, value: float) -> None:
        duration = self.state.duration
        self.current_time = max(0.0, min(float(value), duration))
        if not self.headless:
            self.slider.eventson = False
            self.slider.set_val(self.current_time)
            self.slider.eventson = True
        self.update_all()

    def adjust_gate(self, delta: float) -> None:
        self.gate_width = float(np.clip(self.gate_width + delta, 2.0, 60.0))

    def adjust_rate(self, delta: float) -> None:
        self.gate_rate = float(np.clip(self.gate_rate + delta, 0.25, 4.0))

    def toggle_fidelity(self) -> None:
        if self.fidelity == "Performance":
            self.fidelity = "Hi-Fi"
        else:
            self.fidelity = "Performance"

    def toggle_fine_step(self) -> None:
        self.fine_step, self.fine_step_alt = self.fine_step_alt, self.fine_step
        self.update_hud()

    # ------------------------------------------------------------------
    def update_all(self) -> None:
        self.update_cursors()
        self.update_slice()
        self.update_3d()
        self.update_hud()
        if not self.headless:
            self.figure.canvas.draw_idle()

    def update_cursors(self) -> None:
        self.spec_cursor.set_xdata([self.current_time, self.current_time])
        self.wave_cursor.set_xdata([self.current_time, self.current_time])

    def update_slice(self) -> None:
        times = self.state.times
        idx = int(np.argmin(np.abs(times - self.current_time)))
        self.slice_line.set_ydata(self.state.db[:, idx])

    def update_3d(self) -> None:
        times = self.state.times
        freqs = self.state.freqs
        db = self.state.db

        if times.size == 0 or freqs.size == 0:
            return

        if not hasattr(self, "_surface"):
            self._surface = None

        follow = min(1.0, 0.1 * self.gate_rate)
        self.gate_center += (self.current_time - self.gate_center) * follow

        half = self.gate_width / 2.0
        start = max(float(times[0]), self.gate_center - half)
        end = min(float(times[-1]), self.gate_center + half)
        window = _time_window_slice(times, start, end)
        gate_times = times[window]
        if gate_times.size == 0:
            idx = int(np.argmin(np.abs(times - self.current_time)))
            window = slice(idx, min(idx + 1, times.size))
            gate_times = times[window]
        gate_db = db[:, window]

        subsample = (
            self.config.interactive_subsample
            if self.fidelity == "Performance"
            else self.config.hifi_subsample
        )
        sub_t = max(1, int(subsample[0]))
        sub_f = max(1, int(subsample[1]))
        gate_times = gate_times[::sub_t]
        gate_db = gate_db[::sub_f, ::sub_t]
        gate_freqs = freqs[::sub_f]

        max_cols = max(1, getattr(self.config, "max_gate_columns", 512))
        max_rows = max(1, getattr(self.config, "max_gate_rows", 256))
        if gate_times.size > max_cols:
            stride = int(np.ceil(gate_times.size / max_cols))
            gate_times = gate_times[::stride]
            gate_db = gate_db[:, ::stride]
        if gate_freqs.size > max_rows:
            stride = int(np.ceil(gate_freqs.size / max_rows))
            gate_freqs = gate_freqs[::stride]
            gate_db = gate_db[::stride, :]

        if gate_times.size == 0 or gate_freqs.size == 0:
            return

        time_grid, freq_grid = np.meshgrid(gate_times, gate_freqs, copy=False)
        if self._surface is not None:
            self._surface.remove()
        self._surface = self.ax3d.plot_surface(time_grid, freq_grid, gate_db, cmap=DEFAULT_COLORMAP)
        self.ax3d.set_xlim(start, end)
        self.ax3d.set_ylim(freqs[0], freqs[-1])
        self.ax3d.set_zlim(self.config.db_floor, self.config.db_ceiling)

    def update_hud(self) -> None:
        text = hud_lines(
            gate_width=self.gate_width,
            gate_rate=self.gate_rate,
            fidelity=self.fidelity,
            fine_step=self.fine_step,
        )
        self.hud.set_text(text)
        self.fine_label.set_text(f"fine={self.fine_step:.1f}s")

    # ------------------------------------------------------------------
    def show(self) -> None:  # pragma: no cover - GUI entry
        if self.headless:
            log_info("Headless mode active; figure not shown")
            return
        plt.show()

    def render_frame(self, time_value: float) -> np.ndarray:
        """Render a frame for the MP4 exporter at ``time_value`` seconds."""

        self.set_time(time_value)
        self.figure.canvas.draw()
        width, height = self.figure.canvas.get_width_height()
        buffer = np.frombuffer(self.figure.canvas.tostring_argb(), dtype=np.uint8)
        buffer = buffer.reshape((height, width, 4))
        rgba = buffer[:, :, [1, 2, 3, 0]]
        return rgba
