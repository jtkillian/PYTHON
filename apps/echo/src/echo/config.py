"""Configuration utilities for the Echo application."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class EchoConfig:
    """Application configuration with sensible defaults.

    The dataclass is intentionally flat so it can be serialized easily and
    hashed for cache keys.
    """

    # Analysis parameters
    n_fft: int = 2048
    hop_length: int = 256
    win_length: int | None = None
    window: str = "hann"
    sr: int | None = None
    center: bool = True
    mono: bool = True
    db_floor: float = -90.0
    db_ceiling: float = 0.0

    # 3D rendering parameters
    interactive_subsample: tuple[int, int] = (2, 2)
    hifi_subsample: tuple[int, int] = (1, 1)
    gate_width_s: float = 10.0
    gate_rate: float = 1.0
    max_gate_columns: int = 512
    max_gate_rows: int = 256

    # Paths
    cache_dir: str = "dist/caches"

    # UI
    fine_step: float = 1.0
    fine_step_alt: float = 0.5
    waveform_max_points: int = 20000

    # Playback
    enable_playback: bool = True

    extra: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        """Return a shallow dictionary suitable for hashing or serialization."""

        data = asdict(self)
        data["interactive_subsample"] = tuple(self.interactive_subsample)
        data["hifi_subsample"] = tuple(self.hifi_subsample)
        data["cache_dir"] = str(self.cache_dir)
        return data

    def resolve_cache_dir(self, root: Path) -> Path:
        """Resolve the cache directory relative to the application root."""

        cache_path = root / self.cache_dir
        cache_path.mkdir(parents=True, exist_ok=True)
        return cache_path

    def update_from_cli(self, overrides: dict[str, object]) -> None:
        """Apply CLI overrides to the configuration in place."""

        for key, value in overrides.items():
            if not hasattr(self, key):  # pragma: no cover - defensive
                raise AttributeError(f"Unknown configuration option: {key}")
            setattr(self, key, value)
