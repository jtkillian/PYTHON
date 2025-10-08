"""Disk caching utilities for spectrogram data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from .config import EchoConfig
from .stft import SpectrogramResult, load_spectrogram, save_spectrogram
from .utils import hash_config, log_info


def cache_paths(root: Path, file_path: Path, config: EchoConfig) -> tuple[Path, Path]:
    cache_dir = config.resolve_cache_dir(root)
    cache_key = hash_config(file_path, config.as_dict())
    cache_npz = cache_dir / f"{cache_key}.npz"
    cache_meta = cache_dir / f"{cache_key}.json"
    return cache_npz, cache_meta


def load_cache(cache_npz: Path, cache_meta: Path) -> Optional[SpectrogramResult]:
    if not cache_npz.exists() or not cache_meta.exists():
        return None
    try:
        with cache_meta.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)
        log_info(f"Loading cached spectrogram {cache_npz.name}")
        result = load_spectrogram(cache_npz)
        result["meta"] = meta
        return result
    except Exception:  # pragma: no cover - corrupted cache
        return None


def save_cache(cache_npz: Path, cache_meta: Path, result: SpectrogramResult, meta: Dict[str, object]) -> None:
    save_spectrogram(result, cache_npz)
    with cache_meta.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    log_info(f"Cached spectrogram {cache_npz.name}")
