"""Utility script to materialize the bundled sine tone demo WAV."""

from __future__ import annotations

import base64
from pathlib import Path

BASE64_FILE = Path(__file__).resolve().parents[1] / "assets" / "wav" / "sine_base64.txt"
DEFAULT_TARGET = Path(__file__).resolve().parents[1] / "assets" / "wav" / "sine.wav"


def generate(target: Path = DEFAULT_TARGET, overwrite: bool = False) -> Path:
    """Write the sine wave sample to *target* if needed."""

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        return target
    data = base64.b64decode(BASE64_FILE.read_text().encode())
    target.write_bytes(data)
    return target


def main(overwrite: bool = False) -> None:
    path = generate(overwrite=overwrite)
    print(f"Sample written to {path}")


if __name__ == "__main__":  # pragma: no cover - CLI utility
    main()
