import importlib
import sys
from pathlib import Path


def test_package_importable() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    pkg = importlib.import_module("echo")
    assert hasattr(pkg, "__all__") or hasattr(pkg, "__package__")


def test_assets_present() -> None:
    assert (Path(__file__).resolve().parents[1] / "assets" / "wav").exists()
