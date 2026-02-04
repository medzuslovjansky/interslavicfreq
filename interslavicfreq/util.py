from __future__ import annotations
from pathlib import Path

def data_path(filename: str | None = None) -> Path:
    """
    Get a path to a file in the data directory.
    """
    base = Path(__file__).parent
    if filename is None:
        return base / "data" / "frequency"
    return base / "data" / "frequency" / filename