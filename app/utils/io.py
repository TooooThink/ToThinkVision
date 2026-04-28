"""File I/O utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json(data: Any, path: Path) -> Path:
    """Save data as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    return path


def load_json(path: Path) -> Any:
    """Load JSON from file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_files(directory: Path, pattern: str = "*") -> list[Path]:
    """List files in directory matching pattern."""
    return sorted(directory.glob(pattern))
