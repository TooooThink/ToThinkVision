"""Base exporter class for all export formats."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from app.config import settings
from app.schemas import StructuredOutput


class BaseExporter(ABC):
    """Abstract base for all exporters."""

    format_name: str = "base"
    file_extension: str = ".json"
    mime_type: str = "application/json"

    @abstractmethod
    def export(self, data: StructuredOutput) -> Path:
        """Export structured data to target format. Returns output file path."""
        ...

    @staticmethod
    def save_json(data: Any, output_path: Path) -> Path:
        """Save data as JSON to the given path."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        return output_path

    def _output_path(self, source_name: str, suffix: str = "") -> Path:
        """Generate output file path."""
        stem = Path(source_name).stem
        ext = suffix or self.file_extension
        return settings.output_dir / f"{stem}_{self.format_name}{ext}"
