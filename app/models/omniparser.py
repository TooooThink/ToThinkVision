"""OmniParser v2 — Microsoft's precise UI element detection."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

_parser = None


class OmniParser:
    """Wrapper for OmniParser v2 UI element detection."""

    def __init__(self):
        self.parser = None
        self._init_model()

    def _init_model(self):
        """Load OmniParser models."""
        try:
            # Try to load from cloned OmniParser repo
            omniparser_dir = Path(settings.model_cache_dir) / "OmniParser"
            if not omniparser_dir.exists():
                raise RuntimeError(f"OmniParser repo not found at {omniparser_dir}")

            import sys
            sys.path.insert(0, str(omniparser_dir))
            from util.utils import get_omniparser_model

            self.parser = get_omniparser_model(device=settings.device)
            logger.info("OmniParser loaded")
        except (ImportError, Exception) as e:
            raise RuntimeError(f"OmniParser failed to load: {e}")

    def parse(self, img: np.ndarray) -> list[dict]:
        """Parse UI elements from a screenshot.

        Args:
            img: (H, W, 3) RGB image

        Returns:
            list of {type, bbox_normalized, interactivity, content, confidence}
        """
        from PIL import Image
        pil_img = Image.fromarray(img)

        # OmniParser API may vary by version
        if hasattr(self.parser, "parse"):
            parsed, icons = self.parser.parse(pil_img)
        else:
            # Alternative API
            from util.utils import parse_image
            parsed, icons = parse_image(pil_img, self.parser)

        results = []
        for elem in parsed:
            results.append({
                "type": elem.get("type", "unknown"),
                "bbox": elem.get("bbox", [0, 0, 0.1, 0.1]),
                "interactivity": elem.get("interactivity", False),
                "content": elem.get("content", ""),
                "confidence": elem.get("confidence", 0.8),
            })
        return results

    def parse_to_boxes(self, img: np.ndarray) -> list[dict]:
        """Parse UI elements and return pixel-coordinate boxes.

        Returns:
            list of {bbox_pixel, type, interactivity, content, confidence}
        """
        h, w = img.shape[:2]
        results = self.parse(img)
        for r in results:
            nb = r["bbox"]  # normalized [x1, y1, x2, y2]
            r["bbox_pixel"] = [
                nb[0] * w, nb[1] * h,
                (nb[2] - nb[0]) * w, (nb[3] - nb[1]) * h
            ]
        return results


def get_omniparser() -> OmniParser:
    """Get or create OmniParser instance."""
    global _parser
    if _parser is None:
        _parser = OmniParser()
    return _parser
