"""Color extraction utilities."""

from __future__ import annotations

from collections import Counter

import numpy as np


def extract_dominant_color(img: np.ndarray, bbox: list[float], n_colors: int = 5) -> tuple[str, list[str]]:
    """Extract dominant color from image region defined by bbox [x, y, w, h].
    Returns (dominant_hex, top_n_hex_list).
    """
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    h_img, w_img = img.shape[:2]
    x = max(0, min(x, w_img - 1))
    y = max(0, min(y, h_img - 1))
    x2 = min(x + w, w_img)
    y2 = min(y + h, h_img)
    if x2 <= x or y2 <= y:
        return "#808080", ["#808080"]

    region = img[y:y2, x:x2]
    # Quantize colors
    quantized = (region // 32) * 32  # Reduce to 8 levels per channel
    pixels = quantized.reshape(-1, 3)

    color_counts = Counter()
    for r, g, b in pixels:
        color_counts[(int(r), int(g), int(b))] += 1

    top_colors = color_counts.most_common(n_colors)
    hex_colors = [_rgb_to_hex(r, g, b) for (r, g, b), _ in top_colors]
    dominant = hex_colors[0] if hex_colors else "#808080"
    return dominant, hex_colors


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert RGB values (0-255) to hex color string."""
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"
