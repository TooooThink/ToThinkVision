"""PaddleOCR wrapper for text recognition."""

from __future__ import annotations

import logging

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

_ocr = None


def _get_ocr():
    """Lazy-load PaddleOCR engine."""
    global _ocr
    if _ocr is not None:
        return _ocr
    try:
        from paddleocr import PaddleOCR
        _ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        return _ocr
    except ImportError:
        raise RuntimeError("paddleocr not installed. Install with: pip install paddleocr")


def run_ocr(img: np.ndarray) -> list[dict]:
    """Run OCR on image. Returns list of {bbox, text, confidence}."""
    ocr_engine = _get_ocr()

    import cv2
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    result = ocr_engine.ocr(img_bgr, cls=True)
    texts = []
    if result and result[0]:
        for line in result[0]:
            if line is None:
                continue
            box_info, (text, conf) = line
            if conf < settings.ocr_threshold:
                continue
            box = np.array(box_info)
            x_min = float(box[:, 0].min())
            y_min = float(box[:, 1].min())
            bw = float(box[:, 0].max() - x_min)
            bh = float(box[:, 1].max() - y_min)
            texts.append({
                "bbox": [x_min, y_min, bw, bh],
                "text": text,
                "confidence": float(conf),
            })
    return texts

