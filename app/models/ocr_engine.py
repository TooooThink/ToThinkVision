"""PaddleOCR wrapper for text recognition."""

from __future__ import annotations

import logging

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

_ocr = None


def _get_mock_ocr(img: np.ndarray) -> list[dict]:
    """Generate mock OCR results for testing."""
    h, w = img.shape[:2]
    rng = np.random.RandomState(77)
    mock_texts = ["Sample Text", "Button", "Label 1", "Menu", "Title"]
    results = []
    for i, text in enumerate(mock_texts):
        max_bw = max(40, w // 4)
        max_bh = max(15, h // 8)
        bw = rng.randint(15, min(120, max_bw))
        bh = rng.randint(8, min(30, max_bh))
        x = rng.randint(10, w - bw - 10)
        y = rng.randint(10, h - bh - 10)
        results.append({
            "bbox": [float(x), float(y), float(bw), float(bh)],
            "text": text,
            "confidence": 0.92 - i * 0.05,
        })
    return results


def _get_ocr():
    """Lazy-load PaddleOCR engine."""
    global _ocr
    if _ocr is not None:
        return _ocr
    if settings.mock_mode:
        logger.info("Using mock OCR (MOCK_MODE=true)")
        return "mock"
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        return ocr
    except ImportError:
        logger.warning("paddleocr not installed, falling back to mock mode")
        return "mock"


def run_ocr(img: np.ndarray) -> list[dict]:
    """Run OCR on image. Returns list of {bbox, text, confidence}."""
    ocr_engine = _get_ocr()
    if ocr_engine == "mock":
        return _get_mock_ocr(img)

    try:
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
    except Exception as e:
        logger.error(f"OCR failed: {e}, using mock fallback")
        return _get_mock_ocr(img)
