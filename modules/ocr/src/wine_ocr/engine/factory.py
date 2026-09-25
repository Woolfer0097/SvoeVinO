"""Factory for OCR backend implementations."""

from __future__ import annotations

from ..config import OCRConfig
from ..exceptions import UnsupportedOCREngineError
from .base import OCREngine


def create_engine(config: OCRConfig) -> OCREngine:
    """Create the configured OCR engine without touching unrelated modules."""

    engine_name = config.engine.strip().lower()
    if engine_name == "paddleocr":
        from .paddle import PaddleOCREngine

        return PaddleOCREngine(config)

    raise UnsupportedOCREngineError(f"Unsupported OCR engine: {config.engine}")
