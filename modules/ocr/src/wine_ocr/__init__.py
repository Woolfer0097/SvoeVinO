"""Lightweight public contracts for the wine OCR module.

Importing this package must stay cheap: no OCR backend is initialized here.
"""

from .contracts import (
    BoundingBox,
    CandidateField,
    OCRResult,
    Point,
    RecognizedTextBlock,
    TextBlock,
)

__all__ = [
    "BoundingBox",
    "CandidateField",
    "OCRResult",
    "Point",
    "RecognizedTextBlock",
    "TextBlock",
]

__version__ = "0.1.0"
