"""OCR backend protocol."""

from __future__ import annotations

from typing import Any, Protocol

from ..contracts import RecognizedTextBlock


class OCREngine(Protocol):
    """Text detector/recognizer boundary used by the application layer."""

    @property
    def name(self) -> str:
        """Return the backend name and model family."""

        ...

    def recognize(self, image: Any, source_variant: str) -> list[RecognizedTextBlock]:
        """Recognize text blocks from a prepared image."""

        ...
