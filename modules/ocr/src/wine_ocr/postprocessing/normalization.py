"""Text normalization helpers.

Normalization is intentionally conservative. Raw OCR output remains available in
the final result, and this layer avoids context-free substitutions such as O/0
or I/1 because those can damage wine names and vintages.
"""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+", flags=re.UNICODE)
_TOKEN_RE = re.compile(
    r"[0-9]+(?:[.,][0-9]+)?%?|[a-zA-Zа-яА-ЯёЁ]+(?:[-'][a-zA-Zа-яА-ЯёЁ]+)?",
    flags=re.UNICODE,
)


def normalize_text(text: str) -> str:
    """Return a normalized copy of OCR text for matching and field extraction."""

    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\u00a0", " ")
    normalized = normalized.casefold()
    normalized = _WHITESPACE_RE.sub(" ", normalized)
    return normalized.strip()


def tokenize_text(text: str) -> list[str]:
    """Tokenize normalized OCR text into simple word/number tokens."""

    return [match.group(0) for match in _TOKEN_RE.finditer(text)]
