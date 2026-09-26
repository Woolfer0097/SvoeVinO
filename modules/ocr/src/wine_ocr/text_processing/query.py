"""Prepare a matching-only text view while preserving the OCR result."""

from __future__ import annotations

import re


_VISUAL_LATIN_TO_CYRILLIC = str.maketrans({
    "a": "а", "b": "в", "c": "с", "e": "е", "h": "н", "k": "к",
    "m": "м", "o": "о", "p": "р", "t": "т", "x": "х", "y": "у",
    "3": "з", "6": "б", "0": "о",
})
_WORD = re.compile(r"\w+")


def _cyrillic_view(text: str) -> str:
    """Repair words composed entirely of Cyrillic and OCR lookalikes."""

    def convert(match: re.Match[str]) -> str:
        token = match.group().casefold()
        if len(token) < 3 or token.isdecimal():
            return token
        if all(
            ord(character) in _VISUAL_LATIN_TO_CYRILLIC
            or "а" <= character <= "я"
            or character == "ё"
            for character in token
        ):
            return token.translate(_VISUAL_LATIN_TO_CYRILLIC)
        return token

    return _WORD.sub(convert, text)


def prepare_embedding_text(normalized_text: str, candidate_name: str | None) -> str:
    """Keep every OCR word and emphasize the likely central label title once."""

    full_text = _cyrillic_view(normalized_text)
    name = _cyrillic_view(candidate_name or "").strip()
    return f"{name} {full_text}" if name else full_text
