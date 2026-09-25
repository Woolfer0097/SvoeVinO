"""Extract lightweight discriminative field candidates from OCR text."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator

from ..contracts import CandidateField, TextBlock

YEAR_RE = re.compile(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)")
PERCENT_RE = re.compile(r"(?<!\d)(\d{1,2}(?:[.,]\d{1,2})?)\s*(%)")
VOLUME_RE = re.compile(
    r"(?<!\d)(0[.,]\d{1,3}|[1-9]\d{0,3}(?:[.,]\d{1,3})?)\s*"
    r"(мл|ml|л|l|cl|сl)(?![a-zA-Zа-яА-ЯёЁ])",
    flags=re.IGNORECASE,
)


def extract_candidate_fields(
    blocks: Iterable[TextBlock],
    normalized_text: str,
) -> list[CandidateField]:
    """Extract simple year, percentage, and volume candidates."""

    candidates: list[CandidateField] = []
    for source_text, confidence in _iter_sources(blocks, normalized_text):
        candidates.extend(_extract_years(source_text, confidence))
        candidates.extend(_extract_percentages(source_text, confidence))
        candidates.extend(_extract_volumes(source_text, confidence))
    return _deduplicate_candidates(candidates)


def _iter_sources(
    blocks: Iterable[TextBlock],
    normalized_text: str,
) -> Iterator[tuple[str, float | None]]:
    yielded = False
    for block in blocks:
        yielded = True
        yield block.normalized_text, block.confidence
    if not yielded and normalized_text:
        yield normalized_text, None


def _extract_years(text: str, confidence: float | None) -> list[CandidateField]:
    return [
        CandidateField(
            field_type="year",
            value=match.group(1),
            normalized_value=match.group(1),
            source_text=text,
            confidence=confidence,
        )
        for match in YEAR_RE.finditer(text)
    ]


def _extract_percentages(text: str, confidence: float | None) -> list[CandidateField]:
    candidates: list[CandidateField] = []
    for match in PERCENT_RE.finditer(text):
        value = f"{match.group(1)}{match.group(2)}"
        candidates.append(
            CandidateField(
                field_type="percentage",
                value=value,
                normalized_value=value.replace(",", "."),
                source_text=text,
                confidence=confidence,
            )
        )
    return candidates


def _extract_volumes(text: str, confidence: float | None) -> list[CandidateField]:
    candidates: list[CandidateField] = []
    for match in VOLUME_RE.finditer(text):
        number = match.group(1)
        unit = _normalize_volume_unit(match.group(2))
        candidates.append(
            CandidateField(
                field_type="volume",
                value=f"{number} {match.group(2)}",
                normalized_value=f"{number.replace(',', '.')} {unit}",
                source_text=text,
                confidence=confidence,
            )
        )
    return candidates


def _normalize_volume_unit(unit: str) -> str:
    normalized = unit.casefold()
    if normalized in {"мл", "ml"}:
        return "ml"
    if normalized in {"л", "l"}:
        return "l"
    if normalized in {"cl", "сl"}:
        return "cl"
    return normalized


def _deduplicate_candidates(candidates: list[CandidateField]) -> list[CandidateField]:
    deduplicated: dict[tuple[str, str], CandidateField] = {}
    for candidate in candidates:
        key = (candidate.field_type, candidate.normalized_value)
        current = deduplicated.get(key)
        if current is None or _confidence_value(candidate.confidence) > _confidence_value(
            current.confidence
        ):
            deduplicated[key] = candidate
    return list(deduplicated.values())


def _confidence_value(value: float | None) -> float:
    return value if value is not None else -1.0
