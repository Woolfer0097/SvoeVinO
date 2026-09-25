"""Choose a provisional wine-name phrase from the central bottle text."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..contracts import BoundingBox, CandidateField, RecognizedTextBlock
from .normalization import normalize_text

YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
GENERIC_PHRASES = (
    "российское вино",
    "семейная винодельня",
    "произведено",
    "на территории",
)
GENERIC_LINES = {
    "вино", "винодельня", "згу", "знмп", "сухое", "полусухое",
    "полусладкое", "сладкое", "красное", "белое", "розовое",
    "брют", "brut",
}


@dataclass(frozen=True)
class _NameOption:
    value: str
    source_text: str
    confidence: float | None
    bbox: BoundingBox | None
    source_variant: str
    block_count: int
    smallest_line_height: float


def extract_candidate_name(
    blocks: list[RecognizedTextBlock],
    image_size: tuple[int, int],
) -> CandidateField | None:
    """Return one name-like phrase, preferring the central-crop OCR pass.

    This is a visual/text heuristic. Its confidence is OCR confidence, not a
    calibrated probability that the phrase is the catalog wine name.
    """

    width, _ = image_size
    for variant in ("central_crop", "full"):
        plausible = [
            block
            for block in blocks
            if block.source_variant == variant
            and _clean_name(block.text)
            and block.confidence is not None
            and block.confidence >= 0.55
            and _near_image_center(block.bbox, width)
        ]
        high_confidence = [block for block in plausible if block.confidence >= 0.7]
        regular_height = max(
            (_height(block.bbox) for block in high_confidence), default=0.0
        )
        eligible = [
            block
            for block in plausible
            if block.confidence >= 0.7
            or (
                regular_height > 0
                and _height(block.bbox) >= regular_height * 3
            )
        ]
        if not eligible:
            continue

        max_height = max((_height(block.bbox) for block in eligible), default=1.0)
        options = [_option_for_block(block) for block in eligible]
        pairable = [block for block in high_confidence if block.confidence >= 0.75]
        for index, first in enumerate(pairable):
            for second in pairable[index + 1 :]:
                if _can_join(first, second):
                    options.append(_option_for_pair(first, second))

        best = max(options, key=lambda option: _score(option, width, max_height))
        return CandidateField(
            field_type="name",
            value=best.value,
            normalized_value=normalize_text(best.value),
            source_text=best.source_text,
            confidence=best.confidence,
            metadata={
                "source_variant": best.source_variant,
                "selection": "central_text_heuristic",
                "text_block_count": best.block_count,
            },
        )
    return None


def _clean_name(text: str) -> str:
    cleaned = YEAR_RE.sub("", text).strip(" \t\r\n/|,.;:-_\"'()[]")
    normalized = normalize_text(cleaned)
    if len([character for character in normalized if character.isalpha()]) < 3:
        return ""
    if len(cleaned) > 60 or normalized in GENERIC_LINES:
        return ""
    if normalized.startswith(("згу ", "3гу ", "zgu ", "знмп ")):
        return ""
    if any(phrase in normalized for phrase in GENERIC_PHRASES):
        return ""
    return cleaned


def _near_image_center(box: BoundingBox | None, width: int) -> bool:
    if box is None or width <= 0:
        return True
    center_x = (box.min_x + box.max_x) / 2
    return abs(center_x - width / 2) <= width * 0.25


def _height(box: BoundingBox | None) -> float:
    return max(1.0, box.max_y - box.min_y) if box is not None else 1.0


def _option_for_block(block: RecognizedTextBlock) -> _NameOption:
    return _NameOption(
        value=_clean_name(block.text),
        source_text=block.text,
        confidence=block.confidence,
        bbox=block.bbox,
        source_variant=block.source_variant,
        block_count=1,
        smallest_line_height=_height(block.bbox),
    )


def _can_join(
    first: RecognizedTextBlock,
    second: RecognizedTextBlock,
) -> bool:
    left, right = first.bbox, second.bbox
    if left is None or right is None:
        return False
    height_ratio = min(_height(left), _height(right)) / max(_height(left), _height(right))
    if height_ratio < 0.6:
        return False
    overlap_y = min(left.max_y, right.max_y) - max(left.min_y, right.min_y)
    if overlap_y >= min(_height(left), _height(right)) * 0.4:
        gap_x = max(left.min_x, right.min_x) - min(left.max_x, right.max_x)
        return 0 <= gap_x <= max(_height(left), _height(right))

    if first.confidence is None or second.confidence is None:
        return False
    if min(first.confidence, second.confidence) < 0.75:
        return False
    first_width = left.max_x - left.min_x
    second_width = right.max_x - right.min_x
    if first_width <= 0 or second_width <= 0:
        return False
    if min(first_width, second_width) / max(first_width, second_width) < 0.65:
        return False
    if abs((left.min_x + left.max_x - right.min_x - right.max_x) / 2) > min(
        first_width, second_width
    ) * 0.25:
        return False
    upper, lower = (left, right) if left.min_y <= right.min_y else (right, left)
    gap_y = lower.min_y - upper.max_y
    overlap_x = min(left.max_x, right.max_x) - max(left.min_x, right.min_x)
    min_width = min(left.max_x - left.min_x, right.max_x - right.min_x)
    return -max(_height(left), _height(right)) * 0.25 <= gap_y <= max(
        _height(left), _height(right)
    ) * 0.75 and overlap_x >= min_width * 0.4


def _option_for_pair(
    first: RecognizedTextBlock,
    second: RecognizedTextBlock,
) -> _NameOption:
    assert first.bbox is not None and second.bbox is not None
    left, right = first.bbox, second.bbox
    overlap_y = min(left.max_y, right.max_y) - max(left.min_y, right.min_y)
    if overlap_y >= min(_height(left), _height(right)) * 0.4:
        ordered = sorted((first, second), key=lambda block: block.bbox.min_x)
    else:
        ordered = sorted((first, second), key=lambda block: block.bbox.min_y)

    value = " ".join(_clean_name(block.text) for block in ordered)
    points = left.points + right.points
    return _NameOption(
        value=value,
        source_text="\n".join(block.text for block in ordered),
        confidence=min(
            (block.confidence for block in ordered if block.confidence is not None),
            default=None,
        ),
        bbox=BoundingBox(points),
        source_variant=first.source_variant,
        block_count=2,
        smallest_line_height=min(_height(left), _height(right)),
    )


def _score(option: _NameOption, width: int, max_height: float) -> float:
    box = option.bbox
    size_score = min(option.smallest_line_height / max_height, 1.0)
    if box is None or width <= 0:
        center_score = 0.5
    else:
        center_x = (box.min_x + box.max_x) / 2
        center_score = max(0.0, 1.0 - abs(center_x - width / 2) / (width / 2))
    confidence = option.confidence if option.confidence is not None else 0.5
    letter_count = sum(character.isalpha() for character in option.value)
    return (
        1.5 * size_score
        + 1.5 * center_score
        + 1.5 * confidence
        + 0.8 * min(letter_count / 20, 1)
        + (0.8 if option.block_count == 2 else 0)
    )
