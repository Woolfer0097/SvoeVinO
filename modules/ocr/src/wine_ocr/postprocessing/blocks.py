"""Text block ordering and duplicate suppression."""

from __future__ import annotations

from ..contracts import RecognizedTextBlock, TextBlock
from .normalization import normalize_text


def build_text_blocks(raw_blocks: list[RecognizedTextBlock]) -> list[TextBlock]:
    """Convert raw engine blocks into normalized public blocks."""

    blocks: list[TextBlock] = []
    for block in raw_blocks:
        normalized = normalize_text(block.text)
        if not normalized:
            continue
        blocks.append(
            TextBlock(
                text=block.text,
                normalized_text=normalized,
                confidence=block.confidence,
                bbox=block.bbox,
                source_variant=block.source_variant,
            )
        )
    return deduplicate_text_blocks(sort_text_blocks(blocks))


def sort_text_blocks(blocks: list[TextBlock]) -> list[TextBlock]:
    """Sort blocks in natural reading order using bounding boxes when present."""

    return sorted(
        blocks,
        key=lambda block: (
            block.bbox.min_y if block.bbox is not None else float("inf"),
            block.bbox.min_x if block.bbox is not None else float("inf"),
            block.normalized_text,
        ),
    )


def deduplicate_text_blocks(
    blocks: list[TextBlock],
    *,
    iou_threshold: float = 0.6,
) -> list[TextBlock]:
    """Drop obvious duplicates from full-image and central-crop passes."""

    kept: list[TextBlock] = []
    for block in blocks:
        duplicate_index = _find_duplicate_index(block, kept, iou_threshold)
        if duplicate_index is None:
            kept.append(block)
            continue

        current = kept[duplicate_index]
        if _confidence_value(block.confidence) > _confidence_value(current.confidence):
            kept[duplicate_index] = block
    return kept


def _find_duplicate_index(
    block: TextBlock,
    kept: list[TextBlock],
    iou_threshold: float,
) -> int | None:
    for index, candidate in enumerate(kept):
        if candidate.normalized_text != block.normalized_text:
            continue
        if candidate.bbox is None or block.bbox is None:
            return index
        if candidate.bbox.intersection_over_union(block.bbox) >= iou_threshold:
            return index
    return None


def _confidence_value(value: float | None) -> float:
    return value if value is not None else -1.0
