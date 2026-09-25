"""Application service that coordinates OCR processing."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from ..config import OCRConfig
from ..contracts import OCRResult, RecognizedTextBlock
from ..engine.base import OCREngine
from ..engine.factory import create_engine
from ..postprocessing.blocks import build_text_blocks
from ..postprocessing.fields import extract_candidate_fields
from ..postprocessing.normalization import normalize_text, tokenize_text
from ..preprocessing.images import ImageVariant, build_image_variants, decode_image


def run_ocr_from_bytes(
    image_bytes: bytes,
    *,
    config: OCRConfig | None = None,
    engine: OCREngine | None = None,
) -> OCRResult:
    """Decode uploaded image bytes and run OCR."""

    active_config = config if config is not None else OCRConfig.from_env()
    image = decode_image(image_bytes, active_config)
    return run_ocr_on_image(image, config=active_config, engine=engine)


def run_ocr_on_image(
    image: Any,
    *,
    config: OCRConfig | None = None,
    engine: OCREngine | None = None,
) -> OCRResult:
    """Run OCR on an already decoded image-like object."""

    started_at = perf_counter()
    active_config = config if config is not None else OCRConfig.from_env()
    active_engine = engine if engine is not None else create_engine(active_config)

    raw_blocks: list[RecognizedTextBlock] = []
    for variant in build_image_variants(image, active_config):
        variant_blocks = active_engine.recognize(variant.image, variant.name)
        raw_blocks.extend(_map_blocks_to_original_image(variant_blocks, variant))

    text_blocks = build_text_blocks(raw_blocks)
    raw_text = "\n".join(block.text for block in text_blocks)
    normalized_text = normalize_text(raw_text)
    tokens = tokenize_text(normalized_text)
    candidate_fields = extract_candidate_fields(text_blocks, normalized_text)
    warnings = []
    if not text_blocks:
        warnings.append("No text blocks were recognized")

    return OCRResult(
        raw_text=raw_text,
        normalized_text=normalized_text,
        text_blocks=text_blocks,
        tokens=tokens,
        candidate_fields=candidate_fields,
        engine=getattr(active_engine, "name", active_config.engine),
        processing_time_ms=(perf_counter() - started_at) * 1000,
        warnings=warnings,
    )


def _map_blocks_to_original_image(
    blocks: list[RecognizedTextBlock],
    variant: ImageVariant,
) -> list[RecognizedTextBlock]:
    if variant.origin_x == 0 and variant.origin_y == 0:
        return blocks

    mapped: list[RecognizedTextBlock] = []
    for block in blocks:
        mapped.append(
            RecognizedTextBlock(
                text=block.text,
                confidence=block.confidence,
                bbox=block.bbox.translated(variant.origin_x, variant.origin_y)
                if block.bbox is not None
                else None,
                source_variant=variant.name,
            )
        )
    return mapped
