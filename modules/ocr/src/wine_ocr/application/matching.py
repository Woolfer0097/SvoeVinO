"""Thin orchestration: existing OCR, text embedding, then catalog comparison."""

from __future__ import annotations
import re
import math

from ..config import OCRConfig
from ..embedding_comparison import EmbeddingRepository, compare_embedding
from ..text_processing import TextEmbedder, prepare_embedding_text
from .runtime import CachedOCRRuntime


def match_image_from_bytes(
    image_bytes: bytes,
    *,
    config: OCRConfig,
    ocr_runtime: CachedOCRRuntime,
    text_embedder: TextEmbedder,
    repository: EmbeddingRepository,
    include_evidence: bool = False,
) -> dict:
    """Find up to ten catalog IDs without changing the existing OCR result."""

    result = ocr_runtime.run_ocr_from_bytes(image_bytes, config)
    if not result.normalized_text.strip():
        return {"top_10": {}}
    query_text = prepare_embedding_text(result.normalized_text, result.candidate_name)
    embedding = text_embedder.embed(query_text)
    matches = compare_embedding(embedding, repository, top_k=10)
    if include_evidence:
        years = []
        for candidate in getattr(result, "candidate_fields", []):
            if candidate.field_type != "year" or candidate.confidence is None:
                continue
            if candidate.confidence < .8 or not re.fullmatch(r"(19|20)\d{2}", candidate.value):
                continue
            # A founding date is not a vintage. Ambiguous multiple years are
            # preserved as evidence, never silently interpreted as one vintage.
            if re.search(r"основан|основания|since|founded|established", candidate.source_text, re.I):
                continue
            years.append(int(candidate.value))
        # Character-weighted confidence from actual alphabetic text blocks;
        # prices and isolated numeric fragments must not inflate confidence.
        total, weighted = 0, 0.0
        for block in getattr(result, "text_blocks", []):
            confidence = block.confidence
            size = sum(character.isalpha() for character in block.normalized_text)
            if (size >= 3 and confidence is not None and math.isfinite(confidence)
                    and 0 <= confidence <= 1):
                total += size
                weighted += size * confidence
        matches["evidence"] = {
            "text": result.normalized_text,
            "candidate_name": result.candidate_name,
            "years": sorted(set(years)),
            "text_confidence": weighted / total if total else None,
        }
    return matches
