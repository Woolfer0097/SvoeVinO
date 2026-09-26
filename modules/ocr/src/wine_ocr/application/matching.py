"""Thin orchestration: existing OCR, text embedding, then catalog comparison."""

from __future__ import annotations

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
) -> dict[str, dict[str, float]]:
    """Find up to ten catalog IDs without changing the existing OCR result."""

    result = ocr_runtime.run_ocr_from_bytes(image_bytes, config)
    if not result.normalized_text.strip():
        return {"top_10": {}}
    query_text = prepare_embedding_text(result.normalized_text, result.candidate_name)
    embedding = text_embedder.embed(query_text)
    return compare_embedding(embedding, repository, top_k=10)
