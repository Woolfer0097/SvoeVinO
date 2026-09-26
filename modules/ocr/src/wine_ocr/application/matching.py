"""Thin orchestration: existing OCR, text embedding, then catalog comparison."""

from __future__ import annotations

from ..config import OCRConfig
from ..embedding_comparison import EmbeddingRepository, compare_embedding
from ..text_processing import TextEmbedder
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
    embedding = text_embedder.embed(result.normalized_text)
    return compare_embedding(embedding, repository, top_k=10)
