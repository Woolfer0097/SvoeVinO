"""Turn OCR text into a query embedding without importing ML libraries eagerly."""

from .embedding import E5TextEmbedder, TextEmbedder, TextEmbedding, TextEmbeddingError
from .query import prepare_embedding_text

__all__ = [
    "E5TextEmbedder", "TextEmbedder", "TextEmbedding", "TextEmbeddingError",
    "prepare_embedding_text",
]
