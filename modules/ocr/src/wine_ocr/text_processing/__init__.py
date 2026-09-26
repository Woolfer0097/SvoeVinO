"""Turn OCR text into a query embedding without importing ML libraries eagerly."""

from .embedding import E5TextEmbedder, TextEmbedder, TextEmbedding, TextEmbeddingError

__all__ = ["E5TextEmbedder", "TextEmbedder", "TextEmbedding", "TextEmbeddingError"]
