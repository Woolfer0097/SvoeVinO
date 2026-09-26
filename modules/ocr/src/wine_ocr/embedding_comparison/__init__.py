"""Read-only PostgreSQL search for catalog text embeddings."""

from .repository import (
    EmbeddingComparisonError,
    EmbeddingMatch,
    EmbeddingRepository,
    PostgresEmbeddingRepository,
)
from .service import compare_embedding

__all__ = [
    "EmbeddingComparisonError",
    "EmbeddingMatch",
    "EmbeddingRepository",
    "PostgresEmbeddingRepository",
    "compare_embedding",
]
