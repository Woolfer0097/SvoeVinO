"""Search existing ``wines.description_text_embedding`` rows without writes."""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


class EmbeddingComparisonError(RuntimeError):
    """Catalog vector search is unavailable or returned invalid data."""


@dataclass(frozen=True)
class EmbeddingMatch:
    wine_id: int
    cosine_distance: float


class EmbeddingRepository(Protocol):
    def find_nearest(
        self, values: Sequence[float], model_name: str, limit: int
    ) -> list[EmbeddingMatch]:
        """Find the closest catalog rows made by the same text model."""


SEARCH_SQL = """
SELECT id, description_text_embedding <=> %(query)s AS distance
FROM wines
WHERE description_text_embedding IS NOT NULL
  AND description_text_embedding_model = %(model_name)s
  AND vector_dims(description_text_embedding) = %(dimension)s
ORDER BY description_text_embedding <=> %(query)s, id
LIMIT %(limit)s
"""


class PostgresEmbeddingRepository:
    """Open a short-lived PostgreSQL connection for each read-only search."""

    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url

    def find_nearest(
        self, values: Sequence[float], model_name: str, limit: int
    ) -> list[EmbeddingMatch]:
        if len(values) == 0 or not all(math.isfinite(float(value)) for value in values):
            raise EmbeddingComparisonError("Query embedding is empty or invalid")
        if limit <= 0:
            raise EmbeddingComparisonError("Candidate limit must be positive")
        database_url = self.database_url or os.getenv("DATABASE_URL")
        if not database_url:
            raise EmbeddingComparisonError("DATABASE_URL is not configured")

        try:
            import numpy as np
            import psycopg
            from pgvector.psycopg import register_vector

            with psycopg.connect(
                database_url,
                autocommit=True,
                connect_timeout=10,
                options="-c default_transaction_read_only=on",
            ) as connection:
                register_vector(connection)
                rows = connection.execute(
                    SEARCH_SQL,
                    {
                        "query": np.asarray(values, dtype=np.float32),
                        "model_name": model_name,
                        "dimension": len(values),
                        "limit": limit,
                    },
                ).fetchall()
        except Exception as exc:
            raise EmbeddingComparisonError("Cannot search catalog text embeddings") from exc

        try:
            return [
                EmbeddingMatch(wine_id=int(wine_id), cosine_distance=float(distance))
                for wine_id, distance in rows
            ]
        except (TypeError, ValueError) as exc:
            raise EmbeddingComparisonError("Catalog search returned invalid rows") from exc
