"""PostgreSQL/pgvector implementation of the reference repository."""

from __future__ import annotations

from collections.abc import Collection, Sequence

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from ...contracts import ReferenceImageRecord, ReferenceMatch, ReferenceWine
from ...retrieval.reference_repository import RepositoryError, UpsertOutcome
from .connection import connect, initialize_schema

UPSERT_SQL = """
INSERT INTO reference_images (wine_id, slug, image_uri, model_name, embedding)
VALUES (%(wine_id)s, %(slug)s, %(image_uri)s, %(model_name)s, %(embedding)s)
ON CONFLICT (image_uri) DO UPDATE SET
    wine_id = EXCLUDED.wine_id,
    slug = EXCLUDED.slug,
    model_name = EXCLUDED.model_name,
    embedding = EXCLUDED.embedding,
    updated_at = NOW()
RETURNING (xmax = 0) AS inserted
"""

# Cosine distance; similarity score is 1 - distance.
SEARCH_SQL = """
SELECT wine_id, slug, image_uri, embedding <=> %(query)s AS distance
FROM reference_images
WHERE model_name = %(model_name)s
ORDER BY embedding <=> %(query)s
LIMIT %(limit)s
"""

COUNT_SQL = "SELECT COUNT(*) FROM reference_images"
COUNT_BY_MODEL_SQL = (
    "SELECT COUNT(*) FROM reference_images WHERE model_name = %(model_name)s"
)
DELETE_EXCEPT_SQL = """
DELETE FROM reference_images
WHERE model_name = %(model_name)s AND NOT (image_uri = ANY(%(keep)s))
"""
LIST_WINES_SQL = """
SELECT wine_id, min(slug), array_agg(image_uri ORDER BY image_uri)
FROM reference_images
WHERE model_name = %(model_name)s
GROUP BY wine_id
ORDER BY wine_id
LIMIT %(limit)s OFFSET %(offset)s
"""
WINE_IDS_SQL = (
    "SELECT DISTINCT wine_id FROM reference_images WHERE model_name = %(model_name)s"
)


class PostgresReferenceRepository:
    """Keep reference embeddings in ``reference_images`` and search them with pgvector."""

    def __init__(self, connection: psycopg.Connection) -> None:
        self._connection = connection

    @classmethod
    def connect(cls, database_url: str | None = None) -> PostgresReferenceRepository:
        """Connect, create the schema if needed and register the vector type."""

        connection = connect(database_url)
        try:
            initialize_schema(connection)
            register_vector(connection)
        except psycopg.Error as exc:
            connection.close()
            raise RepositoryError(f"Cannot register the vector type: {exc}") from exc
        except BaseException:
            connection.close()
            raise
        return cls(connection)

    def upsert_reference_embedding(
        self, record: ReferenceImageRecord
    ) -> UpsertOutcome:
        params = {
            "wine_id": record.wine_id,
            "slug": record.slug,
            "image_uri": record.image_uri,
            "model_name": record.model_name,
            "embedding": _to_vector(record.embedding),
        }
        try:
            (inserted,) = self._connection.execute(UPSERT_SQL, params).fetchone()
        except psycopg.Error as exc:
            raise RepositoryError(
                f"Cannot save reference embedding for {record.image_uri}: {exc}"
            ) from exc
        return "inserted" if inserted else "updated"

    def get_reference_count(self, model_name: str | None = None) -> int:
        if model_name is None:
            query, params = COUNT_SQL, None
        else:
            query, params = COUNT_BY_MODEL_SQL, {"model_name": model_name}
        try:
            (count,) = self._connection.execute(query, params).fetchone()
        except psycopg.Error as exc:
            raise RepositoryError(f"Cannot count reference embeddings: {exc}") from exc
        return int(count)

    def delete_references_except(
        self, model_name: str, keep_image_uris: Collection[str]
    ) -> int:
        params = {"model_name": model_name, "keep": sorted(keep_image_uris)}
        try:
            cursor = self._connection.execute(DELETE_EXCEPT_SQL, params)
        except psycopg.Error as exc:
            raise RepositoryError(f"Cannot delete stale reference photos: {exc}") from exc
        return max(cursor.rowcount, 0)

    def list_references(
        self, model_name: str, limit: int, offset: int = 0
    ) -> list[ReferenceWine]:
        params = {"model_name": model_name, "limit": limit, "offset": offset}
        try:
            rows = self._connection.execute(LIST_WINES_SQL, params).fetchall()
        except psycopg.Error as exc:
            raise RepositoryError(f"Cannot list indexed wines: {exc}") from exc
        return [
            ReferenceWine(wine_id=wine_id, slug=slug, image_uris=list(image_uris))
            for wine_id, slug, image_uris in rows
        ]

    def get_wine_ids(self, model_name: str) -> set[str]:
        try:
            rows = self._connection.execute(
                WINE_IDS_SQL, {"model_name": model_name}
            ).fetchall()
        except psycopg.Error as exc:
            raise RepositoryError(f"Cannot list indexed wines: {exc}") from exc
        return {wine_id for (wine_id,) in rows}

    def search_similar(
        self,
        query_embedding: Sequence[float],
        limit: int,
        model_name: str,
    ) -> list[ReferenceMatch]:
        params = {
            "query": _to_vector(query_embedding),
            "model_name": model_name,
            "limit": limit,
        }
        try:
            rows = self._connection.execute(SEARCH_SQL, params).fetchall()
        except psycopg.Error as exc:
            raise RepositoryError(f"Cannot search reference embeddings: {exc}") from exc
        return [
            ReferenceMatch(
                wine_id=wine_id,
                slug=slug,
                image_uri=image_uri,
                distance=float(distance),
            )
            for wine_id, slug, image_uri, distance in rows
        ]

    def close(self) -> None:
        self._connection.close()


def _to_vector(values: Sequence[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)
