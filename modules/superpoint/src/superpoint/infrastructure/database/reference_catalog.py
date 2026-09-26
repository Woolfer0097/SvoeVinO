"""Read reference photo paths from the DINOv2 PostgreSQL catalog."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol


class CatalogError(RuntimeError):
    """Raised when the reference catalog cannot be read."""


class CandidatesNotFoundError(LookupError):
    """Raised when one or more requested ids have no reference photo."""

    def __init__(self, missing_ids: Sequence[str]) -> None:
        self.missing_ids = list(missing_ids)
        missing = ", ".join(self.missing_ids)
        super().__init__(f"No reference photo for: {missing}")


class ReferenceCatalog(Protocol):
    """Lookup of reference image paths by wine id."""

    def image_uris(self, wine_ids: Sequence[str]) -> dict[str, list[str]]:
        """Return image paths grouped by wine id. Missing ids are omitted."""

        ...


class PostgresReferenceCatalog:
    """Read ``reference_images`` without importing the retrieval package."""

    def __init__(
        self,
        database_url: str | None = None,
        connect: Callable | None = None,
    ) -> None:
        self._database_url = database_url
        self._connect = connect

    def image_uris(self, wine_ids: Sequence[str]) -> dict[str, list[str]]:
        if not wine_ids:
            return {}
        url = self._database_url or _database_url()
        if self._connect is not None:
            with self._connect(url) as connection:
                rows = _fetch(connection, wine_ids)
        else:
            try:
                import psycopg
            except ImportError as exc:
                raise CatalogError("psycopg is not installed") from exc
            try:
                with psycopg.connect(url, connect_timeout=5) as connection:
                    rows = _fetch(connection, wine_ids)
            except psycopg.Error as exc:
                raise CatalogError(f"Cannot read reference photos: {exc}") from exc
        grouped: dict[str, list[str]] = {}
        for wine_id, image_uri in rows:
            grouped.setdefault(wine_id, []).append(image_uri)
        return grouped


def _database_url() -> str:
    from ...config import get_database_url

    return get_database_url()


def _fetch(connection, wine_ids: Sequence[str]):
    return connection.execute(
        """
        SELECT wine_id, image_uri
        FROM reference_images
        WHERE wine_id = ANY(%s)
        ORDER BY wine_id, image_uri
        """,
        (list(wine_ids),),
    ).fetchall()
