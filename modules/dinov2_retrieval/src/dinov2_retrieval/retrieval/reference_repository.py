"""Storage boundary for reference photo embeddings."""

from __future__ import annotations

from collections.abc import Collection
from typing import Literal, Protocol

from ..contracts import ReferenceImageRecord, ReferenceWine
from .base import Retriever

UpsertOutcome = Literal["inserted", "updated"]


class RepositoryError(RuntimeError):
    """Raised when the reference embedding storage fails."""


class ReferenceRepository(Retriever, Protocol):
    """Save reference embeddings and search among them."""

    def upsert_reference_embedding(
        self, record: ReferenceImageRecord
    ) -> UpsertOutcome:
        """Insert the record, or update the existing one with the same image_uri."""

        ...

    def get_reference_count(self, model_name: str | None = None) -> int:
        """Return how many reference photos are stored, optionally per model."""

        ...

    def delete_references_except(
        self, model_name: str, keep_image_uris: Collection[str]
    ) -> int:
        """Delete photos of ``model_name`` not in ``keep_image_uris``; return the count."""

        ...

    def list_references(
        self, model_name: str, limit: int, offset: int = 0
    ) -> list[ReferenceWine]:
        """Return indexed wines of ``model_name`` ordered by wine_id, with photos."""

        ...

    def get_wine_ids(self, model_name: str) -> set[str]:
        """Return every wine_id that has at least one photo of ``model_name``."""

        ...

    def close(self) -> None:
        """Release the underlying connection."""

        ...
