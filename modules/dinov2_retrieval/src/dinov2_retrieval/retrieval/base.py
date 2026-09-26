"""Base interface for similar-image search backends."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ..contracts import ReferenceMatch


class Retriever(Protocol):
    """Search reference photo embeddings closest to a query embedding."""

    def search_similar(
        self,
        query_embedding: Sequence[float],
        limit: int,
        model_name: str,
    ) -> Sequence[ReferenceMatch]:
        """Return up to ``limit`` reference photos of ``model_name``, closest first."""

        ...
