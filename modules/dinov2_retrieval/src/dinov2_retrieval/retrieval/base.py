"""Base interface for future similar-image search backends."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class Retriever(Protocol):
    """Search an image index using a query embedding."""

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int,
    ) -> Sequence[object]:
        """Return up to ``top_k`` similar-image results."""

        ...
