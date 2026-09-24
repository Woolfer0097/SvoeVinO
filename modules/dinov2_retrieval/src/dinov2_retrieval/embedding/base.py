"""Base interface for image embedders."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from PIL import Image


class EmbeddingError(ValueError):
    """Raised when the model cannot be loaded or returns an invalid embedding."""


class Embedder(Protocol):
    """Convert a prepared image into an embedding vector."""

    @property
    def model_name(self) -> str:
        """Return the model identifier."""

        ...

    @property
    def device(self) -> str:
        """Return the runtime device identifier."""

        ...

    def embed(self, image: Image.Image) -> Sequence[float]:
        """Return an embedding for the supplied RGB image."""

        ...
