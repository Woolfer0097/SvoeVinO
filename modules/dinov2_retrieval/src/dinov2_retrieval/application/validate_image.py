"""Use case for obtaining and validating a query image."""

from __future__ import annotations

from typing import Protocol

from ..contracts import RetrievalRequest, ValidatedImage
from ..infrastructure.storage.local_storage import LocalImageStorage


class ImageStorage(Protocol):
    """Storage boundary used by the image validation use case."""

    def validate(self, image_uri: str) -> ValidatedImage:
        """Obtain and validate an image identified by its URI."""

        ...


def validate_image(
    request: RetrievalRequest,
    storage: ImageStorage | None = None,
) -> ValidatedImage:
    """Coordinate image retrieval and validation without creating an embedding."""

    image_storage = storage if storage is not None else LocalImageStorage()
    return image_storage.validate(request.image_uri)
