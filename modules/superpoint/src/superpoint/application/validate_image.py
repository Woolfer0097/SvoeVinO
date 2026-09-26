"""Use case for obtaining and validating one local image."""

from __future__ import annotations

from typing import Protocol

from ..contracts import ValidatedImage
from ..infrastructure.storage.local_storage import LocalImageStorage


class ImageStorage(Protocol):
    """Storage boundary used by image validation."""

    def validate(self, image_uri: str) -> ValidatedImage:
        """Obtain and validate an image identified by its URI."""

        ...


def validate_image(
    image_uri: str,
    storage: ImageStorage | None = None,
) -> ValidatedImage:
    """Validate one image without matching it."""

    image_storage = storage if storage is not None else LocalImageStorage()
    return image_storage.validate(image_uri)
