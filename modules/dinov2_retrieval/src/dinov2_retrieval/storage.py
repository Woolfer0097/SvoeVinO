"""Backward-compatible facade for the moved storage and application layers."""

from .application.validate_image import validate_image
from .config import SUPPORTED_IMAGE_MIME_TYPES
from .infrastructure.storage.local_storage import (
    CorruptedImageError,
    ImageNotFoundError,
    ImageOutsideDataRootError,
    ImageTooLargeError,
    ImageValidationError,
    LocalImageStorage,
    UnsupportedImageFormatError,
)

SUPPORTED_MIME_TYPES = SUPPORTED_IMAGE_MIME_TYPES

__all__ = [
    "CorruptedImageError",
    "ImageNotFoundError",
    "ImageOutsideDataRootError",
    "ImageTooLargeError",
    "ImageValidationError",
    "LocalImageStorage",
    "SUPPORTED_MIME_TYPES",
    "UnsupportedImageFormatError",
    "validate_image",
]
