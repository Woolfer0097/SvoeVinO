"""Image validation scaffold for the future DINOv2 retrieval module."""

from .application.validate_image import validate_image
from .contracts import EmbeddingResult, RetrievalRequest, ValidatedImage
from .infrastructure.storage.local_storage import LocalImageStorage

__all__ = [
    "LocalImageStorage",
    "EmbeddingResult",
    "RetrievalRequest",
    "ValidatedImage",
    "validate_image",
]
