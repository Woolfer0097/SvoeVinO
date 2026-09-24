"""Visual wine search by photo: DINOv2 embeddings stored in PostgreSQL/pgvector."""

from .application.validate_image import validate_image
from .contracts import (
    EmbeddingResult,
    ReferenceImageRecord,
    RetrievalRequest,
    SearchRequest,
    SearchResponse,
    ValidatedImage,
    WineCandidate,
)
from .infrastructure.storage.local_storage import LocalImageStorage

__all__ = [
    "LocalImageStorage",
    "EmbeddingResult",
    "ReferenceImageRecord",
    "RetrievalRequest",
    "SearchRequest",
    "SearchResponse",
    "ValidatedImage",
    "WineCandidate",
    "validate_image",
]
