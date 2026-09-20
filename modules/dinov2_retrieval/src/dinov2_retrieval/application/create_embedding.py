"""Application use case for creating an image embedding."""

from __future__ import annotations

from pathlib import Path

from ..contracts import EmbeddingResult
from ..embedding.base import Embedder
from ..infrastructure.storage.local_storage import LocalImageStorage
from ..preprocessing.image_preprocessor import ImagePreprocessor
from .validate_image import ImageStorage


def create_embedding(
    image_uri: str,
    storage: ImageStorage | None = None,
    preprocessor: ImagePreprocessor | None = None,
    embedder: Embedder | None = None,
) -> EmbeddingResult:
    """Validate, prepare and embed one image without external persistence."""

    image_storage = storage if storage is not None else LocalImageStorage()
    image_preprocessor = (
        preprocessor if preprocessor is not None else ImagePreprocessor()
    )
    validated_image = image_storage.validate(image_uri)

    if embedder is None:
        from ..embedding.dinov2_embedder import DinoV2Embedder

        image_embedder: Embedder = DinoV2Embedder()
    else:
        image_embedder = embedder

    with image_preprocessor.open_rgb(Path(validated_image.path)) as image:
        embedding = [float(value) for value in image_embedder.embed(image)]

    return EmbeddingResult(
        path=str(validated_image.path),
        model=image_embedder.model_name,
        dimension=len(embedding),
        device=image_embedder.device,
        embedding=embedding,
    )
