"""Use case: embed reference photos and store them for the vector search."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from ..contracts import (
    IndexingError,
    IndexingStats,
    ReferenceImage,
    ReferenceImageRecord,
)
from ..embedding.base import Embedder
from ..infrastructure.storage.local_storage import LocalImageStorage
from ..preprocessing.image_preprocessor import ImagePreprocessor
from ..retrieval.reference_repository import ReferenceRepository
from .create_embedding import create_embedding
from .validate_image import ImageStorage

logger = logging.getLogger(__name__)


def index_reference_images(
    references: Sequence[ReferenceImage],
    repository: ReferenceRepository,
    *,
    prune: bool = False,
    storage: ImageStorage | None = None,
    preprocessor: ImagePreprocessor | None = None,
    embedder: Embedder | None = None,
) -> IndexingStats:
    """Embed every reference photo exactly like a query photo and upsert it.

    The model is created once for the whole run. A broken photo is recorded in
    the stats and does not stop the run; repository errors do, because nothing
    can be saved after them. Upserting by image_uri makes reruns idempotent.

    With ``prune`` the database is synced to ``references``: stored photos of
    the same model that are not in the list are deleted. Photos that failed in
    this run keep their old rows, and nothing is deleted if no photo was
    processed, so a wrong path cannot empty the index.
    """

    image_storage = storage if storage is not None else LocalImageStorage()
    image_preprocessor = (
        preprocessor if preprocessor is not None else ImagePreprocessor()
    )
    if embedder is None:
        from ..embedding.dinov2_embedder import DinoV2Embedder

        image_embedder: Embedder = DinoV2Embedder()
    else:
        image_embedder = embedder

    total = len(references)
    inserted = updated = skipped = 0
    errors: list[IndexingError] = []
    indexed_paths: set[str] = set()

    for position, reference in enumerate(references, start=1):
        try:
            result = create_embedding(
                reference.image_uri,
                storage=image_storage,
                preprocessor=image_preprocessor,
                embedder=image_embedder,
            )
        except Exception as exc:  # one broken photo must not stop the run
            errors.append(
                IndexingError(
                    wine_id=reference.wine_id,
                    image_uri=reference.image_uri,
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            )
            logger.warning(
                "[%d/%d] failed %s: %s", position, total, reference.image_uri, exc
            )
            continue

        # Different spellings of one path (relative/absolute) resolve to one file.
        if result.path in indexed_paths:
            skipped += 1
            logger.info(
                "[%d/%d] skipped %s: already indexed in this run",
                position,
                total,
                result.path,
            )
            continue
        indexed_paths.add(result.path)

        outcome = repository.upsert_reference_embedding(
            ReferenceImageRecord(
                wine_id=reference.wine_id,
                slug=reference.slug,
                image_uri=result.path,
                model_name=result.model,
                embedding=result.embedding,
            )
        )
        if outcome == "inserted":
            inserted += 1
        else:
            updated += 1
        logger.info("[%d/%d] %s %s", position, total, outcome, result.path)

    processed = inserted + updated
    deleted = 0
    if prune and processed:
        keep = indexed_paths | {
            path
            for error in errors
            for path in (error.image_uri, _absolute(error.image_uri))
        }
        deleted = repository.delete_references_except(image_embedder.model_name, keep)
        logger.info("pruned %d reference photos that are not in the source", deleted)
    elif prune:
        logger.warning("prune skipped: no reference photo was processed")

    if not errors:
        status = "ok"
    elif processed:
        status = "partial"
    else:
        status = "failed"

    return IndexingStats(
        status=status,
        model_name=image_embedder.model_name,
        total=total,
        processed=processed,
        inserted=inserted,
        updated=updated,
        skipped=skipped,
        deleted=deleted,
        errors=len(errors),
        error_details=errors,
        reference_count=repository.get_reference_count(image_embedder.model_name),
    )


def _absolute(image_uri: str) -> str:
    """Stored paths are resolved; resolve an absolute input path the same way."""

    path = Path(image_uri)
    return str(path.resolve(strict=False)) if path.is_absolute() else image_uri
