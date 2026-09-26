"""Use case: measure Recall@K of the visual search on labelled user photos."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

from ..contracts import (
    EvaluationError,
    EvaluationQuery,
    EvaluationReport,
    IncorrectQuery,
    SearchRequest,
)
from ..embedding.base import Embedder
from ..infrastructure.storage.local_storage import LocalImageStorage
from ..preprocessing.image_preprocessor import ImagePreprocessor
from ..retrieval.reference_repository import ReferenceRepository, RepositoryError
from .search_similar_wines import search_similar_wines
from .validate_image import ImageStorage

logger = logging.getLogger(__name__)

RECALL_LEVELS = (1, 5, 20)
NOT_IN_TOP_K = "expected wine is not in top-k"
WINE_NOT_INDEXED = "WineNotIndexedError"


def evaluate_retrieval(
    queries: Sequence[EvaluationQuery],
    repository: ReferenceRepository,
    *,
    top_k: int,
    recall_levels: Iterable[int] = RECALL_LEVELS,
    storage: ImageStorage | None = None,
    preprocessor: ImagePreprocessor | None = None,
    embedder: Embedder | None = None,
    raw_limit: int | None = None,
) -> EvaluationReport:
    """Search every query photo exactly like the ``search`` command and score it.

    Recall is reported for ``recall_levels`` and ``top_k``, so the search goes
    as deep as the largest of them. The model is created once for the run and
    query photos are only read, never stored. A broken photo or a query whose
    wine has no reference photos is recorded in ``errors`` and does not stop
    the run; database errors do.
    """

    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    image_storage = storage if storage is not None else LocalImageStorage()
    image_preprocessor = (
        preprocessor if preprocessor is not None else ImagePreprocessor()
    )
    if embedder is None:
        from ..embedding.dinov2_embedder import DinoV2Embedder

        image_embedder: Embedder = DinoV2Embedder()
    else:
        image_embedder = embedder

    levels = sorted({*recall_levels, top_k})
    search_depth = levels[-1]
    indexed_wine_ids = repository.get_wine_ids(image_embedder.model_name)

    total = len(queries)
    ranks: list[int | None] = []
    incorrect: list[IncorrectQuery] = []
    errors: list[EvaluationError] = []

    for position, query in enumerate(queries, start=1):
        if query.wine_id not in indexed_wine_ids:
            errors.append(
                _error(
                    query,
                    WINE_NOT_INDEXED,
                    f"wine_id {query.wine_id} has no reference photos indexed "
                    f"for model {image_embedder.model_name}",
                )
            )
            logger.warning(
                "[%d/%d] skipped %s: wine %s is not indexed",
                position,
                total,
                query.query_image_uri,
                query.wine_id,
            )
            continue

        try:
            response = search_similar_wines(
                SearchRequest(
                    request_id=f"evaluation-{position}",
                    image_uri=query.query_image_uri,
                    top_k=search_depth,
                ),
                repository,
                storage=image_storage,
                preprocessor=image_preprocessor,
                embedder=image_embedder,
                raw_limit=raw_limit,
            )
        except RepositoryError:
            raise
        except Exception as exc:  # one broken photo must not stop the run
            errors.append(_error(query, type(exc).__name__, str(exc)))
            logger.warning(
                "[%d/%d] failed %s: %s", position, total, query.query_image_uri, exc
            )
            continue

        predicted = [candidate.wine_id for candidate in response.candidates]
        rank = (
            predicted.index(query.wine_id) + 1 if query.wine_id in predicted else None
        )
        ranks.append(rank)
        if rank is None or rank > top_k:
            incorrect.append(
                IncorrectQuery(
                    query_image_uri=query.query_image_uri,
                    expected_wine_id=query.wine_id,
                    predicted_wine_ids=predicted[:top_k],
                    expected_rank=rank,
                    reason=NOT_IN_TOP_K,
                )
            )
        logger.info(
            "[%d/%d] %s %s: expected %s, rank %s",
            position,
            total,
            "hit" if rank is not None and rank <= top_k else "miss",
            query.query_image_uri,
            query.wine_id,
            rank if rank is not None else f">{search_depth}",
        )

    recall_at, correct_at = compute_recall(ranks, levels)
    processed = len(ranks)
    if not errors:
        status = "ok"
    elif processed:
        status = "partial"
    else:
        status = "failed"

    return EvaluationReport(
        status=status,
        model_name=image_embedder.model_name,
        queries_total=total,
        queries_processed=processed,
        queries_with_errors=len(errors),
        top_k=top_k,
        recall_at_k=recall_at[top_k],
        correct_queries=correct_at[top_k],
        recall_at=recall_at,
        correct_at=correct_at,
        indexed_wines=len(indexed_wine_ids),
        incorrect_queries=incorrect,
        errors=errors,
        message=(
            None
            if indexed_wine_ids
            else f"No reference photos are indexed for model "
            f"{image_embedder.model_name}; run the index command first"
        ),
    )


def compute_recall(
    ranks: Sequence[int | None], levels: Iterable[int]
) -> tuple[dict[int, float | None], dict[int, int]]:
    """Return Recall@K and the number of correct queries for every K.

    ``ranks`` holds the 1-based position of the correct wine among distinct
    wines for each processed query, or None when it was not found. Recall is
    None when there is no processed query to divide by.
    """

    recall_at: dict[int, float | None] = {}
    correct_at: dict[int, int] = {}
    for k in sorted(set(levels)):
        correct = sum(1 for rank in ranks if rank is not None and rank <= k)
        correct_at[k] = correct
        recall_at[k] = round(correct / len(ranks), 4) if ranks else None
    return recall_at, correct_at


def _error(query: EvaluationQuery, error_type: str, reason: str) -> EvaluationError:
    return EvaluationError(
        query_image_uri=query.query_image_uri,
        expected_wine_id=query.wine_id,
        error_type=error_type,
        reason=reason,
    )
