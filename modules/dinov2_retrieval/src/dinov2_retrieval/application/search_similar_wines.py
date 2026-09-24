"""Use case: find the Top-K distinct wines most similar to a user photo."""

from __future__ import annotations

from collections.abc import Iterable

from ..config import get_raw_retrieval_limit
from ..contracts import ReferenceMatch, SearchRequest, SearchResponse, WineCandidate
from ..embedding.base import Embedder
from ..preprocessing.image_preprocessor import ImagePreprocessor
from ..retrieval.base import Retriever
from .create_embedding import create_embedding
from .validate_image import ImageStorage


def search_similar_wines(
    request: SearchRequest,
    retriever: Retriever,
    *,
    storage: ImageStorage | None = None,
    preprocessor: ImagePreprocessor | None = None,
    embedder: Embedder | None = None,
    raw_limit: int | None = None,
) -> SearchResponse:
    """Embed the query photo and return the best-matching wines.

    One wine may have several reference photos, so ``raw_limit``
    (``RAW_RETRIEVAL_LIMIT``, at least ``top_k``) nearest photos are fetched
    first, then grouped by wine_id; each wine is represented by its closest
    photo. If they cover fewer than ``top_k`` wines while more photos exist,
    the limit is doubled until ``top_k`` wines are found or the index is
    exhausted. The query photo is only embedded, never stored.
    """

    query = create_embedding(
        request.image_uri,
        storage=storage,
        preprocessor=preprocessor,
        embedder=embedder,
    )
    limit = max(get_raw_retrieval_limit(raw_limit), request.top_k)
    while True:
        matches = retriever.search_similar(
            query.embedding, limit=limit, model_name=query.model
        )
        candidates = select_top_wines(matches, request.top_k)
        if len(candidates) >= request.top_k or len(matches) < limit:
            break
        limit *= 2

    return SearchResponse(
        request_id=request.request_id,
        status="ok" if candidates else "no_results",
        model_name=query.model,
        query_embedding_dimension=query.dimension,
        candidates=candidates,
        message=(
            None
            if candidates
            else f"No reference photos are indexed for model {query.model}; "
            "run the index command first"
        ),
    )


def select_top_wines(
    matches: Iterable[ReferenceMatch], top_k: int
) -> list[WineCandidate]:
    """Keep the closest photo of every wine and return the ``top_k`` best wines."""

    best_by_wine: dict[str, ReferenceMatch] = {}
    for match in matches:
        best = best_by_wine.get(match.wine_id)
        if best is None or match.distance < best.distance:
            best_by_wine[match.wine_id] = match

    ranked = sorted(
        best_by_wine.values(), key=lambda match: (match.distance, match.wine_id)
    )
    return [
        WineCandidate(
            wine_id=match.wine_id,
            slug=match.slug,
            score=1.0 - match.distance,
            distance=match.distance,
            best_image_uri=match.image_uri,
        )
        for match in ranked[:top_k]
    ]
