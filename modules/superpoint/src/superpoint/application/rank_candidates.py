"""Score a query photo against catalog ids from the reference database."""

from __future__ import annotations

from collections.abc import Sequence

from ..contracts import CandidateScore, VerificationRequest
from ..matching.base import PhotoMatcher
from .validate_image import ImageStorage
from .verify_photos import GeometryVerifier, verify_photos
from ..infrastructure.database.reference_catalog import (
    CandidatesNotFoundError,
    ReferenceCatalog,
)


def rank_candidates(
    query_uri: str,
    candidate_ids: Sequence[str],
    *,
    query_storage: ImageStorage,
    reference_storage: ImageStorage,
    catalog: ReferenceCatalog,
    matcher: PhotoMatcher,
    geometry: GeometryVerifier,
    photos: dict[str, list[str]] | None = None,
) -> list[CandidateScore]:
    """Compare ``query_uri`` with every reference photo of each candidate id.

    A wine can have several photos. Its score is the best inlier ratio among
    the photos that pass the verification thresholds; a rejected pair scores 0.
    The list is sorted by score descending, then by the request order.

    ``photos`` lets the caller resolve the catalog before taking the inference
    lock. When it is omitted, this function reads the catalog itself.
    """

    resolved = photos if photos is not None else catalog.image_uris(candidate_ids)
    missing = [wine_id for wine_id in candidate_ids if not resolved.get(wine_id)]
    if missing:
        raise CandidatesNotFoundError(missing)

    order = {wine_id: index for index, wine_id in enumerate(candidate_ids)}
    scored: list[CandidateScore] = []
    for wine_id in candidate_ids:
        best = 0.0
        for image_uri in resolved[wine_id]:
            result = verify_photos(
                VerificationRequest(query_uri=query_uri, reference_uri=image_uri),
                storage=query_storage,
                reference_storage=reference_storage,
                matcher=matcher,
                geometry=geometry,
            )
            pair_score = result.inlier_ratio if result.verified else 0.0
            best = max(best, pair_score)
        scored.append(CandidateScore(id=wine_id, score=best))
    scored.sort(key=lambda item: (-item.score, order[item.id]))
    return scored
