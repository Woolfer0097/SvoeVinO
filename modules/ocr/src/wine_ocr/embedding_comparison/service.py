"""Convert the closest catalog embeddings into the public Top-10 response."""

from __future__ import annotations

import math

from ..text_processing import TextEmbedding
from .repository import EmbeddingComparisonError, EmbeddingRepository


def compare_embedding(
    embedding: TextEmbedding,
    repository: EmbeddingRepository,
    *,
    top_k: int = 10,
) -> dict[str, dict[str, float]]:
    """Return distinct catalog row IDs ordered by descending similarity."""

    if top_k <= 0:
        raise EmbeddingComparisonError("Candidate limit must be positive")
    matches = repository.find_nearest(embedding.values, embedding.model_name, top_k)
    if any(not math.isfinite(match.cosine_distance) for match in matches):
        raise EmbeddingComparisonError("Catalog search returned a non-finite distance")
    ranked: dict[str, float] = {}
    for match in sorted(matches, key=lambda item: (item.cosine_distance, item.wine_id)):
        if match.cosine_distance < -1e-6 or match.cosine_distance > 2 + 1e-6:
            raise EmbeddingComparisonError("Catalog search returned an invalid cosine distance")
        key = str(match.wine_id)
        if key in ranked:
            continue
        # Cosine distance is in [0, 2]. Rescale to [0, 1]; this is not a probability.
        ranked[key] = max(0.0, min(1.0, 1.0 - match.cosine_distance / 2.0))
        if len(ranked) == top_k:
            break
    return {"top_10": ranked}
