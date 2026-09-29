"""Slug-level fusion; scores are ranking signals, not probabilities."""
from __future__ import annotations

import math


def scores_by_id(items: list[dict], allowed: set[str] | None = None) -> dict[str, float]:
    scores = {}
    for item in items:
        wine_id = str(item["id"])
        score = float(item["score"])
        if not wine_id or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid branch score")
        if allowed is not None and wine_id not in allowed:
            raise ValueError("Verifier returned an ID outside DINO shortlist")
        scores[wine_id] = max(scores.get(wine_id, 0), score)
    return dict(sorted(scores.items(), key=lambda item: -item[1])[:10])


def fuse(
    candidates: list[dict], visual: dict[str, float], ocr: dict[str, float],
    *, ocr_weight: float = .5, text_primary: bool = False,
):
    """Text leads when sufficient; sparse text only reranks visual candidates."""
    by_slug = {}
    if not math.isfinite(ocr_weight) or not 0 <= ocr_weight <= 1:
        raise ValueError("OCR weight must be between zero and one")
    visual_weight = 1 - ocr_weight
    for candidate in candidates:
        by_slug.setdefault(candidate["slug"], candidate)
    has_geometry = any(visual.get(slug, 0) > 0 for slug in by_slug)
    if text_primary and ocr:
        mode = "text_primary" if has_geometry else "text_primary_dino"
        supported = (visual.keys() | ocr.keys()) & by_slug.keys()
        if not has_geometry:
            supported |= {slug for slug, item in by_slug.items() if item.get("score") is not None}
        ranked = []
        for slug in supported:
            image_score = visual.get(slug, 0) if has_geometry else max(0, by_slug[slug].get("score") or 0)
            ranked.append((slug, visual_weight * image_score + ocr_weight * ocr.get(slug, 0)))
    elif has_geometry:
        mode = "weighted_visual" if visual.keys() & ocr.keys() else "visual_fallback"
        ranked = [(slug, visual_weight * score + ocr_weight * ocr.get(slug, 0))
                  for slug, score in visual.items() if slug in by_slug and score > 0]
    else:
        mode = "dino_fallback"
        ranked = [(slug, visual_weight * candidate["score"]) for slug, candidate in by_slug.items()
                  if candidate.get("score") is not None]
    order = {slug: index for index, slug in enumerate(by_slug)}
    ranked.sort(key=lambda item: (-item[1], order[item[0]]))
    result = []
    for slug, score in ranked:
        candidate = dict(by_slug[slug])
        candidate.update(
            score=score, visual_score=visual.get(slug), ocr_score=ocr.get(slug),
            dino_score=by_slug[slug]["score"],
        )
        # The fused score is not a cosine distance.
        candidate.pop("distance", None)
        result.append(candidate)
    return result, mode
