"""Model-independent OCR and retrieval metrics for A-E comparisons."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .models import OCRPrediction, OCRTruth


def normalize_text(value: str) -> str:
    """Normalize Unicode, case, and whitespace without deleting punctuation."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def levenshtein_distance(reference: str, prediction: str) -> int:
    """Compute edit distance with two rows of memory."""

    if len(reference) < len(prediction):
        reference, prediction = prediction, reference
    previous = list(range(len(prediction) + 1))
    for row_index, reference_char in enumerate(reference, start=1):
        current = [row_index]
        for column_index, prediction_char in enumerate(prediction, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column_index] + 1,
                    previous[column_index - 1]
                    + (reference_char != prediction_char),
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(reference: str, prediction: str) -> float:
    """Return normalized edit distance with explicit empty-truth behavior."""

    normalized_reference = normalize_text(reference)
    normalized_prediction = normalize_text(prediction)
    if not normalized_reference:
        return 0.0 if not normalized_prediction else 1.0
    return levenshtein_distance(
        normalized_reference, normalized_prediction
    ) / len(normalized_reference)


def exact_field_matches(
    truth: OCRTruth,
    prediction: OCRPrediction,
) -> dict[str, bool]:
    """Return exact normalized matches for requested wine fields."""

    matches = {
        "name": normalize_text(truth.name) == normalize_text(prediction.name),
        "producer": normalize_text(truth.producer)
        == normalize_text(prediction.producer),
        "year": normalize_text(truth.year) == normalize_text(prediction.year),
    }
    matches["all"] = all(matches.values())
    return matches


def _unit_vector(vector: np.ndarray, *, role: str) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{role} embedding contains non-finite values")
    norm = float(np.linalg.norm(array))
    if norm <= 0:
        raise ValueError(f"{role} embedding is a zero vector")
    return array / norm


def retrieval_rank(
    query: np.ndarray,
    expected_catalog_id: str,
    gallery: Mapping[str, np.ndarray],
) -> int:
    """Return one-based cosine rank in a deterministic fixed gallery."""

    if expected_catalog_id not in gallery:
        raise ValueError(
            f"Expected catalog id is absent from gallery: {expected_catalog_id}"
        )
    if not gallery:
        raise ValueError("Gallery must not be empty")
    query_unit = _unit_vector(query, role="Query")
    catalog_ids = sorted(gallery)
    units = []
    for catalog_id in catalog_ids:
        unit = _unit_vector(gallery[catalog_id], role=f"Gallery item {catalog_id}")
        if unit.shape != query_unit.shape:
            raise ValueError("Query and gallery embedding dimensions differ")
        units.append(unit)
    scores = np.asarray(units) @ query_unit
    order = np.argsort(-scores, kind="stable")
    expected_index = catalog_ids.index(expected_catalog_id)
    return int(np.flatnonzero(order == expected_index)[0]) + 1


def recall_at_k(rank: int, ks: Sequence[int] = (1, 5)) -> dict[int, bool]:
    """Convert a one-based rank into Recall@k hit flags."""

    if rank < 1:
        raise ValueError("Rank must be one-based and positive")
    if any(k < 1 for k in ks):
        raise ValueError("Recall cutoffs must be positive")
    return {int(k): rank <= int(k) for k in ks}


def paired_degradation_rate(
    baseline: Sequence[Any],
    candidate: Sequence[Any],
    *,
    lower_is_better: bool,
) -> dict[str, int | float]:
    """Measure paired regressions rather than only aggregate score changes."""

    if len(baseline) != len(candidate):
        raise ValueError("Paired metric inputs must have the same length")
    degraded = 0
    for baseline_value, candidate_value in zip(baseline, candidate, strict=True):
        if lower_is_better:
            is_degraded = candidate_value > baseline_value
        else:
            is_degraded = candidate_value < baseline_value
        degraded += int(is_degraded)
    count = len(baseline)
    return {
        "count": count,
        "degraded_count": degraded,
        "rate": degraded / count if count else 0.0,
    }

