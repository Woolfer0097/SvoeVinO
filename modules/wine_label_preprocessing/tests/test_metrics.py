from __future__ import annotations

import numpy as np
import pytest

from wine_label_preprocessing.metrics import (
    character_error_rate,
    exact_field_matches,
    normalize_text,
    paired_degradation_rate,
    retrieval_rank,
    recall_at_k,
)
from wine_label_preprocessing.models import OCRPrediction, OCRTruth


def test_text_normalization_is_unicode_case_and_whitespace_stable() -> None:
    assert normalize_text("  ＷＩＮＥ\nRéserve  ") == "wine réserve"


def test_cer_and_empty_truth() -> None:
    assert character_error_rate("wine", "wane") == pytest.approx(0.25)
    assert character_error_rate("", "") == 0.0
    assert character_error_rate("", "x") == 1.0


def test_exact_fields_are_reported_separately_and_jointly() -> None:
    truth = OCRTruth(
        full_text="Chateau Example 2020",
        name="Chateau Example",
        producer="Example Winery",
        year="2020",
    )
    prediction = OCRPrediction(
        full_text="chateau example 2021",
        name="  CHATEAU   EXAMPLE ",
        producer="Example Winery",
        year="2021",
    )

    matches = exact_field_matches(truth, prediction)

    assert matches == {
        "name": True,
        "producer": True,
        "year": False,
        "all": False,
    }


def test_recall_uses_fixed_cosine_gallery() -> None:
    gallery = {
        "a": np.array([1.0, 0.0], dtype=np.float32),
        "b": np.array([0.0, 1.0], dtype=np.float32),
    }

    rank = retrieval_rank(np.array([0.9, 0.1]), "a", gallery)

    assert rank == 1
    assert recall_at_k(rank, (1, 5)) == {1: True, 5: True}


def test_zero_embedding_is_rejected() -> None:
    with pytest.raises(ValueError, match="zero"):
        retrieval_rank(
            np.array([0.0, 0.0]),
            "a",
            {"a": np.array([1.0, 0.0])},
        )


def test_paired_degradation_is_relative_to_baseline() -> None:
    assert paired_degradation_rate(
        [0.1, 0.3, 0.2], [0.2, 0.2, 0.2], lower_is_better=True
    ) == {"count": 3, "degraded_count": 1, "rate": pytest.approx(1 / 3)}
    assert paired_degradation_rate(
        [True, True, False], [True, False, True], lower_is_better=False
    ) == {"count": 3, "degraded_count": 1, "rate": pytest.approx(1 / 3)}
