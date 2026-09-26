from __future__ import annotations

import math
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from wine_ocr.embedding_comparison import (
    EmbeddingComparisonError,
    EmbeddingMatch,
    PostgresEmbeddingRepository,
    compare_embedding,
)
from wine_ocr.text_processing import TextEmbedding


class FakeRepository:
    def __init__(self, matches: list[EmbeddingMatch]) -> None:
        self.matches = matches
        self.calls: list[tuple[tuple[float, ...], str, int]] = []

    def find_nearest(
        self, values: tuple[float, ...], model_name: str, limit: int
    ) -> list[EmbeddingMatch]:
        self.calls.append((values, model_name, limit))
        return self.matches


class EmbeddingComparisonTests(unittest.TestCase):
    def test_returns_ranked_ids_and_scores_from_matching_model(self) -> None:
        embedding = TextEmbedding((0.6, 0.8), "intfloat/multilingual-e5-base")
        repository = FakeRepository(
            [EmbeddingMatch(12, 0.4), EmbeddingMatch(11, 0.1), EmbeddingMatch(12, 0.5)]
        )

        result = compare_embedding(embedding, repository)

        self.assertEqual(repository.calls, [((0.6, 0.8), embedding.model_name, 10)])
        self.assertEqual(list(result["top_10"]), ["11", "12"])
        self.assertAlmostEqual(result["top_10"]["11"], 0.95)
        self.assertAlmostEqual(result["top_10"]["12"], 0.8)

    def test_rejects_invalid_distance(self) -> None:
        repository = FakeRepository([EmbeddingMatch(1, math.nan)])
        with self.assertRaises(EmbeddingComparisonError):
            compare_embedding(TextEmbedding((1.0,), "test-model"), repository)

    def test_missing_database_url_fails_before_connecting(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(EmbeddingComparisonError, "DATABASE_URL"):
                PostgresEmbeddingRepository().find_nearest((1.0,), "test-model", 10)


if __name__ == "__main__":
    unittest.main()
