from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from wine_ocr.application.matching import match_image_from_bytes
from wine_ocr.config import OCRConfig
from wine_ocr.embedding_comparison import EmbeddingMatch
from wine_ocr.text_processing import TextEmbedding


class FakeOCRRuntime:
    def __init__(self, text: str, candidate_name: str | None = None) -> None:
        self.text = text
        self.candidate_name = candidate_name
        self.calls: list[bytes] = []

    def run_ocr_from_bytes(self, data: bytes, config: OCRConfig):
        self.calls.append(data)
        return SimpleNamespace(normalized_text=self.text, candidate_name=self.candidate_name)


class FakeTextEmbedder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, text: str) -> TextEmbedding:
        self.calls.append(text)
        return TextEmbedding((0.6, 0.8), "intfloat/multilingual-e5-base")


class FakeRepository:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[float, ...], str, int]] = []

    def find_nearest(self, values, model_name: str, limit: int):
        self.calls.append((values, model_name, limit))
        return [EmbeddingMatch(42, 0.2)]


class MatchingTests(unittest.TestCase):
    def test_evidence_confidence_excludes_prices(self):
        runtime = FakeOCRRuntime("1298 каберне совиньон")
        base = runtime.run_ocr_from_bytes
        def enriched(data, config):
            result = base(data, config)
            result.text_blocks = [
                SimpleNamespace(normalized_text="1298", confidence=1.0),
                SimpleNamespace(normalized_text="каберне совиньон", confidence=.8),
            ]
            return result
        runtime.run_ocr_from_bytes = enriched
        result = match_image_from_bytes(
            b"image", config=OCRConfig(), ocr_runtime=runtime,
            text_embedder=FakeTextEmbedder(), repository=FakeRepository(), include_evidence=True,
        )
        self.assertAlmostEqual(result["evidence"]["text_confidence"], .8)

    def test_optional_evidence_carries_only_confident_non_foundation_years(self):
        runtime = FakeOCRRuntime("wine 2023 since 1900")
        base = runtime.run_ocr_from_bytes

        def enriched(data, config):
            result = base(data, config)
            result.candidate_fields = [
                SimpleNamespace(field_type="year", value="2023", confidence=.95, source_text="2023"),
                SimpleNamespace(field_type="year", value="2022", confidence=.5, source_text="2022"),
                SimpleNamespace(field_type="year", value="1900", confidence=.99, source_text="since 1900"),
            ]
            return result

        runtime.run_ocr_from_bytes = enriched
        result = match_image_from_bytes(
            b"image", config=OCRConfig(), ocr_runtime=runtime,
            text_embedder=FakeTextEmbedder(), repository=FakeRepository(),
            include_evidence=True,
        )
        self.assertEqual(result["evidence"]["years"], [2023])
        self.assertEqual(result["top_10"], {"42": .9})

    def test_full_pipeline_hands_ocr_text_to_embedding_and_comparison(self) -> None:
        ocr = FakeOCRRuntime("красная стрелка 2023", "КРАСНАЯ СТРЕЛКА")
        embedder = FakeTextEmbedder()
        repository = FakeRepository()

        result = match_image_from_bytes(
            b"image", config=OCRConfig(), ocr_runtime=ocr,
            text_embedder=embedder, repository=repository,
        )

        self.assertEqual(ocr.calls, [b"image"])
        self.assertEqual(embedder.calls, ["красная стрелка красная стрелка 2023"])
        self.assertEqual(repository.calls, [((0.6, 0.8), "intfloat/multilingual-e5-base", 10)])
        self.assertEqual(result, {"top_10": {"42": 0.9}})

    def test_empty_ocr_text_does_not_embed_or_query_database(self) -> None:
        embedder = FakeTextEmbedder()
        repository = FakeRepository()

        result = match_image_from_bytes(
            b"image", config=OCRConfig(), ocr_runtime=FakeOCRRuntime(" "),
            text_embedder=embedder, repository=repository,
        )

        self.assertEqual(result, {"top_10": {}})
        self.assertEqual(embedder.calls, [])
        self.assertEqual(repository.calls, [])


if __name__ == "__main__":
    unittest.main()
