from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from wine_ocr.text_processing import E5TextEmbedder, TextEmbeddingError, prepare_embedding_text
from wine_ocr.text_processing.embedding import split_query_text


class WordTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[str]:
        tokens = text.split()
        return ["<s>", *tokens, "</s>"] if add_special_tokens else tokens


class TextProcessingTests(unittest.TestCase):
    def test_long_text_is_split_without_losing_words(self) -> None:
        text = "alpha beta gamma delta epsilon"
        chunks = split_query_text(text, WordTokenizer(), max_tokens=5)

        self.assertEqual(chunks, ["alpha beta", "gamma delta", "epsilon"])
        self.assertEqual(" ".join(chunks), text)

    def test_oversized_single_word_fails_instead_of_truncating(self) -> None:
        with self.assertRaises(TextEmbeddingError):
            split_query_text("alpha", WordTokenizer(), max_tokens=3)

    def test_constructing_embedder_does_not_load_model(self) -> None:
        embedder = E5TextEmbedder()
        self.assertIsNone(embedder._model)
        self.assertIsNone(embedder._tokenizer)

    def test_query_repairs_ocr_lookalikes_without_losing_years(self) -> None:
        query = prepare_embedding_text("дehиcob 3akat camapa 2023", "3AKAT")
        self.assertEqual(query, "закат денисов закат самара 2023")

    def test_query_preserves_latin_titles_and_all_ocr_words(self) -> None:
        query = prepare_embedding_text("velvet season 2020", "VELVET SEASON")
        self.assertEqual(query, "velvet season velvet season 2020")

    def test_query_repairs_central_wine_name(self) -> None:
        query = prepare_embedding_text(
            "denisov camapa kpachaa стрелка ry6in h 0", "KPACHAA СТРЕЛКА"
        )
        self.assertEqual(
            query, "краснаа стрелка denisov самара краснаа стрелка ry6in h 0"
        )


if __name__ == "__main__":
    unittest.main()
