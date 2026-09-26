from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from wine_ocr.text_processing import E5TextEmbedder, TextEmbeddingError
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


if __name__ == "__main__":
    unittest.main()
