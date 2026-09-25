from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from wine_ocr.contracts import BoundingBox, CandidateField, OCRResult, Point
from wine_ocr.contracts import RecognizedTextBlock, TextBlock
from wine_ocr.postprocessing.blocks import build_text_blocks
from wine_ocr.postprocessing.fields import extract_candidate_fields
from wine_ocr.postprocessing.normalization import normalize_text, tokenize_text
from wine_ocr.reporting import render_txt_report


def _box(left: float, top: float, right: float, bottom: float) -> BoundingBox:
    return BoundingBox(
        (
            Point(left, top),
            Point(right, top),
            Point(right, bottom),
            Point(left, bottom),
        )
    )


class PostprocessingTests(unittest.TestCase):
    def test_normalize_text_is_conservative(self) -> None:
        text = "  Вино\u00a0Wine X  2023  13,5%  "

        self.assertEqual(normalize_text(text), "вино wine x 2023 13,5%")
        self.assertEqual(
            tokenize_text(normalize_text(text)),
            ["вино", "wine", "x", "2023", "13,5%"],
        )

    def test_extract_candidate_fields(self) -> None:
        block = TextBlock(
            text="Wine X 2023 13,5% 0,75 л 750 мл",
            normalized_text=normalize_text("Wine X 2023 13,5% 0,75 л 750 мл"),
            confidence=0.91,
            bbox=None,
        )

        candidates = extract_candidate_fields([block], block.normalized_text)
        values = {
            (candidate.field_type, candidate.normalized_value)
            for candidate in candidates
        }

        self.assertIn(("year", "2023"), values)
        self.assertIn(("percentage", "13.5%"), values)
        self.assertIn(("volume", "0.75 l"), values)
        self.assertIn(("volume", "750 ml"), values)

    def test_build_text_blocks_sorts_and_deduplicates(self) -> None:
        blocks = [
            RecognizedTextBlock("Bottom", 0.8, _box(0, 40, 30, 50), "full"),
            RecognizedTextBlock("Top", 0.7, _box(0, 0, 20, 10), "full"),
            RecognizedTextBlock("Top", 0.95, _box(1, 1, 21, 11), "central_crop"),
        ]

        result = build_text_blocks(blocks)

        self.assertEqual([block.normalized_text for block in result], ["top", "bottom"])
        self.assertEqual(result[0].confidence, 0.95)
        self.assertEqual(result[0].source_variant, "central_crop")

    def test_render_txt_report_contains_required_sections(self) -> None:
        result = OCRResult(
            raw_text="Wine X\n2023",
            normalized_text="wine x 2023",
            text_blocks=[
                TextBlock("Wine X", "wine x", 0.9, None, "full"),
                TextBlock("2023", "2023", 0.8, None, "full"),
            ],
            tokens=["wine", "x", "2023"],
            candidate_fields=[
                CandidateField("name", "Wine X", "wine x", "Wine X", 0.9),
                CandidateField("year", "2023", "2023", "2023", 0.8),
            ],
            engine="fake",
            processing_time_ms=12.345,
            candidate_name="Wine X",
        )

        report = render_txt_report(result)

        self.assertIn("Raw lines", report)
        self.assertIn("Normalized text", report)
        self.assertIn("Candidate name", report)
        self.assertIn("- Wine X", report)
        self.assertIn("Candidate years", report)
        self.assertIn("- 2023", report)


if __name__ == "__main__":
    unittest.main()
