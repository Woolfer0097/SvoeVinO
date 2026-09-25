from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from wine_ocr.contracts import BoundingBox, Point, RecognizedTextBlock
from wine_ocr.postprocessing.names import extract_candidate_name


def _block(
    text: str,
    confidence: float,
    box: tuple[int, int, int, int],
    variant: str = "central_crop",
) -> RecognizedTextBlock:
    left, top, right, bottom = box
    return RecognizedTextBlock(
        text=text,
        confidence=confidence,
        bbox=BoundingBox(
            (
                Point(left, top),
                Point(right, top),
                Point(right, bottom),
                Point(left, bottom),
            )
        ),
        source_variant=variant,
    )


class CandidateNameTests(unittest.TestCase):
    def test_prefers_a_centered_two_line_name_over_generic_and_full_text(self) -> None:
        blocks = [
            _block("WRONG FULL TEXT", 0.99, (300, 100, 700, 250), "full"),
            _block("СЕМЕЙНАЯ ВИНОДЕЛЬНЯ", 0.99, (250, 100, 750, 200)),
            _block("KPACHAA", 0.93, (320, 300, 680, 380)),
            _block("СТРЕЛКА", 0.88, (330, 385, 670, 460)),
        ]

        candidate = extract_candidate_name(blocks, (1000, 1000))

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.value, "KPACHAA СТРЕЛКА")
        self.assertEqual(candidate.field_type, "name")
        self.assertEqual(candidate.metadata["source_variant"], "central_crop")
        self.assertEqual(candidate.metadata["text_block_count"], 2)

    def test_joins_fragments_on_the_same_line_in_reading_order(self) -> None:
        blocks = [
            _block("SELECTION", 0.99, (500, 300, 800, 380)),
            _block("NOBLE", 0.98, (200, 305, 490, 375)),
            _block("GOLUBITSKOE", 0.99, (200, 390, 800, 480)),
        ]

        candidate = extract_candidate_name(blocks, (1000, 1000))

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.value, "NOBLE SELECTION")

    def test_falls_back_to_full_pass_and_removes_year_from_name(self) -> None:
        blocks = [
            _block("2023", 0.99, (400, 400, 600, 500)),
            _block("MERLOT 2023", 0.92, (300, 300, 700, 380), "full"),
        ]

        candidate = extract_candidate_name(blocks, (1000, 1000))

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.value, "MERLOT")
        self.assertEqual(candidate.source_text, "MERLOT 2023")
        self.assertEqual(candidate.metadata["source_variant"], "full")

    def test_returns_none_without_plausible_name_text(self) -> None:
        blocks = [
            _block("2023", 0.99, (400, 400, 600, 500)),
            _block("АО", 0.99, (400, 500, 600, 600)),
            _block("LOW CONFIDENCE", 0.65, (400, 600, 600, 700)),
        ]

        self.assertIsNone(extract_candidate_name(blocks, (1000, 1000)))

    def test_ignores_regulatory_origin_lines(self) -> None:
        blocks = [
            _block("ЗГУ КУБАНЬ. АНАПА", 0.95, (300, 300, 700, 400)),
            _block("MERLOT", 0.9, (350, 500, 650, 580)),
        ]

        candidate = extract_candidate_name(blocks, (1000, 1000))

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.value, "MERLOT")


if __name__ == "__main__":
    unittest.main()
