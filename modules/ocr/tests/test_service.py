from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from wine_ocr.application.service import run_ocr_on_image
from wine_ocr.config import OCRConfig
from wine_ocr.contracts import BoundingBox, Point, RecognizedTextBlock


def _box(left: float, top: float, right: float, bottom: float) -> BoundingBox:
    return BoundingBox(
        (
            Point(left, top),
            Point(right, top),
            Point(right, bottom),
            Point(left, bottom),
        )
    )


class FakeImage:
    def __init__(self, width: int = 100, height: int = 80) -> None:
        self.size = (width, height)
        self.crop_box: tuple[int, int, int, int] | None = None

    def copy(self) -> "FakeImage":
        return self

    def crop(self, box: tuple[int, int, int, int]) -> "FakeImage":
        left, top, right, bottom = box
        cropped = FakeImage(right - left, bottom - top)
        cropped.crop_box = box
        return cropped


class MockEngine:
    name = "mock-engine"

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def recognize(
        self,
        image: FakeImage,
        source_variant: str,
    ) -> list[RecognizedTextBlock]:
        self.calls.append((source_variant, image.crop_box))
        if source_variant == "full":
            return [
                RecognizedTextBlock(
                    text="Wine X 2022",
                    confidence=0.8,
                    bbox=_box(5, 5, 40, 15),
                    source_variant=source_variant,
                )
            ]
        if source_variant == "central_crop":
            return [
                RecognizedTextBlock(
                    text="13,5% 0,75 л",
                    confidence=0.9,
                    bbox=_box(0, 0, 10, 10),
                    source_variant=source_variant,
                )
            ]
        return []


class EmptyEngine:
    name = "empty-engine"

    def recognize(
        self,
        image: FakeImage,
        source_variant: str,
    ) -> list[RecognizedTextBlock]:
        return []


class OCRServiceTests(unittest.TestCase):
    def test_run_ocr_on_image_uses_mock_engine_and_maps_crop_boxes(self) -> None:
        engine = MockEngine()
        config = OCRConfig(enable_central_crop=True, central_crop_fraction=0.5)

        result = run_ocr_on_image(FakeImage(), config=config, engine=engine)

        self.assertEqual(
            engine.calls,
            [
                ("full", None),
                ("central_crop", (25, 20, 75, 60)),
            ],
        )
        crop_block = next(
            block
            for block in result.text_blocks
            if block.source_variant == "central_crop"
        )
        self.assertIsNotNone(crop_block.bbox)
        self.assertEqual(crop_block.bbox.min_x, 25)
        self.assertEqual(crop_block.bbox.min_y, 20)

        values = {
            (candidate.field_type, candidate.normalized_value)
            for candidate in result.candidate_fields
        }
        self.assertIn(("year", "2022"), values)
        self.assertIn(("percentage", "13.5%"), values)
        self.assertIn(("volume", "0.75 l"), values)
        self.assertEqual(result.engine, "mock-engine")

    def test_run_ocr_on_image_warns_when_no_text_is_found(self) -> None:
        config = OCRConfig(enable_central_crop=False)

        result = run_ocr_on_image(FakeImage(), config=config, engine=EmptyEngine())

        self.assertEqual(result.raw_text, "")
        self.assertEqual(result.normalized_text, "")
        self.assertEqual(result.text_blocks, [])
        self.assertEqual(result.warnings, ["No text blocks were recognized"])


if __name__ == "__main__":
    unittest.main()
