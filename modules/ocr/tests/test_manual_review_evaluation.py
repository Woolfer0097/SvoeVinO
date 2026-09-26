from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REVIEW_TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(REVIEW_TOOLS))

from evaluate_manual_review import (
    contains_phrase,
    evaluate_image,
    evaluate_run,
    parse_ground_truth,
)
from run_match_review import validate_match_response
from run_manual_review import prepare_run


class ManualReviewEvaluationTests(unittest.TestCase):
    def test_match_validator_rejects_non_mapping_response(self) -> None:
        with self.assertRaises(ValueError):
            validate_match_response([{"top_10": {}}], None)

    def test_review_runner_skips_marker_and_numbers_supplied_photos(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            photos = root / "photos"
            photos.mkdir()
            (photos / "README.txt").write_text("put your photos here", encoding="utf-8")
            (photos / "b.webp").write_bytes(b"second")
            (photos / "a.jpg").write_bytes(b"first")

            run_dir, manifest = prepare_run(photos, root / "outputs")

            self.assertEqual([entry["source"] for entry in manifest["images"]], ["a.jpg", "b.webp"])
            self.assertEqual([entry["image"] for entry in manifest["images"]], ["image1.jpg", "image2.webp"])
            self.assertEqual((run_dir / "image1.jpg").read_bytes(), b"first")
            self.assertEqual((run_dir / "image2.webp").read_bytes(), b"second")

    def test_evaluate_run_uses_generated_files_without_private_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            (run_dir / "manifest.json").write_text(
                json.dumps({"images": [{"image": "image1.webp", "source": "sample.webp"}]}),
                encoding="utf-8",
            )
            (run_dir / "image1_output.json").write_text(
                json.dumps({"result": {
                    "candidate_name": "пример",
                    "normalized_text": "пример 2020",
                    "candidate_fields": [{"field_type": "year", "value": "2020"}],
                    "text_blocks": [{"source_variant": "full"}],
                }}),
                encoding="utf-8",
            )
            truth = parse_ground_truth('image1: { name: "пример", years: "2020", other: "пример" }')

            summary = evaluate_run(truth, run_dir)["summary"]

            self.assertEqual(summary["name_candidate_exact"], 1)
            self.assertEqual(summary["years_found"], 1)
            self.assertEqual(summary["other_found"], 1)

    def test_parse_ground_truth_keeps_optional_years_and_phrases(self) -> None:
        truth = parse_ground_truth(
            'image1: { name: "Белая львица", other: "арaтти, российское вино" }\n'
            'image2: { name: "Вино", years: "2020, 2021", other: "брют" }'
        )

        self.assertEqual(truth[1]["years"], [])
        self.assertEqual(truth[1]["other"], ["арaтти", "российское вино"])
        self.assertEqual(truth[2]["years"], ["2020", "2021"])

    def test_phrase_match_is_strict_and_uses_word_boundaries(self) -> None:
        self.assertTrue(contains_phrase("GOLUBITSKOE ESTATE- 2024", "golubitskoe estate"))
        self.assertFalse(contains_phrase("Ультракюве", "ультра кюве"))
        self.assertFalse(contains_phrase("brutal", "brut"))

    def test_evaluate_image_keeps_missing_and_unexpected_years_separate(self) -> None:
        result = {
            "candidate_name": "NOBLE SELECTION",
            "normalized_text": "golubitskoe estate- red blend 2029",
            "candidate_fields": [
                {"field_type": "year", "value": "2029"},
                {"field_type": "name", "value": "NOBLE SELECTION"},
            ],
            "text_blocks": [{"source_variant": "central_crop"}],
        }
        expected = {
            "name": "golubitskoe estate",
            "years": ["2019"],
            "other": ["red blend", "брют"],
        }
        entry = {"image": "image9.webp", "source": "source.webp", "elapsed_seconds": 50.0}

        evaluated = evaluate_image(expected, result, entry)

        self.assertFalse(evaluated["matches"]["name_candidate_exact"])
        self.assertTrue(evaluated["matches"]["name_in_normalized_text"])
        self.assertEqual(evaluated["matches"]["years_missing"], ["2019"])
        self.assertEqual(evaluated["matches"]["years_unexpected"], ["2029"])
        self.assertEqual(
            [item["in_normalized_text"] for item in evaluated["matches"]["other"]],
            [True, False],
        )


if __name__ == "__main__":
    unittest.main()
