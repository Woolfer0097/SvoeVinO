from __future__ import annotations

import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from evaluate_manual_review import contains_phrase, evaluate_image, parse_ground_truth


class ManualReviewEvaluationTests(unittest.TestCase):
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
