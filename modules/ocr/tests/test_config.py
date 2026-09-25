from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from wine_ocr.config import OCRConfig
from wine_ocr.exceptions import ConfigurationError


class OCRConfigTests(unittest.TestCase):
    def test_from_env_parses_supported_values(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OCR_ENGINE": "paddleocr",
                "OCR_DEVICE": "cpu",
                "OCR_PADDLE_LANG": "ru",
                "OCR_EXPECTED_LANGUAGES": "ru,en",
                "OCR_ENABLE_CENTRAL_CROP": "false",
                "OCR_CENTRAL_CROP_FRACTION": "0.5",
                "OCR_MAX_UPLOAD_SIZE_BYTES": "1024",
                "OCR_OUTPUT_DIR": "custom-output",
                "OCR_REPORT_RETENTION": "true",
            },
            clear=True,
        ):
            config = OCRConfig.from_env()

        self.assertEqual(config.engine, "paddleocr")
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.paddle_lang, "ru")
        self.assertEqual(config.expected_languages, ("ru", "en"))
        self.assertFalse(config.enable_central_crop)
        self.assertEqual(config.central_crop_fraction, 0.5)
        self.assertEqual(config.max_upload_size_bytes, 1024)
        self.assertEqual(config.output_dir, Path("custom-output"))
        self.assertTrue(config.report_retention)

    def test_from_env_rejects_invalid_fraction(self) -> None:
        with patch.dict(
            os.environ,
            {"OCR_CENTRAL_CROP_FRACTION": "1.5"},
            clear=True,
        ):
            with self.assertRaises(ConfigurationError):
                OCRConfig.from_env()

    def test_from_env_rejects_invalid_bool(self) -> None:
        with patch.dict(
            os.environ,
            {"OCR_REPORT_RETENTION": "maybe"},
            clear=True,
        ):
            with self.assertRaises(ConfigurationError):
                OCRConfig.from_env()


if __name__ == "__main__":
    unittest.main()
