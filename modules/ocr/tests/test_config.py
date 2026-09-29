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
                "OCR_USE_DOC_ORIENTATION_CLASSIFY": "false",
                "OCR_USE_DOC_UNWARPING": "false",
                "OCR_USE_TEXTLINE_ORIENTATION": "true",
                "OCR_TEXT_DET_LIMIT_SIDE_LEN": "960",
                "OCR_TEXT_DET_LIMIT_TYPE": "max",
                "OCR_TEXT_DET_THRESH": "0.2",
                "OCR_TEXT_DET_BOX_THRESH": "0.4",
                "OCR_TEXT_DETECTION_MODEL_NAME": "PP-OCRv5_mobile_det",
                "OCR_TEXT_RECOGNITION_MODEL_NAME": "eslav_PP-OCRv5_mobile_rec",
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
        self.assertFalse(config.use_doc_orientation_classify)
        self.assertFalse(config.use_doc_unwarping)
        self.assertTrue(config.use_textline_orientation)
        self.assertEqual(config.text_det_limit_side_len, 960)
        self.assertEqual(config.text_det_limit_type, "max")
        self.assertEqual(config.text_det_thresh, 0.2)
        self.assertEqual(config.text_det_box_thresh, 0.4)
        self.assertEqual(config.text_detection_model_name, "PP-OCRv5_mobile_det")
        self.assertEqual(
            config.text_recognition_model_name, "eslav_PP-OCRv5_mobile_rec"
        )
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

    def test_from_env_rejects_invalid_detection_size(self) -> None:
        with patch.dict(
            os.environ,
            {"OCR_TEXT_DET_LIMIT_SIDE_LEN": "0"},
            clear=True,
        ):
            with self.assertRaises(ConfigurationError):
                OCRConfig.from_env()

    def test_from_env_rejects_invalid_detection_limit_type(self) -> None:
        with patch.dict(os.environ, {"OCR_TEXT_DET_LIMIT_TYPE": "unknown"}, clear=True):
            with self.assertRaises(ConfigurationError):
                OCRConfig.from_env()

    def test_from_env_rejects_invalid_detection_threshold(self) -> None:
        with patch.dict(
            os.environ,
            {"OCR_TEXT_DET_BOX_THRESH": "1.5"},
            clear=True,
        ):
            with self.assertRaises(ConfigurationError):
                OCRConfig.from_env()


if __name__ == "__main__":
    unittest.main()
