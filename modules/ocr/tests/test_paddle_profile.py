from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from wine_ocr.config import OCRConfig
from wine_ocr.engine.paddle import PaddleOCREngine


class PaddleProfileTests(unittest.TestCase):
    def test_constructor_passes_explicit_profile_to_paddleocr(self) -> None:
        paddle_cls = Mock(return_value=object())
        config = OCRConfig(
            device="cpu",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_det_limit_side_len=960,
            text_det_thresh=0.2,
            text_det_box_thresh=0.4,
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="eslav_PP-OCRv5_mobile_rec",
        )
        engine = PaddleOCREngine(config)

        with patch(
            "wine_ocr.engine.paddle.import_module",
            return_value=SimpleNamespace(PaddleOCR=paddle_cls),
        ):
            engine._create_client()

        paddle_cls.assert_called_once_with(
            lang="ru",
            device="cpu",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_det_limit_side_len=960,
            text_det_thresh=0.2,
            text_det_box_thresh=0.4,
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="eslav_PP-OCRv5_mobile_rec",
        )

    def test_predict_is_preferred_over_deprecated_ocr_alias(self) -> None:
        engine = PaddleOCREngine(OCRConfig())
        client = SimpleNamespace(predict=Mock(return_value="result"), ocr=Mock())
        image = object()

        self.assertEqual(engine._run_client(client, image), "result")
        client.predict.assert_called_once_with(image)
        client.ocr.assert_not_called()


if __name__ == "__main__":
    unittest.main()
