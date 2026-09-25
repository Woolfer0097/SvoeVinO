from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


class LightweightImportTests(unittest.TestCase):
    def test_package_import_does_not_import_paddleocr(self) -> None:
        sys.modules.pop("paddleocr", None)

        importlib.import_module("wine_ocr")

        self.assertNotIn("paddleocr", sys.modules)

    def test_paddle_wrapper_import_and_init_are_lazy(self) -> None:
        sys.modules.pop("paddleocr", None)

        paddle_module = importlib.import_module("wine_ocr.engine.paddle")
        config_module = importlib.import_module("wine_ocr.config")
        paddle_module.PaddleOCREngine(config_module.OCRConfig())

        self.assertNotIn("paddleocr", sys.modules)


if __name__ == "__main__":
    unittest.main()
