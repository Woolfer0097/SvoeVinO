from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from wine_ocr.application.runtime import CachedOCRRuntime
from wine_ocr.config import OCRConfig


class FakeEngine:
    name = "fake-engine"


class CachedOCRRuntimeTests(unittest.TestCase):
    def test_get_engine_reuses_backend_for_same_engine_settings(self) -> None:
        runtime = CachedOCRRuntime()
        engine = FakeEngine()

        with patch(
            "wine_ocr.application.runtime.create_engine",
            return_value=engine,
        ) as create_engine:
            first = runtime.get_engine(OCRConfig(output_dir=Path("outputs-a")))
            second = runtime.get_engine(OCRConfig(output_dir=Path("outputs-b")))

        self.assertIs(first, engine)
        self.assertIs(second, engine)
        create_engine.assert_called_once()

    def test_get_engine_recreates_backend_when_engine_settings_change(self) -> None:
        runtime = CachedOCRRuntime()
        cpu_engine = FakeEngine()
        gpu_engine = FakeEngine()

        with patch(
            "wine_ocr.application.runtime.create_engine",
            side_effect=[cpu_engine, gpu_engine],
        ) as create_engine:
            first = runtime.get_engine(OCRConfig(device="cpu"))
            second = runtime.get_engine(OCRConfig(device="gpu"))

        self.assertIs(first, cpu_engine)
        self.assertIs(second, gpu_engine)
        self.assertEqual(create_engine.call_count, 2)

    def test_get_engine_recreates_backend_when_paddle_profile_changes(self) -> None:
        runtime = CachedOCRRuntime()
        with patch(
            "wine_ocr.application.runtime.create_engine",
            side_effect=[FakeEngine(), FakeEngine()],
        ) as create_engine:
            runtime.get_engine(OCRConfig(use_doc_unwarping=True))
            runtime.get_engine(OCRConfig(use_doc_unwarping=False))

        self.assertEqual(create_engine.call_count, 2)

    def test_run_ocr_from_bytes_passes_cached_engine_to_pipeline(self) -> None:
        runtime = CachedOCRRuntime()
        engine = FakeEngine()
        config = OCRConfig()
        sentinel = object()

        with (
            patch("wine_ocr.application.runtime.create_engine", return_value=engine),
            patch(
                "wine_ocr.application.runtime.run_ocr_from_bytes",
                return_value=sentinel,
            ) as run_ocr,
        ):
            first = runtime.run_ocr_from_bytes(b"image-a", config)
            second = runtime.run_ocr_from_bytes(b"image-b", config)

        self.assertIs(first, sentinel)
        self.assertIs(second, sentinel)
        self.assertEqual(run_ocr.call_count, 2)
        self.assertIs(run_ocr.call_args_list[0].kwargs["engine"], engine)
        self.assertIs(run_ocr.call_args_list[1].kwargs["engine"], engine)


if __name__ == "__main__":
    unittest.main()
