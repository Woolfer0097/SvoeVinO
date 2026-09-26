from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from wine_ocr.config import OCRConfig
from wine_ocr.embedding_comparison import EmbeddingComparisonError, EmbeddingMatch
from wine_ocr.exceptions import ImageDecodeError
from wine_ocr.text_processing import TextEmbedding


class FakeHTTPException(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


class FakeResponse:
    def __init__(self, content=None, *args, **kwargs) -> None:
        self.content = content


class FakeFastAPI:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def get(self, path: str):
        return lambda function: function

    def post(self, path: str):
        return lambda function: function


async def fake_run_in_threadpool(function, *args, **kwargs):
    return function(*args, **kwargs)


def load_rest_without_http_dependencies():
    fastapi = ModuleType("fastapi")
    fastapi.FastAPI = FakeFastAPI
    fastapi.File = lambda *args, **kwargs: None
    fastapi.HTTPException = FakeHTTPException
    fastapi.UploadFile = object

    responses = ModuleType("fastapi.responses")
    responses.JSONResponse = FakeResponse
    responses.PlainTextResponse = FakeResponse

    starlette = ModuleType("starlette")
    concurrency = ModuleType("starlette.concurrency")
    concurrency.run_in_threadpool = fake_run_in_threadpool

    path = SRC / "wine_ocr" / "entrypoints" / "rest.py"
    spec = importlib.util.spec_from_file_location("wine_ocr.entrypoints._rest_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {
        "fastapi": fastapi,
        "fastapi.responses": responses,
        "starlette": starlette,
        "starlette.concurrency": concurrency,
    }):
        spec.loader.exec_module(module)
    return module


class FakeUpload:
    filename = "label.jpg"

    async def read(self) -> bytes:
        return b"image bytes"


class FakeOCRRuntime:
    def __init__(self, text: str = "название 2022", error: Exception | None = None) -> None:
        self.text = text
        self.error = error
        self.received: bytes | None = None

    def run_ocr_from_bytes(self, data: bytes, config: OCRConfig):
        self.received = data
        if self.error is not None:
            raise self.error
        return SimpleNamespace(normalized_text=self.text, candidate_name=None)


class FakeTextEmbedder:
    def __init__(self) -> None:
        self.received: str | None = None

    def embed(self, text: str) -> TextEmbedding:
        self.received = text
        return TextEmbedding((0.6, 0.8), "intfloat/multilingual-e5-base")


class FakeRepository:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.received: tuple | None = None

    def find_nearest(self, values, model_name: str, limit: int):
        self.received = (values, model_name, limit)
        if self.error is not None:
            raise self.error
        return [EmbeddingMatch(42, 0.2)]


class RestMatchingTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rest = load_rest_without_http_dependencies()

    async def test_match_route_returns_top_ten_json_shape(self) -> None:
        runtime = FakeOCRRuntime()
        embedder = FakeTextEmbedder()
        repository = FakeRepository()
        with patch.multiple(self.rest, ocr_runtime=runtime, text_embedder=embedder,
                            embedding_repository=repository), \
             patch.object(self.rest.OCRConfig, "from_env", return_value=OCRConfig()):
            response = await self.rest.match_image(FakeUpload())

        self.assertEqual(response.content, {"top_10": {"42": 0.9}})
        self.assertEqual(runtime.received, b"image bytes")
        self.assertEqual(embedder.received, "название 2022")
        self.assertEqual(repository.received, ((0.6, 0.8), "intfloat/multilingual-e5-base", 10))

    async def test_match_route_maps_database_failure_to_503(self) -> None:
        repository = FakeRepository(EmbeddingComparisonError("DATABASE_URL is not configured"))
        with patch.multiple(self.rest, ocr_runtime=FakeOCRRuntime(),
                            text_embedder=FakeTextEmbedder(), embedding_repository=repository), \
             patch.object(self.rest.OCRConfig, "from_env", return_value=OCRConfig()):
            with self.assertRaises(FakeHTTPException) as error:
                await self.rest.match_image(FakeUpload())

        self.assertEqual(error.exception.status_code, 503)
        self.assertIn("DATABASE_URL", error.exception.detail)

    async def test_match_route_maps_bad_image_to_400(self) -> None:
        runtime = FakeOCRRuntime(error=ImageDecodeError("Invalid image"))
        with patch.multiple(self.rest, ocr_runtime=runtime,
                            text_embedder=FakeTextEmbedder(), embedding_repository=FakeRepository()), \
             patch.object(self.rest.OCRConfig, "from_env", return_value=OCRConfig()):
            with self.assertRaises(FakeHTTPException) as error:
                await self.rest.match_image(FakeUpload())

        self.assertEqual(error.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
