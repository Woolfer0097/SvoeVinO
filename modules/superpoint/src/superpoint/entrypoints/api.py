"""Long-running HTTP service with Swagger UI: the matcher is loaded once at start.

Pair verification can be run from Swagger (http://localhost:8001/docs) by
uploading two photos. Uploads are not kept.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse
from PIL import Image
from pydantic import BaseModel

from ..application.verify_photos import GeometryVerifier, verify_photos
from ..config import ConfigurationError, get_max_image_size_bytes
from ..contracts import VerificationRequest, VerificationResult
from ..infrastructure.storage.local_storage import (
    CorruptedImageError,
    ImageNotFoundError,
    ImageOutsideDataRootError,
    ImageTooLargeError,
    ImageValidationError,
    LocalImageStorage,
    UnsupportedImageFormatError,
)
from ..infrastructure.storage.upload_storage import temporary_pair
from ..matching.base import PhotoMatcher
from ..preprocessing.image_preprocessor import ImagePreprocessingError

logger = logging.getLogger(__name__)

ERROR_STATUS_CODES: dict[type[Exception], int] = {
    ImageNotFoundError: 404,
    ImageOutsideDataRootError: 400,
    UnsupportedImageFormatError: 415,
    ImageTooLargeError: 413,
    CorruptedImageError: 422,
    ImagePreprocessingError: 422,
}

WARMUP_IMAGE_SIZE = (64, 64)
# Two max-sized photos plus multipart boundaries and part headers.
_UPLOAD_OVERHEAD_BYTES = 64 * 1024

TAG_VERIFY = "Проверка"
TAG_SERVICE = "Служебное"

OPENAPI_TAGS = [
    {
        "name": TAG_VERIFY,
        "description": "Сравнить фото запроса с эталоном. Совпадение принимается "
        "только если достаточно соответствий SuperPoint+LightGlue лежат на одной "
        "гомографии.",
    },
    {
        "name": TAG_SERVICE,
        "description": "Проверка, что сервис запущен и модель загружена.",
    },
]

SWAGGER_UI_PARAMETERS = {
    "tryItOutEnabled": True,
    "displayRequestDuration": True,
    "docExpansion": "list",
    "defaultModelsExpandDepth": -1,
}

DESCRIPTION = """
Проверка пары фотографий: SuperPoint извлекает точки, LightGlue строит
соответствия, OpenCV RANSAC принимает пару только если они согласованы одной
гомографией. Модель загружается один раз при старте сервиса.

`verified: false` — это обычный ответ **200**, а не ошибка: проверка дошла до
вердикта, и пара не совпала.

**Быстрая проверка:** **Проверка → `POST /verify`** — выберите два файла и
нажмите `Execute`. В ответе `query_path` и `reference_path` — имена
загруженных файлов, а не пути на диске: фото не сохраняются.
"""


class HealthStatus(BaseModel):
    """Liveness payload: the process is serving and the matcher is loaded."""

    status: str
    model: str
    device: str


def upload_label(filename: str | None, fallback: str) -> str:
    """Return the uploaded file name, never a directory path."""

    name = Path(filename or "").name
    return name or fallback


def max_upload_body_bytes() -> int:
    """Upper bound for one pair request, including multipart framing."""

    return 2 * get_max_image_size_bytes() + _UPLOAD_OVERHEAD_BYTES


class MatcherProvider:
    """Load SuperPoint+LightGlue once and reuse it for every request."""

    def __init__(self, matcher: PhotoMatcher | None = None) -> None:
        self._matcher = matcher
        self._lock = Lock()

    @property
    def loaded(self) -> PhotoMatcher | None:
        """Return the matcher if it is already loaded, without starting a load."""

        return self._matcher

    def get(self) -> PhotoMatcher:
        if self._matcher is None:
            with self._lock:
                if self._matcher is None:
                    from ..matching.superpoint_lightglue import (
                        SuperPointLightGlueMatcher,
                    )

                    self._matcher = SuperPointLightGlueMatcher()
        return self._matcher


class LimitUploadSize:
    """Reject a request body larger than two configured images plus framing.

    Starlette otherwise spools the whole multipart body before the handler
    reads it. Content-Length is rejected before that read; a missing or lying
    length is capped while the body is consumed.
    """

    def __init__(self, app: object) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive, send) -> None:
        if scope["type"] != "http" or scope.get("path") != "/verify":
            await self.app(scope, receive, send)
            return
        try:
            limit = max_upload_body_bytes()
        except ConfigurationError as exc:
            await _send_json(send, 500, str(exc))
            return
        declared = _content_length(scope)
        if declared is not None and declared > limit:
            await _send_json(send, 413, _too_large_detail(limit))
            return

        chunks: list[bytes] = []
        total = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            body = message.get("body", b"")
            total += len(body)
            if total > limit:
                await _send_json(send, 413, _too_large_detail(limit))
                return
            chunks.append(body)
            if not message.get("more_body", False):
                break

        payload = b"".join(chunks)
        sent = False

        async def replay():
            nonlocal sent
            if sent:
                return {"type": "http.request", "body": b"", "more_body": False}
            sent = True
            return {"type": "http.request", "body": payload, "more_body": False}

        await self.app(scope, replay, send)


def _content_length(scope: dict) -> int | None:
    for key, value in scope.get("headers", []):
        if key.lower() == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None


def _too_large_detail(limit: int) -> str:
    return f"Upload is too large; maximum request body is {limit} bytes"


async def _send_json(send, status_code: int, detail: str) -> None:
    body = JSONResponse(status_code=status_code, content={"detail": detail}).body
    await send(
        {
            "type": "http.response.start",
            "status": status_code,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def create_app(
    matcher: PhotoMatcher | None = None,
    geometry: GeometryVerifier | None = None,
) -> FastAPI:
    """Build the API; collaborators are injectable for tests.

    The matcher is loaded when the service starts, so the first request is as
    fast as the others and a missing model fails the start, not a user request.
    """

    matcher_provider = MatcherProvider(matcher)
    geometry_box: list[GeometryVerifier | None] = [geometry]
    inference_lock = Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        loaded = matcher_provider.get()
        # One warm-up pass: the first real inference is otherwise several
        # times slower while the runtime initializes its kernels.
        blank = Image.new("RGB", WARMUP_IMAGE_SIZE, color=(128, 128, 128))
        loaded.match(blank, blank)
        if geometry_box[0] is None:
            from ..verification.homography import HomographyVerifier

            geometry_box[0] = HomographyVerifier()
        logger.info("matcher %s is loaded on %s", loaded.model_name, loaded.device)
        yield

    app = FastAPI(
        title="SuperPoint — проверка пары фото",
        description=DESCRIPTION,
        version="0.1.0",
        lifespan=lifespan,
        openapi_tags=OPENAPI_TAGS,
        swagger_ui_parameters=SWAGGER_UI_PARAMETERS,
    )

    @app.exception_handler(ImageValidationError)
    @app.exception_handler(ImagePreprocessingError)
    async def image_error_handler(request: Request, exc: Exception) -> JSONResponse:
        status_code = ERROR_STATUS_CODES.get(type(exc), 400)
        return JSONResponse(status_code=status_code, content={"detail": str(exc)})

    @app.exception_handler(ConfigurationError)
    async def configuration_error_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    app.add_middleware(LimitUploadSize)

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    @app.post(
        "/verify",
        tags=[TAG_VERIFY],
        response_model=VerificationResult,
        summary="Проверить загруженную пару фото",
    )
    def verify(
        query: UploadFile = File(
            description="фото запроса: jpg, jpeg, png или webp"
        ),
        reference: UploadFile = File(
            description="эталон: jpg, jpeg, png или webp"
        ),
    ) -> VerificationResult:
        """Оба файла проходят те же проверки, что фото в `DATA_ROOT`, и **не
        сохраняются**: они лежат во временной папке только на время запроса.

        `query_path` и `reference_path` в ответе — имена загруженных файлов,
        не пути на диске. `verified: false` тоже возвращается с кодом 200.
        """

        with temporary_pair(
            query.file,
            reference.file,
            query_filename=query.filename,
            query_content_type=query.content_type,
            reference_filename=reference.filename,
            reference_content_type=reference.content_type,
        ) as (upload_storage, query_uri, reference_uri):
            result = _verify(query_uri, reference_uri, upload_storage)
        return result.model_copy(
            update={
                "query_path": upload_label(query.filename, "query"),
                "reference_path": upload_label(reference.filename, "reference"),
            }
        )

    @app.get(
        "/health",
        tags=[TAG_SERVICE],
        response_model=HealthStatus,
        summary="Модель загружена",
    )
    def health() -> HealthStatus:
        """Docker healthcheck: сервис отвечает только после загрузки матчера.
        В ответе имя модели и устройство, на котором она работает."""

        loaded = matcher_provider.loaded
        if loaded is None:
            raise HTTPException(status_code=503, detail="Matcher is not loaded")
        return HealthStatus(
            status="ok",
            model=loaded.model_name,
            device=loaded.device,
        )

    def _verify(
        query_uri: str,
        reference_uri: str,
        search_storage: LocalImageStorage,
    ) -> VerificationResult:
        if geometry_box[0] is None:
            raise HTTPException(status_code=503, detail="Matcher is not loaded")
        with inference_lock:
            return verify_photos(
                VerificationRequest(
                    query_uri=query_uri,
                    reference_uri=reference_uri,
                ),
                storage=search_storage,
                matcher=matcher_provider.get(),
                geometry=geometry_box[0],
            )

    return app
