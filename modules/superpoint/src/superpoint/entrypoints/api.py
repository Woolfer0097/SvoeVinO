"""Long-running HTTP service with Swagger UI: the matcher is loaded once at start.

Candidate ranking can be run from Swagger (http://localhost:8001/docs) by
uploading a query photo and a JSON list of catalog ids. The upload is not kept.
Reference photos are read from the DINOv2 PostgreSQL catalog.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse
from PIL import Image
from pydantic import BaseModel

from ..application.rank_candidates import rank_candidates
from ..application.verify_photos import GeometryVerifier
from ..config import ConfigurationError, get_max_image_size_bytes
from ..contracts import CandidateScore
from ..infrastructure.database.reference_catalog import (
    CandidatesNotFoundError,
    CatalogError,
    PostgresReferenceCatalog,
    ReferenceCatalog,
)
from ..infrastructure.storage.local_storage import (
    CorruptedImageError,
    ImageNotFoundError,
    ImageOutsideDataRootError,
    ImageTooLargeError,
    ImageValidationError,
    LocalImageStorage,
    UnsupportedImageFormatError,
)
from ..infrastructure.storage.upload_storage import temporary_image
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
MAX_CANDIDATES = 20
# One max-sized photo, a short JSON id list, and multipart framing.
_UPLOAD_OVERHEAD_BYTES = 64 * 1024

TAG_VERIFY = "Проверка"
TAG_SERVICE = "Служебное"

OPENAPI_TAGS = [
    {
        "name": TAG_VERIFY,
        "description": "Сравнить фото запроса с эталонами из каталога DINOv2. "
        "Оценка — доля inlier после RANSAC, но только если пара прошла пороги; "
        "иначе 0. У вина с несколькими фото берётся лучшая.",
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
Проверка фото запроса против списка кандидатов из каталога DINOv2.
SuperPoint извлекает точки, LightGlue строит соответствия, OpenCV RANSAC
считает, какая доля соответствий лежит на одной гомографии. Модель
загружается один раз при старте сервиса.

Эталонные фото сервис читает из PostgreSQL (`reference_images`) по `wine_id`
и открывает `image_uri` внутри `DATA_ROOT`. Загруженное фото не сохраняется.

`score` — доля inlier (от 0 до 1) у лучшего эталонного фото этого id, если
пара прошла пороги `MIN_MATCHES`, `MIN_INLIERS` и `MIN_INLIER_RATIO`. Иначе
0: две точки из двух не становятся единицей. Список отсортирован по `score`
по убыванию. Низкий score — обычный ответ **200**, а не ошибка.

**Быстрая проверка:** **Проверка → `POST /verify`** — файл `query` и поле
`candidates` с JSON-массивом id (не больше 20), затем `Execute`.
"""


class HealthStatus(BaseModel):
    """Liveness payload: the process is serving and the matcher is loaded."""

    status: str
    model: str
    device: str


def max_upload_body_bytes() -> int:
    """Upper bound for one query photo plus the candidate list."""

    return get_max_image_size_bytes() + _UPLOAD_OVERHEAD_BYTES


def parse_candidate_ids(raw: str) -> list[str]:
    """Parse wine ids from a JSON array or a comma-separated list.

    Swagger sends a text field. A pasted array is often wrapped in extra
    quotes, so the raw value is not valid JSON until that layer is removed.
    """

    parsed = _decode_candidate_list(raw.strip().lstrip("\ufeff"))
    if not parsed:
        raise ValueError(
            "candidates must be a JSON array of wine ids or a comma-separated list"
        )
    if len(parsed) > MAX_CANDIDATES:
        raise ValueError(
            f"candidates must contain at most {MAX_CANDIDATES} ids"
        )
    if any(not item.strip() for item in parsed):
        raise ValueError("candidates must not contain empty ids")
    ids = [item.strip() for item in parsed]
    if len(set(ids)) != len(ids):
        raise ValueError("candidates must not contain duplicate ids")
    return ids


def _decode_candidate_list(text: str) -> list[str] | None:
    parsed = _loads_id_list(text)
    if parsed is not _INVALID_JSON:
        return parsed
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        parsed = _loads_id_list(text[1:-1].strip())
        if parsed is not _INVALID_JSON:
            return parsed
    quoted = re.findall(r'"([^"\\]+)"', text)
    if quoted and "[" in text:
        return quoted
    if "[" in text or "{" in text:
        return None
    pieces = [piece.strip().strip("'\"") for piece in re.split(r"[,\n]", text)]
    pieces = [piece for piece in pieces if piece]
    return pieces or None


_INVALID_JSON = object()


def _loads_id_list(text: str) -> list[str] | None | object:
    """Return ids, None for JSON that is not a string list, or a sentinel."""

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        unescaped = text.replace('\\"', '"')
        if unescaped == text:
            return _INVALID_JSON
        try:
            parsed = json.loads(unescaped)
        except json.JSONDecodeError:
            return _INVALID_JSON
    if isinstance(parsed, str):
        return _loads_id_list(parsed.strip())
    if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
        return parsed
    return None


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
    catalog: ReferenceCatalog | None = None,
    reference_storage: LocalImageStorage | None = None,
) -> FastAPI:
    """Build the API; collaborators are injectable for tests.

    The matcher is loaded when the service starts, so the first request is as
    fast as the others and a missing model fails the start, not a user request.
    The catalog is opened per request, so a down database does not block health.
    """

    matcher_provider = MatcherProvider(matcher)
    geometry_box: list[GeometryVerifier | None] = [geometry]
    reference_catalog = catalog if catalog is not None else PostgresReferenceCatalog()
    inference_lock = Lock()

    def catalog_storage() -> LocalImageStorage:
        if reference_storage is not None:
            return reference_storage
        return LocalImageStorage()

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
        title="SuperPoint — проверка кандидатов",
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

    @app.exception_handler(CandidatesNotFoundError)
    async def missing_candidates_handler(
        request: Request, exc: CandidatesNotFoundError
    ) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(CatalogError)
    async def catalog_error_handler(
        request: Request, exc: CatalogError
    ) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    app.add_middleware(LimitUploadSize)

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    @app.post(
        "/verify",
        tags=[TAG_VERIFY],
        response_model=list[CandidateScore],
        summary="Сравнить фото запроса со списком кандидатов",
    )
    def verify(
        query: UploadFile = File(
            description="фото запроса: jpg, jpeg, png или webp"
        ),
        candidates: str = Form(
            description="JSON-массив wine_id или список через запятую, не больше 20. "
            "В Swagger массив можно вставить как есть, с переносами строк.",
            examples=[
                "shepot, agrolayn_mountain_eagle_cabernet_sauvignon_kaberne_sovinon_krasnoe_suhoe_135_c2ea2996ff"
            ],
        ),
    ) -> list[CandidateScore]:
        """Фото запроса не сохраняется. Эталоны читаются из PostgreSQL по id
        и сравниваются по очереди. Ответ — `id` и `score`, по убыванию score.
        """

        try:
            candidate_ids = parse_candidate_ids(candidates)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        with temporary_image(
            query.file,
            filename=query.filename,
            content_type=query.content_type,
        ) as (upload_storage, query_uri):
            upload_storage.validate(query_uri)
            return _rank(query_uri, candidate_ids, upload_storage)

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

    def _rank(
        query_uri: str,
        candidate_ids: Sequence[str],
        query_storage: LocalImageStorage,
    ) -> list[CandidateScore]:
        if geometry_box[0] is None:
            raise HTTPException(status_code=503, detail="Matcher is not loaded")
        # Resolve files before the matcher lock so a slow database does not
        # block other comparisons.
        photos = reference_catalog.image_uris(candidate_ids)
        with inference_lock:
            return rank_candidates(
                query_uri,
                candidate_ids,
                query_storage=query_storage,
                reference_storage=catalog_storage(),
                catalog=reference_catalog,
                matcher=matcher_provider.get(),
                geometry=geometry_box[0],
                photos=photos,
            )

    return app
