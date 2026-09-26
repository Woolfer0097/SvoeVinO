"""Long-running HTTP service with Swagger UI: the model is loaded once at start.

Everything the CLI does can be run from Swagger (http://localhost:8000/docs):
search by an uploaded photo, indexing, listing and viewing reference photos,
Recall@K evaluation and the full health check.
"""

from __future__ import annotations

import io
import logging
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, closing
from threading import Lock

from fastapi import (
    Body,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from PIL import Image
from pydantic import BaseModel, Field, model_validator

from ..application.check_health import DatabaseInspector, check_health
from ..application.create_embedding import create_embedding
from ..application.evaluate_retrieval import evaluate_retrieval
from ..application.index_reference_images import index_reference_images
from ..application.search_similar_wines import search_similar_wines
from ..application.validate_image import ImageStorage, validate_image
from ..config import ConfigurationError, get_default_top_k, resolve_data_path
from ..contracts import (
    EmbeddingResult,
    EvaluationReport,
    HealthReport,
    ImageRequest,
    IndexingStats,
    ReferenceList,
    RetrievalRequest,
    SearchRequest,
    SearchResponse,
    ValidatedImage,
)
from ..embedding.base import Embedder, EmbeddingError
from ..infrastructure.csv_manifest import ManifestError
from ..infrastructure.evaluation_manifest import read_evaluation_queries
from ..infrastructure.reference_manifest import load_references
from ..infrastructure.storage.local_storage import (
    CorruptedImageError,
    ImageNotFoundError,
    ImageOutsideDataRootError,
    ImageTooLargeError,
    ImageValidationError,
    LocalImageStorage,
    UnsupportedImageFormatError,
)
from ..infrastructure.storage.upload_storage import temporary_upload
from ..preprocessing.image_preprocessor import (
    ImagePreprocessingError,
    ImagePreprocessor,
)
from ..retrieval.reference_repository import ReferenceRepository, RepositoryError

logger = logging.getLogger(__name__)

ERROR_STATUS_CODES: dict[type[Exception], int] = {
    ImageNotFoundError: 404,
    ImageOutsideDataRootError: 400,
    UnsupportedImageFormatError: 415,
    ImageTooLargeError: 413,
    CorruptedImageError: 422,
    ImagePreprocessingError: 422,
}

RepositoryFactory = Callable[[], ReferenceRepository]
WARMUP_IMAGE_SIZE = (224, 224)
PREVIEW_JPEG_QUALITY = 85

TAG_SEARCH = "Поиск"
TAG_REFERENCES = "Эталоны"
TAG_EVALUATION = "Оценка качества"
TAG_SERVICE = "Служебное"

OPENAPI_TAGS = [
    {
        "name": TAG_SEARCH,
        "description": "Загрузите фото бутылки — сервис вернёт Top-K самых похожих "
        "разных вин из проиндексированных эталонов.",
    },
    {
        "name": TAG_REFERENCES,
        "description": "Индексация эталонных фото из `data/reference`, список "
        "загруженных вин и просмотр любой фотографии из `DATA_ROOT`.",
    },
    {
        "name": TAG_EVALUATION,
        "description": "Recall@1/5/20 на размеченных фото из "
        "`data/evaluation/queries.csv`.",
    },
    {
        "name": TAG_SERVICE,
        "description": "Проверка работоспособности, проверка файла и embedding.",
    },
]

SWAGGER_UI_PARAMETERS = {
    "tryItOutEnabled": True,
    "displayRequestDuration": True,
    "docExpansion": "list",
    "defaultModelsExpandDepth": -1,
}

DESCRIPTION = """
Визуальный поиск вина по фотографии на модели `facebook/dinov2-small`
(загружается один раз при старте сервиса) и PostgreSQL/pgvector.

**Быстрая проверка:**

1. **Эталоны → `POST /index`** — `Execute` с телом `{}` проиндексирует фото из
   `data/reference` (на CPU около 0,6 с на фото).
2. **Поиск → `POST /search`** — выберите файл фото и нажмите `Execute`.
3. **Эталоны → `GET /images`** — вставьте `best_image_uri` из ответа поиска,
   чтобы посмотреть найденное фото.
"""


class UriSearchRequest(BaseModel):
    """Search for a photo that is already inside DATA_ROOT."""

    image_uri: str = Field(min_length=1, examples=["/data/queries/test.jpeg"])
    top_k: int | None = Field(
        default=None, gt=0, description="по умолчанию DEFAULT_TOP_K"
    )
    request_id: str | None = Field(
        default=None, description="по умолчанию новый UUID"
    )


class IndexRequest(BaseModel):
    """Where to take reference photos from; empty body means DATA_ROOT/reference."""

    manifest: str | None = Field(
        default=None,
        description="CSV wine_id,slug,image_uri; по умолчанию "
        "data/reference/manifest.csv, если он есть",
    )
    reference_dir: str | None = Field(
        default=None,
        description="папка с фото: фото в папке = одно вино, подпапка = одно вино; "
        "по умолчанию /data/reference",
    )
    prune: bool = Field(
        default=False,
        description="удалить из базы эталоны этой модели, которых больше нет "
        "в источнике",
    )

    @model_validator(mode="after")
    def one_source(self) -> IndexRequest:
        if self.manifest is not None and self.reference_dir is not None:
            raise ValueError("укажите manifest или reference_dir, но не оба")
        return self


INDEX_EXAMPLES = {
    "default": {
        "summary": "data/reference (обычный случай)",
        "description": "manifest.csv из data/reference, если он есть, иначе фото "
        "в папке. Поля manifest и reference_dir не нужны.",
        "value": {"prune": True},
    },
    "keep": {
        "summary": "data/reference, ничего не удалять из базы",
        "value": {},
    },
    "manifest": {
        "summary": "другой manifest.csv",
        "value": {"manifest": "/data/reference/manifest.csv"},
    },
    "folder": {
        "summary": "другая папка с фото",
        "value": {"reference_dir": "/data/reference"},
    },
}


class EvaluateRequest(BaseModel):
    """Recall@K on labelled user photos."""

    queries: str = Field(
        default="evaluation/queries.csv",
        description="CSV query_image_uri,wine_id; относительный путь — от DATA_ROOT",
        examples=["/data/evaluation/queries.csv"],
    )
    top_k: int | None = Field(
        default=None, gt=0, description="по умолчанию DEFAULT_TOP_K"
    )


class EmbedderProvider:
    """Load the embedder once and reuse it for every request."""

    def __init__(self, embedder: Embedder | None = None) -> None:
        self._embedder = embedder
        self._lock = Lock()

    def get(self) -> Embedder:
        if self._embedder is None:
            with self._lock:
                if self._embedder is None:
                    from ..embedding.dinov2_embedder import DinoV2Embedder

                    self._embedder = DinoV2Embedder()
        return self._embedder


def create_app(
    storage: ImageStorage | None = None,
    embedder: Embedder | None = None,
    repository_factory: RepositoryFactory | None = None,
    database_inspector: DatabaseInspector | None = None,
) -> FastAPI:
    """Build the API; collaborators are injectable for tests.

    The model is loaded when the service starts, so the first request is as
    fast as the others and a missing model fails the start, not a user request.
    A database connection is opened per request.
    """

    image_storage = storage if storage is not None else LocalImageStorage()
    file_storage = (
        image_storage
        if isinstance(image_storage, LocalImageStorage)
        else LocalImageStorage()
    )
    embedder_provider = EmbedderProvider(embedder)
    connect_repository = repository_factory or _connect_postgres
    indexing_lock = Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        loaded = embedder_provider.get()
        # One warm-up pass: the first real inference is otherwise several
        # times slower while the runtime initializes its kernels.
        loaded.embed(Image.new("RGB", WARMUP_IMAGE_SIZE, color=(128, 128, 128)))
        logger.info("model %s is loaded on %s", loaded.model_name, loaded.device)
        yield

    app = FastAPI(
        title="DINOv2 Retrieval — поиск вина по фото",
        description=DESCRIPTION,
        version="0.3.0",
        lifespan=lifespan,
        openapi_tags=OPENAPI_TAGS,
        swagger_ui_parameters=SWAGGER_UI_PARAMETERS,
    )

    @app.exception_handler(ImageValidationError)
    @app.exception_handler(ImagePreprocessingError)
    async def image_error_handler(request: Request, exc: Exception) -> JSONResponse:
        status_code = ERROR_STATUS_CODES.get(type(exc), 400)
        return JSONResponse(status_code=status_code, content={"detail": str(exc)})

    @app.exception_handler(ManifestError)
    async def manifest_error_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(ImportError)
    async def ml_unavailable_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={"detail": f"ML dependencies are not installed: {exc}"},
        )

    @app.exception_handler(EmbeddingError)
    async def model_error_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(RepositoryError)
    async def database_error_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(ConfigurationError)
    async def configuration_error_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    # --- Поиск ---------------------------------------------------------------

    @app.post(
        "/search",
        tags=[TAG_SEARCH],
        response_model=SearchResponse,
        response_model_exclude_none=True,
        summary="Найти Top-K вин по загруженной фотографии",
    )
    def search(
        file: UploadFile = File(description="фото бутылки: jpg, jpeg, png или webp"),
        top_k: int | None = Form(default=None, gt=0, description="по умолчанию 20"),
        request_id: str | None = Form(default=None, description="по умолчанию UUID"),
    ) -> SearchResponse:
        """Фото проходит те же проверки и подготовку, что эталоны, и **не
        сохраняется**: оно лежит во временной папке только на время запроса.

        В ответе — разные вина по убыванию `score` (1 — идентичное фото).
        Чтобы посмотреть найденное фото, передайте `best_image_uri` в
        `GET /images`.
        """

        with temporary_upload(file.file, file.filename, file.content_type) as (
            upload_storage,
            image_uri,
        ):
            return _search(image_uri, top_k, request_id, upload_storage)

    @app.post(
        "/search/uri",
        tags=[TAG_SEARCH],
        response_model=SearchResponse,
        response_model_exclude_none=True,
        summary="Найти Top-K вин по фотографии из DATA_ROOT",
    )
    def search_uri(body: UriSearchRequest) -> SearchResponse:
        """То же, что `POST /search`, но для фото, которое уже лежит в `data/`."""

        return _search(body.image_uri, body.top_k, body.request_id, image_storage)

    # --- Эталоны -------------------------------------------------------------

    @app.post(
        "/index",
        tags=[TAG_REFERENCES],
        response_model=IndexingStats,
        summary="Проиндексировать эталонные фото",
    )
    def index(
        body: IndexRequest = Body(openapi_examples=INDEX_EXAMPLES),
    ) -> IndexingStats:
        """Пустое тело `{}` — взять `data/reference`: его `manifest.csv`, если он
        есть, иначе фото в папке (фото = вино, подпапка = вино).

        Повторный запуск обновляет записи, а не создаёт дубликаты;
        `"prune": true` удаляет из базы фото, которых больше нет в источнике.
        Запрос ждёт окончания индексации (на CPU около 0,6 с на фото); новые
        эталоны сразу доступны поиску.
        """

        if not indexing_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="Indexing is already running")
        try:
            references, source = load_references(body.manifest, body.reference_dir)
            with closing(connect_repository()) as repository:
                stats = index_reference_images(
                    references,
                    repository,
                    prune=body.prune,
                    embedder=embedder_provider.get(),
                )
        finally:
            indexing_lock.release()
        return stats.model_copy(update={"source": source})

    @app.get(
        "/references",
        tags=[TAG_REFERENCES],
        response_model=ReferenceList,
        summary="Список проиндексированных вин",
    )
    def references(
        limit: int = Query(default=100, ge=1, le=1000, description="вин на странице"),
        offset: int = Query(default=0, ge=0, description="сколько вин пропустить"),
    ) -> ReferenceList:
        """Вина текущей модели по алфавиту `wine_id`, у каждого — все его фото."""

        model_name = embedder_provider.get().model_name
        with closing(connect_repository()) as repository:
            return ReferenceList(
                model_name=model_name,
                wine_count=len(repository.get_wine_ids(model_name)),
                photo_count=repository.get_reference_count(model_name),
                offset=offset,
                wines=repository.list_references(model_name, limit, offset),
            )

    @app.get(
        "/images",
        tags=[TAG_REFERENCES],
        summary="Показать фотографию из DATA_ROOT",
        response_class=Response,
        responses={
            200: {
                "description": "уменьшенная копия (JPEG) или исходный файл",
                "content": {"image/jpeg": {}, "image/png": {}, "image/webp": {}},
            }
        },
    )
    def images(
        uri: str = Query(
            description="путь внутри контейнера: `/data/queries/test2.webp` или "
            "относительно /data: `queries/test2.webp` (без `data/` в начале); "
            "удобно вставить best_image_uri из ответа поиска",
            examples=["/data/queries/test.jpeg"],
        ),
        max_side: int = Query(
            default=800,
            ge=0,
            le=4096,
            description="уменьшить до этого размера по длинной стороне; "
            "0 — исходный файл",
        ),
    ) -> Response:
        """Swagger покажет картинку прямо в ответе. Уменьшенная копия
        повёрнута по EXIF — так же, как фото подаётся модели."""

        path, media_type = file_storage.locate(uri)
        if max_side == 0:
            return FileResponse(path, media_type=media_type)
        with ImagePreprocessor().open_rgb(path) as image:
            image.thumbnail((max_side, max_side))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=PREVIEW_JPEG_QUALITY)
        return Response(content=buffer.getvalue(), media_type="image/jpeg")

    # --- Оценка качества -----------------------------------------------------

    @app.post(
        "/evaluate",
        tags=[TAG_EVALUATION],
        response_model=EvaluationReport,
        summary="Посчитать Recall@K",
    )
    def evaluate(body: EvaluateRequest) -> EvaluationReport:
        """Каждое фото из `queries.csv` ищется так же, как в `POST /search`, и
        проверяется, на каком месте правильный `wine_id`. Фото запросов в базу
        не записываются."""

        queries = read_evaluation_queries(resolve_data_path(body.queries))
        with closing(connect_repository()) as repository:
            return evaluate_retrieval(
                queries,
                repository,
                top_k=body.top_k or get_default_top_k(),
                embedder=embedder_provider.get(),
            )

    # --- Служебное -----------------------------------------------------------

    @app.get("/health", tags=[TAG_SERVICE], summary="Сервис запущен")
    def health() -> dict[str, str]:
        """Быстрая проверка для Docker healthcheck: модель загружена, сервис
        отвечает."""

        return {"status": "ok"}

    @app.get(
        "/health/details",
        tags=[TAG_SERVICE],
        response_model=HealthReport,
        response_model_exclude_none=True,
        summary="Подробная проверка: конфигурация, PostgreSQL, pgvector, модель",
        responses={503: {"model": HealthReport, "description": "есть ошибки"}},
    )
    def health_details(response: Response) -> HealthReport:
        """То же, что CLI `health`: ничего не меняет, пароль в `DATABASE_URL`
        скрыт. Код 503, если какая-то проверка не прошла."""

        report = check_health(
            inspect_database=database_inspector,
            create_embedder=lambda settings: embedder_provider.get(),
        )
        if report.status != "ok":
            response.status_code = 503
        return report

    @app.post(
        "/validate",
        tags=[TAG_SERVICE],
        response_model=ValidatedImage,
        summary="Проверить изображение без создания embedding",
    )
    def validate(body: ImageRequest) -> ValidatedImage:
        request = RetrievalRequest(
            job_id="api", query_id="api", image_uri=body.image_uri
        )
        return validate_image(request, storage=image_storage)

    @app.post(
        "/embed",
        tags=[TAG_SERVICE],
        response_model=EmbeddingResult,
        summary="Получить полный embedding изображения",
    )
    def embed(body: ImageRequest) -> EmbeddingResult:
        return create_embedding(
            body.image_uri,
            storage=image_storage,
            embedder=embedder_provider.get(),
        )

    def _search(
        image_uri: str,
        top_k: int | None,
        request_id: str | None,
        search_storage: ImageStorage,
    ) -> SearchResponse:
        request = SearchRequest(
            request_id=request_id or uuid.uuid4().hex,
            image_uri=image_uri,
            top_k=top_k or get_default_top_k(),
        )
        with closing(connect_repository()) as repository:
            return search_similar_wines(
                request,
                repository,
                storage=search_storage,
                embedder=embedder_provider.get(),
            )

    return app


def _connect_postgres() -> ReferenceRepository:
    try:
        from ..infrastructure.database.postgres_reference_repository import (
            PostgresReferenceRepository,
        )
    except ImportError as exc:
        raise RepositoryError(
            f"PostgreSQL support is not installed (the db extra): {exc}"
        ) from exc
    return PostgresReferenceRepository.connect()
