"""Use case: check configuration, PostgreSQL schema and model availability."""

from __future__ import annotations

from collections.abc import Callable

from PIL import Image

from ..config import ConfigurationError, Settings, load_settings, redact_database_url
from ..contracts import DatabaseInspection, HealthCheck, HealthReport
from ..embedding.base import Embedder

DatabaseInspector = Callable[[Settings], DatabaseInspection]
EmbedderFactory = Callable[[Settings], Embedder]

DATABASE_CHECKS = ("database", "pgvector", "reference_table")
PROBE_IMAGE_SIZE = (224, 224)


def check_health(
    *,
    inspect_database: DatabaseInspector | None = None,
    create_embedder: EmbedderFactory | None = None,
) -> HealthReport:
    """Run every check and report each one; nothing in the database is changed."""

    checks: dict[str, HealthCheck] = {}
    try:
        settings = load_settings()
    except ConfigurationError as exc:
        checks["config"] = HealthCheck(status="error", message=str(exc))
        for name in (*DATABASE_CHECKS, "model"):
            checks[name] = HealthCheck(
                status="skipped", message="configuration is invalid"
            )
        return HealthReport(status="error", checks=checks)

    checks["config"] = HealthCheck(status="ok", details=_describe_settings(settings))
    checks.update(
        _check_database(settings, inspect_database or _default_inspect_database)
    )
    checks["model"] = _check_model(
        settings, create_embedder or _default_create_embedder
    )
    status = "ok" if all(check.status == "ok" for check in checks.values()) else "error"
    return HealthReport(status=status, checks=checks)


def _describe_settings(settings: Settings) -> dict[str, object]:
    return {
        "data_root": str(settings.data_root),
        "max_image_size_bytes": settings.max_image_size_bytes,
        "supported_image_extensions": list(settings.supported_image_extensions),
        "hf_home": settings.hf_home,
        "database_url": redact_database_url(settings.database_url),
        "dino_model_name": settings.dino_model_name,
        "dino_embedding_dimension": settings.dino_embedding_dimension,
        "default_top_k": settings.default_top_k,
        "raw_retrieval_limit": settings.raw_retrieval_limit,
    }


def _check_database(
    settings: Settings, inspect: DatabaseInspector
) -> dict[str, HealthCheck]:
    url = redact_database_url(settings.database_url)
    try:
        inspection = inspect(settings)
    except Exception as exc:  # the report must describe any failure
        unavailable = HealthCheck(status="skipped", message="database is unavailable")
        return {
            "database": HealthCheck(
                status="error", message=str(exc), details={"url": url}
            ),
            "pgvector": unavailable,
            "reference_table": unavailable,
        }

    return {
        "database": HealthCheck(
            status="ok",
            details={"url": url, "server_version": inspection.server_version},
        ),
        "pgvector": (
            HealthCheck(status="ok", details={"version": inspection.pgvector_version})
            if inspection.pgvector_version
            else HealthCheck(status="error", message="extension vector is not installed")
        ),
        "reference_table": _check_reference_table(
            inspection, settings.dino_embedding_dimension
        ),
    }


def _check_reference_table(
    inspection: DatabaseInspection, expected_dimension: int
) -> HealthCheck:
    if not inspection.table_exists:
        return HealthCheck(
            status="error",
            message="table reference_images does not exist; "
            "the index command creates it",
        )

    details = {
        "embedding_dimension": inspection.embedding_dimension,
        "reference_images": inspection.reference_count,
        "model_reference_images": inspection.model_reference_count,
    }
    if inspection.embedding_dimension != expected_dimension:
        return HealthCheck(
            status="error",
            message=f"embedding column has dimension {inspection.embedding_dimension}, "
            f"but DINO_EMBEDDING_DIMENSION is {expected_dimension}",
            details=details,
        )
    return HealthCheck(status="ok", details=details)


def _check_model(settings: Settings, create: EmbedderFactory) -> HealthCheck:
    try:
        embedder = create(settings)
        probe = Image.new("RGB", PROBE_IMAGE_SIZE, color=(128, 128, 128))
        dimension = len(embedder.embed(probe))
    except Exception as exc:  # the report must describe any failure
        return HealthCheck(
            status="error",
            message=f"{type(exc).__name__}: {exc}",
            details={"model_name": settings.dino_model_name},
        )

    details = {
        "model_name": embedder.model_name,
        "device": embedder.device,
        "embedding_dimension": dimension,
        "hf_home": settings.hf_home,
    }
    if dimension != settings.dino_embedding_dimension:
        return HealthCheck(
            status="error",
            message=f"model returned {dimension} values, "
            f"expected {settings.dino_embedding_dimension}",
            details=details,
        )
    return HealthCheck(status="ok", details=details)


def _default_inspect_database(settings: Settings) -> DatabaseInspection:
    from ..infrastructure.database.connection import inspect_database

    return inspect_database(settings.dino_model_name, settings.database_url)


def _default_create_embedder(settings: Settings) -> Embedder:
    from ..embedding.dinov2_embedder import DinoV2Embedder

    return DinoV2Embedder(
        settings.dino_model_name,
        embedding_dimension=settings.dino_embedding_dimension,
    )
