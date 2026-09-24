"""Environment-backed configuration for the retrieval module."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

DEFAULT_MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024
SUPPORTED_IMAGE_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
DEFAULT_SUPPORTED_IMAGE_EXTENSIONS = tuple(SUPPORTED_IMAGE_MIME_TYPES)
DEFAULT_DINO_MODEL_NAME = "facebook/dinov2-small"
DEFAULT_DINO_EMBEDDING_DIMENSION = 384
DEFAULT_TOP_K = 20
DEFAULT_RAW_RETRIEVAL_LIMIT = 100
DATABASE_URL_SCHEMES = ("postgresql", "postgres")


class ConfigurationError(ValueError):
    """Raised when the module configuration is invalid."""


def get_data_root(value: str | Path | None = None) -> Path:
    """Return a resolved, existing directory used as the image root."""

    raw_value = value if value is not None else os.getenv("DATA_ROOT")
    if not raw_value:
        raise ConfigurationError("DATA_ROOT must be set")

    root = Path(raw_value).expanduser().resolve(strict=False)
    if not root.exists():
        raise ConfigurationError(f"DATA_ROOT does not exist: {root}")
    if not root.is_dir():
        raise ConfigurationError(f"DATA_ROOT is not a directory: {root}")
    return root


def resolve_data_path(raw_path: str | Path) -> Path:
    """Return ``raw_path``; a relative path is taken from DATA_ROOT."""

    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else get_data_root() / path


def get_max_image_size_bytes(value: int | None = None) -> int:
    """Return the configured maximum image size in bytes."""

    return _get_positive_int(
        "MAX_IMAGE_SIZE_BYTES", value, DEFAULT_MAX_IMAGE_SIZE_BYTES
    )


def get_supported_image_extensions(value: str | None = None) -> tuple[str, ...]:
    """Return supported image extensions from configuration."""

    raw_value = (
        value if value is not None else os.getenv("SUPPORTED_IMAGE_EXTENSIONS")
    )
    if raw_value is None:
        return DEFAULT_SUPPORTED_IMAGE_EXTENSIONS

    extensions = tuple(
        dict.fromkeys(
            extension
            if extension.startswith(".")
            else f".{extension}"
            for extension in (item.strip().lower() for item in raw_value.split(","))
            if extension
        )
    )
    unknown_extensions = set(extensions) - set(SUPPORTED_IMAGE_MIME_TYPES)
    if not extensions or unknown_extensions:
        unknown = ", ".join(sorted(unknown_extensions))
        details = f" Unknown extensions: {unknown}." if unknown else ""
        raise ConfigurationError(
            "SUPPORTED_IMAGE_EXTENSIONS must contain only jpg, jpeg, png or webp."
            + details
        )
    return extensions


def get_hf_home(value: str | None = None) -> str | None:
    """Return the Hugging Face cache directory, if one is configured."""

    raw_value = value if value is not None else os.getenv("HF_HOME")
    return raw_value or None


def get_database_url(value: str | None = None) -> str:
    """Return the PostgreSQL connection URL."""

    raw_value = value if value is not None else os.getenv("DATABASE_URL")
    if not raw_value or not raw_value.strip():
        raise ConfigurationError("DATABASE_URL must be set")

    database_url = raw_value.strip()
    if urlsplit(database_url).scheme not in DATABASE_URL_SCHEMES:
        raise ConfigurationError(
            "DATABASE_URL must be a postgresql:// URL, "
            "for example postgresql://user:password@host:5432/database"
        )
    return database_url


def redact_database_url(database_url: str) -> str:
    """Hide the password of a connection URL before it is logged or printed."""

    parts = urlsplit(database_url)
    if parts.password is None:
        return database_url
    user_info, _, host_info = parts.netloc.rpartition("@")
    user = user_info.split(":", 1)[0]
    return urlunsplit(parts._replace(netloc=f"{user}:***@{host_info}"))


def get_dino_model_name(value: str | None = None) -> str:
    """Return the Hugging Face identifier of the DINOv2 model."""

    raw_value = value if value is not None else os.getenv("DINO_MODEL_NAME")
    if raw_value is None:
        return DEFAULT_DINO_MODEL_NAME
    if not raw_value.strip():
        raise ConfigurationError("DINO_MODEL_NAME must not be empty")
    return raw_value.strip()


def get_dino_embedding_dimension(value: int | None = None) -> int:
    """Return the expected embedding dimension of the DINOv2 model."""

    return _get_positive_int(
        "DINO_EMBEDDING_DIMENSION", value, DEFAULT_DINO_EMBEDDING_DIMENSION
    )


def get_default_top_k(value: int | None = None) -> int:
    """Return how many distinct wines a search returns by default."""

    return _get_positive_int("DEFAULT_TOP_K", value, DEFAULT_TOP_K)


def get_raw_retrieval_limit(value: int | None = None) -> int:
    """Return how many nearest reference photos are fetched before grouping."""

    return _get_positive_int(
        "RAW_RETRIEVAL_LIMIT", value, DEFAULT_RAW_RETRIEVAL_LIMIT
    )


@dataclass(frozen=True)
class Settings:
    """All module settings, validated together."""

    data_root: Path
    max_image_size_bytes: int
    supported_image_extensions: tuple[str, ...]
    hf_home: str | None
    database_url: str
    dino_model_name: str
    dino_embedding_dimension: int
    default_top_k: int
    raw_retrieval_limit: int


def load_settings() -> Settings:
    """Read and validate every setting from the environment."""

    return Settings(
        data_root=get_data_root(),
        max_image_size_bytes=get_max_image_size_bytes(),
        supported_image_extensions=get_supported_image_extensions(),
        hf_home=get_hf_home(),
        database_url=get_database_url(),
        dino_model_name=get_dino_model_name(),
        dino_embedding_dimension=get_dino_embedding_dimension(),
        default_top_k=get_default_top_k(),
        raw_retrieval_limit=get_raw_retrieval_limit(),
    )


def _get_positive_int(name: str, value: int | None, default: int) -> int:
    if value is not None:
        result = value
    else:
        raw_value = os.getenv(name)
        if raw_value is None:
            return default
        try:
            result = int(raw_value)
        except ValueError as exc:
            raise ConfigurationError(f"{name} must be a positive integer") from exc

    if result <= 0:
        raise ConfigurationError(f"{name} must be a positive integer")
    return result
