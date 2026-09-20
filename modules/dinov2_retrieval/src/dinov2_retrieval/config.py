"""Environment-backed configuration for local image validation."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024
SUPPORTED_IMAGE_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
DEFAULT_SUPPORTED_IMAGE_EXTENSIONS = tuple(SUPPORTED_IMAGE_MIME_TYPES)


class ConfigurationError(ValueError):
    """Raised when the image storage configuration is invalid."""


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


def get_max_image_size_bytes(value: int | None = None) -> int:
    """Return the configured maximum image size in bytes."""

    if value is not None:
        max_size = value
    else:
        raw_value = os.getenv("MAX_IMAGE_SIZE_BYTES")
        if raw_value is None:
            return DEFAULT_MAX_IMAGE_SIZE_BYTES
        try:
            max_size = int(raw_value)
        except ValueError as exc:
            raise ConfigurationError(
                "MAX_IMAGE_SIZE_BYTES must be a positive integer"
            ) from exc

    if max_size <= 0:
        raise ConfigurationError("MAX_IMAGE_SIZE_BYTES must be a positive integer")
    return max_size


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
