"""Environment-backed configuration for the OCR service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .exceptions import ConfigurationError


def _as_bool(value: str | bool | None, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected boolean value, got: {value!r}")


def _as_int(value: str | int | None, *, default: int) -> int:
    if value is None:
        return default
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("Integer configuration values must be positive")
    return parsed


def _as_float(value: str | float | None, *, default: float) -> float:
    if value is None:
        return default
    parsed = float(value)
    if not 0 < parsed <= 1:
        raise ValueError("Float configuration values must be in (0, 1]")
    return parsed


def _as_languages(value: str | tuple[str, ...] | None) -> tuple[str, ...]:
    if value is None:
        return ("ru", "en")
    if isinstance(value, tuple):
        return value
    return tuple(item.strip().lower() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class OCRConfig:
    """Runtime configuration for OCR processing."""

    engine: str = "paddleocr"
    device: str = "auto"
    paddle_lang: str = "ru"
    expected_languages: tuple[str, ...] = ("ru", "en")
    enable_central_crop: bool = True
    central_crop_fraction: float = 0.72
    max_upload_size_bytes: int = 15 * 1024 * 1024
    output_dir: Path = Path("outputs")
    report_retention: bool = False

    @classmethod
    def from_env(cls) -> "OCRConfig":
        """Build configuration from environment variables."""

        try:
            return cls(
                engine=os.getenv("OCR_ENGINE", cls.engine),
                device=os.getenv("OCR_DEVICE", cls.device),
                paddle_lang=os.getenv("OCR_PADDLE_LANG", cls.paddle_lang),
                expected_languages=_as_languages(os.getenv("OCR_EXPECTED_LANGUAGES")),
                enable_central_crop=_as_bool(
                    os.getenv("OCR_ENABLE_CENTRAL_CROP"),
                    default=cls.enable_central_crop,
                ),
                central_crop_fraction=_as_float(
                    os.getenv("OCR_CENTRAL_CROP_FRACTION"),
                    default=cls.central_crop_fraction,
                ),
                max_upload_size_bytes=_as_int(
                    os.getenv("OCR_MAX_UPLOAD_SIZE_BYTES"),
                    default=cls.max_upload_size_bytes,
                ),
                output_dir=Path(os.getenv("OCR_OUTPUT_DIR", str(cls.output_dir))),
                report_retention=_as_bool(
                    os.getenv("OCR_REPORT_RETENTION"),
                    default=cls.report_retention,
                ),
            )
        except ValueError as exc:
            raise ConfigurationError(f"Invalid OCR configuration: {exc}") from exc
