"""Environment-backed configuration for local photo verification."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024
DEFAULT_MAX_NUM_KEYPOINTS = 2048
DEFAULT_RESIZE = 1024
DEFAULT_DEPTH_CONFIDENCE = 0.95
DEFAULT_WIDTH_CONFIDENCE = 0.99
DEFAULT_FILTER_THRESHOLD = 0.1
DEFAULT_MIN_MATCHES = 15
DEFAULT_MIN_INLIERS = 12
DEFAULT_MIN_INLIER_RATIO = 0.3
DEFAULT_RANSAC_REPROJ_THRESHOLD = 5.0
DEFAULT_RANSAC_CONFIDENCE = 0.999

SUPPORTED_IMAGE_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
DEFAULT_SUPPORTED_IMAGE_EXTENSIONS = tuple(SUPPORTED_IMAGE_MIME_TYPES)


class ConfigurationError(ValueError):
    """Raised when photo verification configuration is invalid."""


@dataclass(frozen=True)
class VerificationThresholds:
    """Thresholds that turn geometric matches into a verification decision."""

    min_matches: int
    min_inliers: int
    min_inlier_ratio: float


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
        max_size = _read_int(
            "MAX_IMAGE_SIZE_BYTES",
            default=DEFAULT_MAX_IMAGE_SIZE_BYTES,
        )
    if max_size <= 0:
        raise ConfigurationError("MAX_IMAGE_SIZE_BYTES must be a positive integer")
    return max_size


def get_supported_image_extensions(value: str | None = None) -> tuple[str, ...]:
    """Return supported image extensions from configuration."""

    raw_value = value if value is not None else os.getenv("SUPPORTED_IMAGE_EXTENSIONS")
    if raw_value is None:
        return DEFAULT_SUPPORTED_IMAGE_EXTENSIONS

    extensions = tuple(
        dict.fromkeys(
            extension if extension.startswith(".") else f".{extension}"
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


def get_max_num_keypoints(value: str | int | None = None) -> int | None:
    """Return the SuperPoint keypoint cap, or None to keep every keypoint."""

    raw_value = (
        value if value is not None else os.getenv("MAX_NUM_KEYPOINTS", "")
    )
    if raw_value == "":
        return DEFAULT_MAX_NUM_KEYPOINTS
    if isinstance(raw_value, str) and raw_value.strip().lower() == "none":
        return None
    parsed = raw_value if isinstance(raw_value, int) else _parse_int(
        "MAX_NUM_KEYPOINTS", str(raw_value)
    )
    if parsed == 0:
        return None
    if parsed < 0:
        raise ConfigurationError("MAX_NUM_KEYPOINTS must be positive, 0 or none")
    return parsed


def get_resize(value: str | int | None = None) -> int | None:
    """Return the long-edge resize used before SuperPoint, or None to disable it."""

    raw_value = value if value is not None else os.getenv("SUPERPOINT_RESIZE", "")
    if raw_value == "":
        return DEFAULT_RESIZE
    if isinstance(raw_value, str) and raw_value.strip().lower() == "none":
        return None
    parsed = raw_value if isinstance(raw_value, int) else _parse_int(
        "SUPERPOINT_RESIZE", str(raw_value)
    )
    if parsed == 0:
        return None
    if parsed < 0:
        raise ConfigurationError("SUPERPOINT_RESIZE must be positive, 0 or none")
    return parsed


def get_depth_confidence(value: float | None = None) -> float:
    """Return LightGlue early-stopping confidence. -1 disables it."""

    parsed = (
        value
        if value is not None
        else _read_float("LIGHTGLUE_DEPTH_CONFIDENCE", DEFAULT_DEPTH_CONFIDENCE)
    )
    return _require_range("LIGHTGLUE_DEPTH_CONFIDENCE", parsed, minimum=-1.0, maximum=1.0)


def get_width_confidence(value: float | None = None) -> float:
    """Return LightGlue point-pruning confidence. -1 disables it."""

    parsed = (
        value
        if value is not None
        else _read_float("LIGHTGLUE_WIDTH_CONFIDENCE", DEFAULT_WIDTH_CONFIDENCE)
    )
    return _require_range("LIGHTGLUE_WIDTH_CONFIDENCE", parsed, minimum=-1.0, maximum=1.0)


def get_filter_threshold(value: float | None = None) -> float:
    """Return the LightGlue match-score threshold."""

    parsed = (
        value
        if value is not None
        else _read_float("MATCH_FILTER_THRESHOLD", DEFAULT_FILTER_THRESHOLD)
    )
    return _require_range("MATCH_FILTER_THRESHOLD", parsed, minimum=0.0, maximum=1.0)


def get_verification_thresholds(
    min_matches: int | None = None,
    min_inliers: int | None = None,
    min_inlier_ratio: float | None = None,
) -> VerificationThresholds:
    """Return the decision thresholds for a verified photo pair."""

    matches = (
        min_matches
        if min_matches is not None
        else _read_int("MIN_MATCHES", DEFAULT_MIN_MATCHES)
    )
    inliers = (
        min_inliers
        if min_inliers is not None
        else _read_int("MIN_INLIERS", DEFAULT_MIN_INLIERS)
    )
    ratio = (
        min_inlier_ratio
        if min_inlier_ratio is not None
        else _read_float("MIN_INLIER_RATIO", DEFAULT_MIN_INLIER_RATIO)
    )
    if matches < 0:
        raise ConfigurationError("MIN_MATCHES must be zero or a positive integer")
    if inliers < 0:
        raise ConfigurationError("MIN_INLIERS must be zero or a positive integer")
    _require_range("MIN_INLIER_RATIO", ratio, minimum=0.0, maximum=1.0)
    return VerificationThresholds(
        min_matches=matches,
        min_inliers=inliers,
        min_inlier_ratio=ratio,
    )


def get_ransac_reproj_threshold(value: float | None = None) -> float:
    """Return the RANSAC reprojection threshold in original image pixels."""

    parsed = (
        value
        if value is not None
        else _read_float("RANSAC_REPROJ_THRESHOLD", DEFAULT_RANSAC_REPROJ_THRESHOLD)
    )
    if parsed <= 0:
        raise ConfigurationError("RANSAC_REPROJ_THRESHOLD must be positive")
    return parsed


def get_ransac_confidence(value: float | None = None) -> float:
    """Return the RANSAC confidence in the open interval (0, 1]."""

    parsed = (
        value
        if value is not None
        else _read_float("RANSAC_CONFIDENCE", DEFAULT_RANSAC_CONFIDENCE)
    )
    if parsed <= 0 or parsed > 1:
        raise ConfigurationError("RANSAC_CONFIDENCE must be in the interval (0, 1]")
    return parsed


def get_requested_device(value: str | None = None) -> str:
    """Return the requested runtime device: auto, cpu or cuda."""

    raw_value = value if value is not None else os.getenv("DEVICE", "auto")
    device = raw_value.strip().lower()
    if device not in {"auto", "cpu", "cuda"}:
        raise ConfigurationError("DEVICE must be auto, cpu or cuda")
    return device


def _read_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    return _parse_int(name, raw_value)


def _parse_int(name: str, raw_value: str) -> int:
    try:
        return int(raw_value.strip())
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc


def _read_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        return float(raw_value.strip())
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number") from exc


def _require_range(name: str, value: float, *, minimum: float, maximum: float) -> float:
    if value < minimum or value > maximum:
        raise ConfigurationError(f"{name} must be between {minimum} and {maximum}")
    return value
