"""Safe local image storage adapter."""

from __future__ import annotations

from pathlib import Path

from ...config import (
    SUPPORTED_IMAGE_MIME_TYPES,
    get_data_root,
    get_max_image_size_bytes,
    get_supported_image_extensions,
)
from ...contracts import ValidatedImage
from ...preprocessing.image_preprocessor import (
    ImagePreprocessingError,
    ImagePreprocessor,
)


class ImageValidationError(ValueError):
    """Base class for errors raised while validating a local image."""


class ImageNotFoundError(ImageValidationError):
    """Raised when the requested image does not exist or is not a file."""


class ImageOutsideDataRootError(ImageValidationError):
    """Raised when a path resolves outside DATA_ROOT."""


class UnsupportedImageFormatError(ImageValidationError):
    """Raised when the image extension is not supported."""


class ImageTooLargeError(ImageValidationError):
    """Raised when the image exceeds the configured size limit."""


class CorruptedImageError(ImageValidationError):
    """Raised when the image cannot be safely decoded."""


class LocalImageStorage:
    """Read and validate images below the configured ``DATA_ROOT``."""

    def __init__(
        self,
        data_root: str | Path | None = None,
        max_file_size_bytes: int | None = None,
        preprocessor: ImagePreprocessor | None = None,
    ) -> None:
        self.data_root = get_data_root(data_root)
        self.max_file_size_bytes = get_max_image_size_bytes(max_file_size_bytes)
        self.supported_extensions = get_supported_image_extensions()
        self.preprocessor = (
            preprocessor if preprocessor is not None else ImagePreprocessor()
        )

    def validate(self, image_uri: str) -> ValidatedImage:
        """Validate ``image_uri`` and return its dimensions and MIME type."""

        image_path = self._resolve_inside_data_root(image_uri)

        if not image_path.exists() or not image_path.is_file():
            raise ImageNotFoundError(f"Image file does not exist: {image_uri}")

        extension = image_path.suffix.lower()
        if extension not in self.supported_extensions:
            raise UnsupportedImageFormatError(
                "Unsupported image extension. Supported extensions: "
                + ", ".join(self.supported_extensions)
            )
        mime_type = SUPPORTED_IMAGE_MIME_TYPES[extension]

        file_size = image_path.stat().st_size
        if file_size > self.max_file_size_bytes:
            raise ImageTooLargeError(
                f"Image is too large: {file_size} bytes; maximum is "
                f"{self.max_file_size_bytes} bytes"
            )

        try:
            with self.preprocessor.open_rgb(image_path) as image:
                width, height = image.size
        except ImagePreprocessingError as exc:
            raise CorruptedImageError(f"Cannot decode image: {image_uri}") from exc

        return ValidatedImage(
            path=image_path,
            width=width,
            height=height,
            mime_type=mime_type,
        )

    def _resolve_inside_data_root(self, image_uri: str) -> Path:
        try:
            requested_path = Path(image_uri).expanduser()
            candidate = (
                requested_path
                if requested_path.is_absolute()
                else self.data_root / requested_path
            )
            resolved_path = candidate.resolve(strict=False)
            resolved_path.relative_to(self.data_root)
        except (TypeError, ValueError, OSError, RuntimeError) as exc:
            raise ImageOutsideDataRootError(
                f"image_uri must point inside DATA_ROOT: {image_uri}"
            ) from exc

        return resolved_path
