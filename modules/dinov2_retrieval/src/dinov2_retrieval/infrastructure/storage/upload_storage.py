"""Uploaded photos: validated like files in DATA_ROOT, but never kept on disk."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import BinaryIO

from ...config import SUPPORTED_IMAGE_MIME_TYPES, get_max_image_size_bytes
from .local_storage import (
    ImageTooLargeError,
    LocalImageStorage,
    UnsupportedImageFormatError,
)

SUFFIX_BY_CONTENT_TYPE = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


def upload_suffix(filename: str | None, content_type: str | None) -> str:
    """Pick the extension from the file name, or else from the content type."""

    suffix = Path(filename or "").suffix.lower()
    if suffix in SUPPORTED_IMAGE_MIME_TYPES:
        return suffix
    media_type = (content_type or "").split(";", 1)[0].strip().lower()
    if media_type in SUFFIX_BY_CONTENT_TYPE:
        return SUFFIX_BY_CONTENT_TYPE[media_type]
    raise UnsupportedImageFormatError(
        f"Unsupported upload {filename or '(no name)'} ({content_type or 'no type'}). "
        "Supported: jpg, jpeg, png, webp"
    )


@contextmanager
def temporary_upload(
    source: BinaryIO,
    filename: str | None = None,
    content_type: str | None = None,
) -> Iterator[tuple[LocalImageStorage, str]]:
    """Yield a storage and an image_uri for the uploaded photo.

    The photo is written to a private temporary folder, so it goes through the
    very same checks and preprocessing as reference photos, and the folder is
    deleted as soon as the block ends.
    """

    suffix = upload_suffix(filename, content_type)
    max_size = get_max_image_size_bytes()
    data = source.read(max_size + 1)
    if len(data) > max_size:
        raise ImageTooLargeError(f"Image is too large; maximum is {max_size} bytes")

    with TemporaryDirectory(prefix="dinov2-upload-") as directory:
        path = Path(directory) / f"upload{suffix}"
        path.write_bytes(data)
        yield LocalImageStorage(data_root=directory), str(path)
