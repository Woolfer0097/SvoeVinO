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


def _read_limited(source: BinaryIO) -> bytes:
    max_size = get_max_image_size_bytes()
    data = source.read(max_size + 1)
    if len(data) > max_size:
        raise ImageTooLargeError(f"Image is too large; maximum is {max_size} bytes")
    return data


@contextmanager
def temporary_pair(
    query: BinaryIO,
    reference: BinaryIO,
    *,
    query_filename: str | None = None,
    query_content_type: str | None = None,
    reference_filename: str | None = None,
    reference_content_type: str | None = None,
) -> Iterator[tuple[LocalImageStorage, str, str]]:
    """Yield one storage and URIs for a query/reference upload pair.

    Both photos are written to a private temporary folder, so they go through
    the same checks as files under DATA_ROOT, and the folder is deleted as
    soon as the block ends.
    """

    query_suffix = upload_suffix(query_filename, query_content_type)
    reference_suffix = upload_suffix(reference_filename, reference_content_type)
    query_bytes = _read_limited(query)
    reference_bytes = _read_limited(reference)

    with TemporaryDirectory(prefix="superpoint-upload-") as directory:
        query_path = Path(directory) / f"query{query_suffix}"
        reference_path = Path(directory) / f"reference{reference_suffix}"
        query_path.write_bytes(query_bytes)
        reference_path.write_bytes(reference_bytes)
        yield (
            LocalImageStorage(data_root=directory),
            str(query_path),
            str(reference_path),
        )
