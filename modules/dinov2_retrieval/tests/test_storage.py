from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from dinov2_retrieval.application.validate_image import validate_image
from dinov2_retrieval.contracts import RetrievalRequest
from dinov2_retrieval.infrastructure.storage.local_storage import (
    CorruptedImageError,
    ImageNotFoundError,
    ImageOutsideDataRootError,
    ImageTooLargeError,
    UnsupportedImageFormatError,
)


def make_request(image_uri: Path) -> RetrievalRequest:
    return RetrievalRequest(
        job_id="job-1",
        query_id="query-1",
        image_uri=str(image_uri),
    )


def write_valid_image(path: Path, image_format: str = "JPEG") -> None:
    image = Image.new("RGB", (32, 16), color=(20, 40, 60))
    image.save(path, format=image_format)


def test_valid_image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    image_path = tmp_path / "queries" / "test.jpg"
    image_path.parent.mkdir()
    write_valid_image(image_path)

    result = validate_image(make_request(image_path))

    assert result.path == image_path.resolve()
    assert result.width == 32
    assert result.height == 16
    assert result.mime_type == "image/jpeg"


def test_missing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))

    with pytest.raises(ImageNotFoundError):
        validate_image(make_request(tmp_path / "missing.jpg"))


def test_path_outside_data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    outside_path = tmp_path / "outside.jpg"
    write_valid_image(outside_path)
    monkeypatch.setenv("DATA_ROOT", str(data_root))

    with pytest.raises(ImageOutsideDataRootError):
        validate_image(make_request(outside_path))


def test_unsupported_format(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    image_path = tmp_path / "test.gif"
    image_path.write_bytes(b"not supported")

    with pytest.raises(UnsupportedImageFormatError):
        validate_image(make_request(image_path))


def test_supported_extensions_are_read_from_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("SUPPORTED_IMAGE_EXTENSIONS", ".png")
    image_path = tmp_path / "test.jpg"
    write_valid_image(image_path)

    with pytest.raises(UnsupportedImageFormatError):
        validate_image(make_request(image_path))


def test_file_too_large(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("MAX_IMAGE_SIZE_BYTES", "10")
    image_path = tmp_path / "test.jpg"
    write_valid_image(image_path)

    with pytest.raises(ImageTooLargeError):
        validate_image(make_request(image_path))


def test_corrupted_image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    image_path = tmp_path / "broken.jpg"
    image_path.write_bytes(b"this is not a valid image")

    with pytest.raises(CorruptedImageError):
        validate_image(make_request(image_path))
