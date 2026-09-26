from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from superpoint.application.validate_image import validate_image
from superpoint.infrastructure.storage.local_storage import (
    CorruptedImageError,
    ImageNotFoundError,
    ImageOutsideDataRootError,
    ImageTooLargeError,
    UnsupportedImageFormatError,
)


def write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 8), color=(10, 20, 30)).save(path, format="JPEG")


def test_validate_image_accepts_a_file_inside_data_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    image_path = tmp_path / "queries" / "photo.jpg"
    write_image(image_path)

    result = validate_image("queries/photo.jpg")

    assert result.path == image_path.resolve()
    assert result.width == 12
    assert result.height == 8
    assert result.mime_type == "image/jpeg"


def test_missing_file_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))

    with pytest.raises(ImageNotFoundError):
        validate_image(str(tmp_path / "missing.jpg"))


def test_path_outside_data_root_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    outside = tmp_path / "outside.jpg"
    write_image(outside)
    monkeypatch.setenv("DATA_ROOT", str(data_root))

    with pytest.raises(ImageOutsideDataRootError):
        validate_image(str(outside))


def test_parent_traversal_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    write_image(tmp_path / "secret.jpg")
    monkeypatch.setenv("DATA_ROOT", str(data_root))

    with pytest.raises(ImageOutsideDataRootError):
        validate_image("../secret.jpg")


def test_unsupported_extension_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    image_path = tmp_path / "photo.gif"
    image_path.write_bytes(b"gif")

    with pytest.raises(UnsupportedImageFormatError):
        validate_image(str(image_path))


def test_oversized_image_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("MAX_IMAGE_SIZE_BYTES", "10")
    image_path = tmp_path / "photo.jpg"
    write_image(image_path)

    with pytest.raises(ImageTooLargeError):
        validate_image(str(image_path))


def test_corrupted_image_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    image_path = tmp_path / "photo.jpg"
    image_path.write_bytes(b"not-an-image")

    with pytest.raises(CorruptedImageError):
        validate_image(str(image_path))
