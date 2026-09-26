from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from wine_label_preprocessing.errors import ImageInputError
from wine_label_preprocessing.image_io import crop_rgb, limit_resolution, load_rgb
from wine_label_preprocessing.models import BBox


def test_numpy_requires_explicit_color_order() -> None:
    with pytest.raises(ImageInputError, match="color order"):
        load_rgb(np.zeros((2, 3, 3), dtype=np.uint8))


def test_bgr_is_converted_to_contiguous_rgb() -> None:
    bgr = np.array([[[3, 2, 1]]], dtype=np.uint8)

    loaded = load_rgb(bgr, input_color_order="BGR")

    assert loaded.array.tolist() == [[[1, 2, 3]]]
    assert loaded.array.flags.c_contiguous
    assert loaded.metadata["input_color_order"] == "BGR"


def test_rejects_wrong_numpy_dtype_and_shape() -> None:
    with pytest.raises(ImageInputError, match="uint8"):
        load_rgb(np.zeros((2, 3, 3), dtype=np.float32), input_color_order="RGB")
    with pytest.raises(ImageInputError, match="shape"):
        load_rgb(np.zeros((2, 3), dtype=np.uint8), input_color_order="RGB")


def test_exif_orientation_is_applied_before_coordinates(tmp_path: Path) -> None:
    path = tmp_path / "oriented.jpg"
    image = Image.new("RGB", (6, 4), color=(20, 40, 60))
    exif = image.getexif()
    exif[274] = 6
    image.save(path, exif=exif)

    loaded = load_rgb(path)

    assert loaded.array.shape == (6, 4, 3)
    assert loaded.metadata["encoded_size"] == [6, 4]
    assert loaded.metadata["oriented_size"] == [4, 6]
    assert loaded.metadata["exif_transposed"] is True


def test_pillow_image_is_not_closed_by_loader() -> None:
    image = Image.new("RGB", (4, 3), color=(1, 2, 3))

    loaded = load_rgb(image)

    assert loaded.array.shape == (3, 4, 3)
    assert image.getpixel((0, 0)) == (1, 2, 3)


def test_bbox_is_clamped_and_half_open() -> None:
    image = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)

    crop, effective = crop_rgb(image, BBox(-2, 1, 20, 5))

    assert crop.shape == (4, 8, 3)
    assert effective == BBox(0, 1, 8, 5)
    np.testing.assert_array_equal(crop, image[1:5, 0:8])


def test_empty_bbox_after_clamping_is_rejected() -> None:
    image = np.zeros((6, 8, 3), dtype=np.uint8)

    with pytest.raises(ImageInputError, match="empty"):
        crop_rgb(image, BBox(20, 1, 30, 5))


def test_limit_resolution_preserves_ratio_and_never_upscales() -> None:
    image = np.zeros((10, 20, 3), dtype=np.uint8)

    original_size, original_scale = limit_resolution(image, 40)
    limited, limited_scale = limit_resolution(image, 10)

    assert original_size.shape == (10, 20, 3)
    assert original_scale == 1.0
    assert limited.shape == (5, 10, 3)
    assert limited_scale == pytest.approx(0.5)


def test_invalid_resolution_limit_is_rejected() -> None:
    with pytest.raises(ImageInputError, match="positive"):
        limit_resolution(np.zeros((2, 2, 3), dtype=np.uint8), 0)
