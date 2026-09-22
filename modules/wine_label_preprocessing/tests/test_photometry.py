from __future__ import annotations

import numpy as np
import pytest

from wine_label_preprocessing.errors import PreprocessingError
from wine_label_preprocessing.models import PhotometricConfig
from wine_label_preprocessing.photometry import (
    apply_photometric,
    mild_ocr_profile,
)


def test_disabled_photometry_is_exact_copy() -> None:
    image = np.full((20, 30, 3), (30, 80, 150), dtype=np.uint8)

    output, operations = apply_photometric(image, PhotometricConfig())

    np.testing.assert_array_equal(output, image)
    assert output is not image
    assert operations == []


def test_mild_profile_preserves_image_contract() -> None:
    image = np.full((32, 48, 3), 50, dtype=np.uint8)

    output, operations = apply_photometric(image, mild_ocr_profile())

    assert output.shape == image.shape
    assert output.dtype == np.uint8
    assert output.flags.c_contiguous
    assert {item["name"] for item in operations} == {"brightness", "clahe"}
    assert float(output.mean()) > float(image.mean())


def test_weak_unsharp_is_separately_switchable() -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, 16:] = 180
    config = PhotometricConfig(unsharp_enabled=True, unsharp_amount=0.15)

    output, operations = apply_photometric(image, config)

    assert [item["name"] for item in operations] == ["unsharp"]
    assert output.dtype == np.uint8


@pytest.mark.parametrize(
    "config, message",
    [
        (PhotometricConfig(min_gain=1.2, max_gain=1.1), "gain"),
        (PhotometricConfig(clahe_clip_limit=0), "CLAHE"),
        (PhotometricConfig(unsharp_sigma=0), "sigma"),
        (PhotometricConfig(unsharp_amount=-0.1), "amount"),
    ],
)
def test_invalid_photometric_config_is_rejected(
    config: PhotometricConfig, message: str
) -> None:
    with pytest.raises(PreprocessingError, match=message):
        apply_photometric(np.zeros((4, 4, 3), dtype=np.uint8), config)

