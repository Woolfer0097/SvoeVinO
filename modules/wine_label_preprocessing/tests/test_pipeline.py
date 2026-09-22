from __future__ import annotations

import numpy as np

from wine_label_preprocessing.models import (
    BBox,
    CylinderGeometry,
    ImageAnnotations,
    PreprocessingConfig,
    DewarpNetResult,
)
from wine_label_preprocessing.photometry import mild_ocr_profile
from wine_label_preprocessing.pipeline import preprocess


def test_default_profiles_do_not_change_pixels() -> None:
    image = np.full((20, 30, 3), (30, 80, 150), dtype=np.uint8)

    result = preprocess(image, input_color_order="RGB")

    np.testing.assert_array_equal(result.original_crop, image)
    np.testing.assert_array_equal(result.embedding_image, image)
    np.testing.assert_array_equal(result.ocr_image, image)
    assert result.cylindrical_image is None
    assert result.metadata["cylindrical"]["reason"] == "label_bbox_missing"


def test_label_bbox_has_priority_and_is_recorded() -> None:
    yy, xx = np.mgrid[:20, :30]
    image = np.stack((xx, yy, np.zeros_like(xx)), axis=-1).astype(np.uint8)
    annotations = ImageAnnotations(
        label_bbox=BBox(5, 4, 25, 16),
        bottle_bbox=BBox(2, 1, 28, 19),
    )

    result = preprocess(image, annotations=annotations, input_color_order="RGB")

    np.testing.assert_array_equal(result.original_crop, image[4:16, 5:25])
    assert result.metadata["crop"]["source"] == "label_bbox"
    assert result.metadata["crop"]["bbox"] == [5, 4, 25, 16]
    assert result.metadata["cylindrical"]["reason"] == "geometry_missing"


def test_original_crop_stays_full_resolution_while_working_branches_are_limited() -> None:
    image = np.full((100, 200, 3), 80, dtype=np.uint8)
    config = PreprocessingConfig(
        embedding_max_long_side=50,
        ocr_max_long_side=100,
        ocr_photometric=mild_ocr_profile(),
    )

    result = preprocess(image, config=config, input_color_order="RGB")

    assert result.original_crop.shape == (100, 200, 3)
    assert result.embedding_image.shape == (25, 50, 3)
    assert result.ocr_image.shape == (50, 100, 3)
    assert result.metadata["embedding"]["photometric_operations"] == []
    assert {op["name"] for op in result.metadata["ocr"]["photometric_operations"]} == {
        "brightness",
        "clahe",
    }


def test_bottle_bbox_is_baseline_fallback_but_not_label_geometry() -> None:
    image = np.zeros((20, 30, 3), dtype=np.uint8)

    result = preprocess(
        image,
        annotations=ImageAnnotations(bottle_bbox=BBox(3, 2, 27, 18)),
        input_color_order="RGB",
    )

    assert result.original_crop.shape == (16, 24, 3)
    assert result.metadata["crop"]["source"] == "bottle_bbox"
    assert result.metadata["cylindrical"]["reason"] == "label_bbox_missing"


def test_metadata_has_stage_timings_and_operations() -> None:
    result = preprocess(
        np.zeros((10, 12, 3), dtype=np.uint8), input_color_order="RGB"
    )

    assert result.metadata["schema_version"] == "1.0"
    assert set(result.metadata["timings_ms"]) >= {
        "load",
        "crop",
        "embedding_branch",
        "ocr_branch",
        "total",
    }
    assert all(value >= 0 for value in result.metadata["timings_ms"].values())
    assert result.metadata["config"]["embedding_max_long_side"] == 2400
    assert result.metadata["annotations"] == {
        "label_bbox": None,
        "bottle_bbox": None,
        "bottle_mask_provided": False,
        "bottle_mask_shape": None,
        "cylinder": None,
    }


def test_explicit_cylinder_geometry_produces_corrected_variant() -> None:
    yy, xx = np.mgrid[:80, :140]
    image = np.stack((xx, yy, np.zeros_like(xx)), axis=-1).astype(np.uint8)
    annotations = ImageAnnotations(
        label_bbox=BBox(30, 20, 110, 60),
        cylinder=CylinderGeometry(70.0, 55.0, -0.6, 0.6),
    )

    result = preprocess(image, annotations=annotations, input_color_order="RGB")

    assert result.cylindrical_image is not None
    assert result.cylindrical_valid_mask is not None
    assert result.metadata["cylindrical"]["status"] == "ok"
    assert result.metadata["cylindrical"]["geometry_source"] == "explicit"
    assert result.metadata["annotations"]["cylinder"] == {
        "cx": 70.0,
        "radius": 55.0,
        "theta_min": -0.6,
        "theta_max": 0.6,
    }


def test_pipeline_estimates_geometry_from_full_image_mask() -> None:
    image = np.full((80, 140, 3), 120, dtype=np.uint8)
    mask = np.zeros((80, 140), dtype=np.uint8)
    mask[:, 20:120] = 1
    annotations = ImageAnnotations(
        label_bbox=BBox(40, 20, 100, 60),
        bottle_mask=mask,
    )

    result = preprocess(image, annotations=annotations, input_color_order="RGB")

    assert result.cylindrical_image is not None
    assert result.metadata["cylindrical"]["geometry_source"] == "mask"
    assert result.metadata["cylindrical"]["geometry_estimate"]["reliable"] is True


def test_pipeline_accepts_mask_relative_to_bottle_bbox() -> None:
    image = np.full((80, 140, 3), 120, dtype=np.uint8)
    roi_mask = np.ones((70, 100), dtype=np.uint8)
    annotations = ImageAnnotations(
        label_bbox=BBox(40, 20, 100, 60),
        bottle_bbox=BBox(20, 5, 120, 75),
        bottle_mask=roi_mask,
    )

    result = preprocess(image, annotations=annotations, input_color_order="RGB")

    assert result.cylindrical_image is not None
    assert result.metadata["cylindrical"]["geometry_source"] == "mask"


def test_relative_mask_is_clipped_with_out_of_bounds_bottle_bbox() -> None:
    image = np.full((70, 100, 3), 120, dtype=np.uint8)
    roi_mask = np.ones((80, 120), dtype=np.uint8)
    annotations = ImageAnnotations(
        label_bbox=BBox(20, 15, 80, 55),
        bottle_bbox=BBox(-10, -5, 110, 75),
        bottle_mask=roi_mask,
    )

    result = preprocess(image, annotations=annotations, input_color_order="RGB")

    assert result.cylindrical_image is not None
    assert result.metadata["cylindrical"]["geometry_source"] == "mask"


def test_pipeline_keeps_baselines_when_geometry_is_unreliable() -> None:
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    sparse_mask = np.zeros((80, 120), dtype=np.uint8)
    sparse_mask[30, 40:80] = 1
    annotations = ImageAnnotations(
        label_bbox=BBox(30, 20, 90, 60), bottle_mask=sparse_mask
    )

    result = preprocess(image, annotations=annotations, input_color_order="RGB")

    assert result.cylindrical_image is None
    assert result.metadata["cylindrical"]["status"] == "skipped"
    assert result.metadata["cylindrical"]["reason"] == "mask_rows_insufficient"


def test_pipeline_uses_injected_dewarpnet_adapter_once() -> None:
    image = np.full((20, 30, 3), 90, dtype=np.uint8)

    class FakeAdapter:
        def __init__(self) -> None:
            self.calls = 0

        def unwrap(self, crop: np.ndarray) -> DewarpNetResult:
            self.calls += 1
            return DewarpNetResult(
                image=255 - crop,
                valid_mask=np.ones(crop.shape[:2], dtype=bool),
                status="ok",
                reason=None,
                metadata={"adapter": "fake"},
            )

    adapter = FakeAdapter()

    result = preprocess(image, input_color_order="RGB", dewarpnet=adapter)

    assert adapter.calls == 1
    assert result.dewarpnet_image is not None
    np.testing.assert_array_equal(result.dewarpnet_image, 255 - image)
    assert result.metadata["dewarpnet"]["status"] == "ok"
    assert result.metadata["dewarpnet"]["adapter"] == "fake"
