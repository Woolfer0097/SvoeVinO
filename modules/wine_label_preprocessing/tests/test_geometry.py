from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from wine_label_preprocessing.errors import GeometryError
from wine_label_preprocessing.geometry import (
    cylindrical_unwrap,
    estimate_cylinder_from_mask,
)
from wine_label_preprocessing.models import (
    BBox,
    CylinderGeometry,
    UnwrapConfig,
)


def coordinate_image(height: int, width: int) -> np.ndarray:
    yy, xx = np.mgrid[:height, :width]
    return np.stack((xx, yy, np.zeros_like(xx)), axis=-1).astype(np.uint8)


def test_unwrap_center_maps_to_cylinder_center_and_preserves_vertical_scale() -> None:
    image = coordinate_image(height=40, width=120)

    result = cylindrical_unwrap(
        image,
        label_bbox=BBox(10, 5, 110, 35),
        geometry=CylinderGeometry(
            cx=60.0,
            radius=50.0,
            theta_min=-0.4,
            theta_max=0.4,
        ),
    )

    assert result.status == "ok"
    assert result.image is not None
    assert result.valid_mask is not None
    assert result.image.shape == (30, 40, 3)
    center = result.image[result.image.shape[0] // 2, result.image.shape[1] // 2]
    assert int(center[0]) == pytest.approx(60, abs=1)
    assert int(center[1]) == pytest.approx(20, abs=1)
    assert result.metadata["cx"] == 60.0
    assert result.metadata["radius"] == 50.0
    assert result.metadata["label_bbox"] == [10, 5, 110, 35]
    assert result.metadata["horizontal_scale"] == pytest.approx(1.0, abs=0.03)
    assert result.metadata["vertical_scale"] == pytest.approx(1.0, abs=0.03)


def test_angle_range_can_be_derived_from_label_bounds() -> None:
    image = coordinate_image(height=60, width=140)

    result = cylindrical_unwrap(
        image,
        label_bbox=BBox(40, 10, 100, 50),
        geometry=CylinderGeometry(cx=70.0, radius=50.0),
    )

    assert result.status == "ok"
    assert result.metadata["angle_source"] == "label_bbox"
    assert result.metadata["requested_theta_min"] == pytest.approx(math.asin(-0.6))
    assert result.metadata["requested_theta_max"] == pytest.approx(math.asin(0.6))


def test_stretch_is_clipped_before_silhouette_edges() -> None:
    result = cylindrical_unwrap(
        coordinate_image(50, 160),
        label_bbox=BBox(5, 5, 155, 45),
        geometry=CylinderGeometry(
            cx=80.0,
            radius=70.0,
            theta_min=-1.5,
            theta_max=1.5,
        ),
        config=UnwrapConfig(max_stretch=2.0),
    )

    assert result.status == "ok"
    assert result.metadata["angle_clipped"] is True
    assert result.metadata["effective_theta_min"] == pytest.approx(-math.acos(0.5))
    assert result.metadata["effective_theta_max"] == pytest.approx(math.acos(0.5))
    assert result.metadata["maximum_stretch"] <= 2.0 + 1e-6


def test_valid_mask_uses_source_bounds_and_bottle_mask() -> None:
    image = np.full((50, 100, 3), 200, dtype=np.uint8)
    mask = np.zeros((50, 100), dtype=np.uint8)
    mask[10:40, 35:65] = 1

    result = cylindrical_unwrap(
        image,
        label_bbox=BBox(20, 10, 80, 40),
        geometry=CylinderGeometry(50.0, 35.0, -0.8, 0.8),
        bottle_mask=mask,
        config=UnwrapConfig(border_rgb=(1, 2, 3)),
    )

    assert result.valid_mask is not None
    assert np.any(result.valid_mask)
    assert np.any(~result.valid_mask)
    assert result.image is not None
    assert np.all(result.image[~result.valid_mask] == np.array([1, 2, 3]))


def test_resolution_limit_is_combined_into_remap() -> None:
    result = cylindrical_unwrap(
        coordinate_image(200, 240),
        label_bbox=BBox(20, 20, 220, 180),
        geometry=CylinderGeometry(120.0, 100.0, -0.8, 0.8),
        config=UnwrapConfig(max_long_side=80),
    )

    assert result.image is not None
    assert max(result.image.shape[:2]) == 80
    assert result.metadata["scale"] == pytest.approx(0.5)
    assert result.metadata["sampling_passes"] == 1


def test_stable_mask_estimates_body_center_and_radius() -> None:
    mask = np.zeros((80, 120), dtype=np.uint8)
    mask[:, 20:100] = 1

    estimate = estimate_cylinder_from_mask(
        mask, BBox(30, 10, 90, 70), UnwrapConfig()
    )

    assert estimate.reliable
    assert estimate.geometry is not None
    assert estimate.geometry.cx == pytest.approx(59.5)
    assert estimate.geometry.radius == pytest.approx(39.5)
    assert estimate.stats["usable_row_fraction"] == 1.0


def test_mask_estimator_rejects_sparse_rows() -> None:
    mask = np.zeros((80, 120), dtype=np.uint8)
    mask[20, 20:100] = 1

    estimate = estimate_cylinder_from_mask(
        mask, BBox(30, 10, 90, 70), UnwrapConfig()
    )

    assert not estimate.reliable
    assert estimate.reason == "mask_rows_insufficient"


def test_mask_estimator_rejects_unstable_widths() -> None:
    mask = np.zeros((80, 120), dtype=np.uint8)
    for y in range(10, 70):
        half_width = 35 if y % 2 else 12
        mask[y, 60 - half_width : 60 + half_width] = 1

    estimate = estimate_cylinder_from_mask(
        mask, BBox(50, 10, 70, 70), UnwrapConfig()
    )

    assert not estimate.reliable
    assert estimate.reason == "mask_radius_unstable"


def test_mask_estimator_rejects_disconnected_silhouette_rows() -> None:
    mask = np.zeros((40, 100), dtype=np.uint8)
    mask[5:35, 10:35] = 1
    mask[5:35, 65:90] = 1

    estimate = estimate_cylinder_from_mask(
        mask, BBox(20, 5, 80, 35), UnwrapConfig()
    )

    assert not estimate.reliable
    assert estimate.reason == "mask_rows_disconnected"


def test_invalid_explicit_geometry_can_raise_in_strict_mode() -> None:
    with pytest.raises(GeometryError, match="radius"):
        cylindrical_unwrap(
            np.zeros((20, 30, 3), dtype=np.uint8),
            label_bbox=BBox(2, 2, 28, 18),
            geometry=CylinderGeometry(15.0, 0.0),
            strict=True,
        )


def test_synthetic_cylindrical_grid_is_recovered_without_quality_claim() -> None:
    source_height, source_width = 100, 180
    label_top, label_bottom = 20, 80
    cx, radius = 90.0, 70.0
    theta_min, theta_max = -0.8, 0.8
    flat_width = int(round(radius * (theta_max - theta_min)))
    flat_height = label_bottom - label_top
    flat = np.full((flat_height, flat_width, 3), 240, dtype=np.uint8)
    flat[:, ::14] = (10, 10, 10)
    flat[::12, :] = (80, 80, 80)
    cv2.putText(flat, "WINE 2024", (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    source = np.zeros((source_height, source_width, 3), dtype=np.uint8)
    yy, xx = np.mgrid[:source_height, :source_width]
    ratio = (xx.astype(np.float32) - cx) / radius
    visible = (np.abs(ratio) <= 1.0) & (yy >= label_top) & (yy < label_bottom)
    theta = np.arcsin(np.clip(ratio, -1.0, 1.0))
    map_u = ((theta - theta_min) / (theta_max - theta_min)) * (flat_width - 1)
    map_v = yy.astype(np.float32) - label_top
    rendered = cv2.remap(
        flat,
        map_u.astype(np.float32),
        map_v.astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    source[visible & (theta >= theta_min) & (theta <= theta_max)] = rendered[
        visible & (theta >= theta_min) & (theta <= theta_max)
    ]

    result = cylindrical_unwrap(
        source,
        label_bbox=BBox(40, label_top, 140, label_bottom),
        geometry=CylinderGeometry(cx, radius, theta_min, theta_max),
        config=UnwrapConfig(max_stretch=2.5),
    )

    assert result.image is not None
    recovered = cv2.resize(result.image, (flat_width, flat_height), interpolation=cv2.INTER_LINEAR)
    interior = (slice(2, -2), slice(2, -2))
    mae = np.mean(
        np.abs(
            recovered[interior].astype(np.int16) - flat[interior].astype(np.int16)
        )
    )
    assert mae < 35
