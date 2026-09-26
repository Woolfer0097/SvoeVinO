"""Conservative bottle geometry estimation and cylindrical inverse mapping."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from .errors import GeometryError, ImageInputError
from .models import (
    BBox,
    CylinderGeometry,
    CylindricalResult,
    GeometryEstimate,
    RGBArray,
    UnwrapConfig,
)


def _clamp_bbox(bbox: BBox, width: int, height: int) -> BBox:
    effective = BBox(
        min(max(int(bbox.x_min), 0), width),
        min(max(int(bbox.y_min), 0), height),
        min(max(int(bbox.x_max), 0), width),
        min(max(int(bbox.y_max), 0), height),
    )
    if effective.width <= 0 or effective.height <= 0:
        raise ImageInputError(
            f"Label bounding box is empty after clamping: {effective.to_list()}"
        )
    return effective


def _validate_mask(mask: np.ndarray) -> np.ndarray:
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ImageInputError("Bottle mask must be a two-dimensional NumPy array")
    if mask.dtype != np.bool_ and not np.issubdtype(mask.dtype, np.integer):
        raise ImageInputError("Bottle mask must use bool or an integer dtype")
    return np.ascontiguousarray(mask != 0)


def normalize_bottle_mask(
    mask: np.ndarray,
    *,
    image_shape: tuple[int, int],
    bottle_bbox: BBox | None,
) -> np.ndarray:
    """Return a full-image boolean mask from full-image or bottle-ROI input."""

    boolean_mask = _validate_mask(mask)
    height, width = image_shape
    if boolean_mask.shape == (height, width):
        return boolean_mask
    if bottle_bbox is None:
        raise ImageInputError(
            "Bottle mask shape must match the full image when bottle_bbox is absent"
        )
    if bottle_bbox.width <= 0 or bottle_bbox.height <= 0:
        raise ImageInputError("bottle_bbox must have positive width and height")
    effective_bbox = _clamp_bbox(bottle_bbox, width, height)
    effective_shape = (effective_bbox.height, effective_bbox.width)
    raw_shape = (bottle_bbox.height, bottle_bbox.width)
    if boolean_mask.shape == effective_shape:
        visible_mask = boolean_mask
    elif boolean_mask.shape == raw_shape:
        y_offset = effective_bbox.y_min - bottle_bbox.y_min
        x_offset = effective_bbox.x_min - bottle_bbox.x_min
        visible_mask = boolean_mask[
            y_offset : y_offset + effective_bbox.height,
            x_offset : x_offset + effective_bbox.width,
        ]
    else:
        raise ImageInputError(
            "Bottle mask shape must match either the full image or bottle_bbox; "
            f"expected {raw_shape} (or clipped {effective_shape}), "
            f"got {boolean_mask.shape}"
        )
    full_mask = np.zeros((height, width), dtype=bool)
    full_mask[
        effective_bbox.y_min : effective_bbox.y_max,
        effective_bbox.x_min : effective_bbox.x_max,
    ] = visible_mask
    return full_mask


def _mask_runs(row: np.ndarray) -> int:
    padded = np.pad(row.astype(np.int8, copy=False), (1, 0))
    return int(np.count_nonzero(np.diff(padded) == 1))


def estimate_cylinder_from_mask(
    bottle_mask: np.ndarray,
    label_bbox: BBox,
    config: UnwrapConfig | None = None,
) -> GeometryEstimate:
    """Estimate visible body center/radius from silhouette rows near the label."""

    effective_config = config or UnwrapConfig()
    mask = _validate_mask(bottle_mask)
    height, width = mask.shape
    bbox = _clamp_bbox(label_bbox, width, height)
    row_count = bbox.height
    centers: list[float] = []
    radii: list[float] = []
    disconnected_rows = 0

    for y in range(bbox.y_min, bbox.y_max):
        row = mask[y]
        xs = np.flatnonzero(row)
        if xs.size < effective_config.mask_min_foreground_width:
            continue
        if _mask_runs(row) > effective_config.mask_max_components_per_row:
            disconnected_rows += 1
            continue
        left = float(xs[0])
        right = float(xs[-1])
        centers.append((left + right) / 2.0)
        radii.append((right - left) / 2.0)

    disconnected_fraction = disconnected_rows / max(row_count, 1)
    usable_fraction = len(centers) / max(row_count, 1)
    base_stats: dict[str, Any] = {
        "rows_total": row_count,
        "rows_usable": len(centers),
        "rows_disconnected": disconnected_rows,
        "usable_row_fraction": usable_fraction,
        "disconnected_row_fraction": disconnected_fraction,
    }
    if disconnected_fraction > 1.0 - effective_config.mask_min_usable_row_fraction:
        return GeometryEstimate(None, False, "mask_rows_disconnected", base_stats)
    if usable_fraction < effective_config.mask_min_usable_row_fraction:
        return GeometryEstimate(None, False, "mask_rows_insufficient", base_stats)

    center_values = np.asarray(centers, dtype=np.float64)
    radius_values = np.asarray(radii, dtype=np.float64)
    cx = float(np.median(center_values))
    radius = float(np.median(radius_values))
    center_mad_ratio = float(np.median(np.abs(center_values - cx)) / max(radius, 1e-9))
    radius_cv = float(np.std(radius_values) / max(radius, 1e-9))
    stats = {
        **base_stats,
        "cx": cx,
        "radius": radius,
        "center_mad_ratio": center_mad_ratio,
        "radius_cv": radius_cv,
    }
    if not math.isfinite(cx) or not math.isfinite(radius) or radius <= 0:
        return GeometryEstimate(None, False, "mask_geometry_nonfinite", stats)
    if center_mad_ratio > effective_config.mask_max_center_mad_ratio:
        return GeometryEstimate(None, False, "mask_center_unstable", stats)
    if radius_cv > effective_config.mask_max_radius_cv:
        return GeometryEstimate(None, False, "mask_radius_unstable", stats)
    if bbox.x_max <= cx - radius or bbox.x_min >= cx + radius:
        return GeometryEstimate(None, False, "label_outside_mask_geometry", stats)
    return GeometryEstimate(
        CylinderGeometry(cx=cx, radius=radius), True, None, stats
    )


def _failure(
    reason: str,
    message: str,
    *,
    strict: bool,
    geometry: CylinderGeometry | None,
    metadata: dict[str, Any] | None = None,
) -> CylindricalResult:
    if strict:
        raise GeometryError(message)
    return CylindricalResult(
        image=None,
        valid_mask=None,
        status="skipped",
        reason=reason,
        geometry=geometry,
        metadata=metadata or {},
    )


def _validate_unwrap_config(config: UnwrapConfig) -> str | None:
    if not math.isfinite(config.max_stretch) or config.max_stretch <= 1.0:
        return "max_stretch must be finite and greater than one"
    if not math.isfinite(config.angle_margin) or not 0 <= config.angle_margin < math.pi / 2:
        return "angle_margin must be in [0, pi/2)"
    if config.max_long_side is not None and config.max_long_side < 1:
        return "max_long_side must be positive or None"
    if config.min_output_width < 1 or config.min_output_height < 1:
        return "minimum output dimensions must be positive"
    if len(config.border_rgb) != 3 or any(not 0 <= value <= 255 for value in config.border_rgb):
        return "border_rgb must contain three values in [0, 255]"
    return None


def cylindrical_unwrap(
    image: RGBArray,
    *,
    label_bbox: BBox,
    geometry: CylinderGeometry,
    config: UnwrapConfig | None = None,
    bottle_mask: np.ndarray | None = None,
    strict: bool = False,
) -> CylindricalResult:
    """Unwrap a visible cylindrical label with one inverse `cv2.remap`."""

    effective_config = config or UnwrapConfig()
    config_error = _validate_unwrap_config(effective_config)
    if config_error is not None:
        return _failure(
            "invalid_config",
            config_error,
            strict=strict,
            geometry=geometry,
        )
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ImageInputError("Cylindrical input must be RGB uint8 with shape HxWx3")

    height, width = image.shape[:2]
    bbox = _clamp_bbox(label_bbox, width, height)
    if (
        not math.isfinite(geometry.cx)
        or not math.isfinite(geometry.radius)
        or geometry.radius <= 0
    ):
        return _failure(
            "invalid_radius",
            "Cylinder radius must be finite and positive",
            strict=strict,
            geometry=geometry,
        )
    if (geometry.theta_min is None) != (geometry.theta_max is None):
        return _failure(
            "partial_angle_range",
            "Both theta_min and theta_max must be supplied together",
            strict=strict,
            geometry=geometry,
        )

    angle_source = "explicit"
    if geometry.theta_min is None:
        angle_source = "label_bbox"
        left_ratio = (float(bbox.x_min) - geometry.cx) / geometry.radius
        right_ratio = (float(bbox.x_max) - geometry.cx) / geometry.radius
        if right_ratio <= -1.0 or left_ratio >= 1.0:
            return _failure(
                "label_not_visible",
                "Label bounding box does not intersect the visible cylinder",
                strict=strict,
                geometry=geometry,
            )
        requested_min = math.asin(float(np.clip(left_ratio, -1.0, 1.0)))
        requested_max = math.asin(float(np.clip(right_ratio, -1.0, 1.0)))
    else:
        requested_min = float(geometry.theta_min)
        requested_max = float(geometry.theta_max)

    if (
        not math.isfinite(requested_min)
        or not math.isfinite(requested_max)
        or requested_min >= requested_max
    ):
        return _failure(
            "invalid_angle_range",
            "Cylinder angles must be finite and strictly increasing",
            strict=strict,
            geometry=geometry,
        )

    stretch_angle = math.acos(1.0 / effective_config.max_stretch)
    visible_limit = math.pi / 2.0 - effective_config.angle_margin
    safety_limit = min(stretch_angle, visible_limit)
    theta_min = max(requested_min, -safety_limit)
    theta_max = min(requested_max, safety_limit)
    metadata: dict[str, Any] = {
        "cx": geometry.cx,
        "radius": geometry.radius,
        "label_bbox": bbox.to_list(),
        "angle_source": angle_source,
        "requested_theta_min": requested_min,
        "requested_theta_max": requested_max,
        "effective_theta_min": theta_min,
        "effective_theta_max": theta_max,
        "angle_clipped": theta_min != requested_min or theta_max != requested_max,
        "max_stretch": effective_config.max_stretch,
        "max_long_side": effective_config.max_long_side,
        "border_rgb": list(effective_config.border_rgb),
    }
    if theta_min >= theta_max:
        return _failure(
            "safe_angle_range_empty",
            "No safely visible angle range remains after stretch limiting",
            strict=strict,
            geometry=geometry,
            metadata=metadata,
        )

    flat_width = geometry.radius * (theta_max - theta_min)
    flat_height = float(bbox.height)
    long_side = max(flat_width, flat_height)
    scale = 1.0
    if effective_config.max_long_side is not None and long_side > effective_config.max_long_side:
        scale = float(effective_config.max_long_side) / long_side
    output_width = max(1, int(round(flat_width * scale)))
    output_height = max(1, int(round(flat_height * scale)))
    metadata.update(
        {
            "scale": scale,
            "unscaled_width": flat_width,
            "unscaled_height": flat_height,
            "output_width": output_width,
            "output_height": output_height,
            "horizontal_scale": output_width / flat_width,
            "vertical_scale": output_height / flat_height,
            "maximum_stretch": max(
                1.0 / math.cos(theta_min), 1.0 / math.cos(theta_max)
            ),
            "sampling_passes": 1,
        }
    )
    if (
        output_width < effective_config.min_output_width
        or output_height < effective_config.min_output_height
    ):
        return _failure(
            "output_too_small",
            "Safe cylindrical output is smaller than configured minimum dimensions",
            strict=strict,
            geometry=geometry,
            metadata=metadata,
        )

    u_fraction = (
        np.arange(output_width, dtype=np.float32) + np.float32(0.5)
    ) / np.float32(output_width)
    theta = np.float32(theta_min) + u_fraction * np.float32(theta_max - theta_min)
    x_line = np.float32(geometry.cx) + np.float32(geometry.radius) * np.sin(theta)
    y_line = np.float32(bbox.y_min) + (
        (np.arange(output_height, dtype=np.float32) + np.float32(0.5))
        * np.float32(flat_height / output_height)
    ) - np.float32(0.5)
    map_x = np.broadcast_to(x_line[None, :], (output_height, output_width)).copy()
    map_y = np.broadcast_to(y_line[:, None], (output_height, output_width)).copy()

    valid = (
        np.isfinite(map_x)
        & np.isfinite(map_y)
        & (map_x >= 0.0)
        & (map_x <= width - 1)
        & (map_y >= 0.0)
        & (map_y <= height - 1)
    )
    normalized_mask: np.ndarray | None = None
    if bottle_mask is not None:
        normalized_mask = _validate_mask(bottle_mask)
        if normalized_mask.shape != (height, width):
            raise ImageInputError(
                "Bottle mask passed to cylindrical_unwrap must match the full image"
            )
        sampled_mask = cv2.remap(
            normalized_mask.astype(np.uint8),
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        valid &= sampled_mask != 0

    output = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=tuple(int(value) for value in effective_config.border_rgb),
    )
    output[~valid] = np.asarray(effective_config.border_rgb, dtype=np.uint8)
    metadata["valid_fraction"] = float(np.mean(valid))
    return CylindricalResult(
        image=np.ascontiguousarray(output),
        valid_mask=np.ascontiguousarray(valid),
        status="ok",
        reason=None,
        geometry=CylinderGeometry(
            geometry.cx,
            geometry.radius,
            theta_min,
            theta_max,
        ),
        metadata=metadata,
    )
