"""High-level preprocessing pipeline with safe, explicit branches."""

from __future__ import annotations

from dataclasses import asdict
from time import perf_counter_ns
from typing import Any

from .geometry import (
    cylindrical_unwrap,
    estimate_cylinder_from_mask,
    normalize_bottle_mask,
)
from .image_io import ImageInput, crop_rgb, limit_resolution, load_rgb
from .models import (
    BBox,
    ColorOrder,
    ImageAnnotations,
    PreprocessingConfig,
    PreprocessingResult,
)
from .photometry import apply_photometric


def _elapsed_ms(start_ns: int) -> float:
    return (perf_counter_ns() - start_ns) / 1_000_000.0


def _choose_crop(
    image: Any,
    annotations: ImageAnnotations,
) -> tuple[Any, BBox, str]:
    if annotations.label_bbox is not None:
        crop, bbox = crop_rgb(image, annotations.label_bbox)
        return crop, bbox, "label_bbox"
    if annotations.bottle_bbox is not None:
        crop, bbox = crop_rgb(image, annotations.bottle_bbox)
        return crop, bbox, "bottle_bbox"
    height, width = image.shape[:2]
    bbox = BBox(0, 0, width, height)
    crop, _ = crop_rgb(image, bbox)
    return crop, bbox, "full_image"


def preprocess(
    image: ImageInput,
    *,
    annotations: ImageAnnotations | None = None,
    config: PreprocessingConfig | None = None,
    input_color_order: ColorOrder | None = None,
    dewarpnet: Any | None = None,
) -> PreprocessingResult:
    """Create safe baseline branches and structured operation metadata."""

    total_start = perf_counter_ns()
    effective_annotations = annotations or ImageAnnotations()
    effective_config = config or PreprocessingConfig()
    timings: dict[str, float] = {}

    stage_start = perf_counter_ns()
    loaded = load_rgb(image, input_color_order=input_color_order)
    timings["load"] = _elapsed_ms(stage_start)

    stage_start = perf_counter_ns()
    original_crop, crop_bbox, crop_source = _choose_crop(
        loaded.array, effective_annotations
    )
    timings["crop"] = _elapsed_ms(stage_start)

    stage_start = perf_counter_ns()
    embedding_base, embedding_scale = limit_resolution(
        original_crop, effective_config.embedding_max_long_side
    )
    embedding_image, embedding_operations = apply_photometric(
        embedding_base, effective_config.embedding_photometric
    )
    timings["embedding_branch"] = _elapsed_ms(stage_start)

    stage_start = perf_counter_ns()
    ocr_base, ocr_scale = limit_resolution(
        original_crop, effective_config.ocr_max_long_side
    )
    ocr_image, ocr_operations = apply_photometric(
        ocr_base, effective_config.ocr_photometric
    )
    timings["ocr_branch"] = _elapsed_ms(stage_start)

    cylindrical_image = None
    cylindrical_valid_mask = None
    geometry_estimate_metadata: dict[str, Any] | None = None
    geometry_source: str | None = None
    cylindrical_metadata: dict[str, Any]
    stage_start = perf_counter_ns()
    if effective_annotations.label_bbox is None:
        cylindrical_metadata = {"status": "skipped", "reason": "label_bbox_missing"}
    elif not effective_config.unwrap.enabled:
        cylindrical_metadata = {"status": "skipped", "reason": "disabled"}
    elif effective_annotations.cylinder is None and effective_annotations.bottle_mask is None:
        cylindrical_metadata = {"status": "skipped", "reason": "geometry_missing"}
    else:
        _, effective_label_bbox = crop_rgb(
            loaded.array, effective_annotations.label_bbox
        )
        full_mask = None
        geometry = effective_annotations.cylinder
        if geometry is not None:
            geometry_source = "explicit"
        else:
            full_mask = normalize_bottle_mask(
                effective_annotations.bottle_mask,
                image_shape=loaded.array.shape[:2],
                bottle_bbox=effective_annotations.bottle_bbox,
            )
            estimate = estimate_cylinder_from_mask(
                full_mask,
                effective_label_bbox,
                effective_config.unwrap,
            )
            geometry_estimate_metadata = {
                "reliable": estimate.reliable,
                "reason": estimate.reason,
                **estimate.stats,
            }
            geometry = estimate.geometry
            geometry_source = "mask"

        if geometry is None:
            cylindrical_metadata = {
                "status": "skipped",
                "reason": geometry_estimate_metadata["reason"],
                "geometry_source": geometry_source,
                "geometry_estimate": geometry_estimate_metadata,
            }
        else:
            if full_mask is None and effective_annotations.bottle_mask is not None:
                full_mask = normalize_bottle_mask(
                    effective_annotations.bottle_mask,
                    image_shape=loaded.array.shape[:2],
                    bottle_bbox=effective_annotations.bottle_bbox,
                )
            cylindrical = cylindrical_unwrap(
                loaded.array,
                label_bbox=effective_label_bbox,
                geometry=geometry,
                config=effective_config.unwrap,
                bottle_mask=full_mask,
            )
            cylindrical_image = cylindrical.image
            cylindrical_valid_mask = cylindrical.valid_mask
            cylindrical_metadata = {
                "status": cylindrical.status,
                "reason": cylindrical.reason,
                "geometry_source": geometry_source,
                "geometry_estimate": geometry_estimate_metadata,
                **cylindrical.metadata,
            }
    timings["cylindrical"] = _elapsed_ms(stage_start)

    dewarpnet_image = None
    dewarpnet_valid_mask = None
    stage_start = perf_counter_ns()
    if dewarpnet is None:
        dewarpnet_metadata: dict[str, Any] = {
            "status": "not_configured",
            "reason": "adapter_not_supplied",
        }
    else:
        dewarpnet_result = dewarpnet.unwrap(original_crop)
        dewarpnet_image = dewarpnet_result.image
        dewarpnet_valid_mask = dewarpnet_result.valid_mask
        dewarpnet_metadata = {
            "status": dewarpnet_result.status,
            "reason": dewarpnet_result.reason,
            **dewarpnet_result.metadata,
        }
    timings["dewarpnet"] = _elapsed_ms(stage_start)

    metadata: dict[str, Any] = {
        "schema_version": "1.0",
        "config": asdict(effective_config),
        "annotations": {
            "label_bbox": (
                effective_annotations.label_bbox.to_list()
                if effective_annotations.label_bbox is not None
                else None
            ),
            "bottle_bbox": (
                effective_annotations.bottle_bbox.to_list()
                if effective_annotations.bottle_bbox is not None
                else None
            ),
            "bottle_mask_provided": effective_annotations.bottle_mask is not None,
            "bottle_mask_shape": (
                list(effective_annotations.bottle_mask.shape)
                if effective_annotations.bottle_mask is not None
                else None
            ),
            "cylinder": (
                asdict(effective_annotations.cylinder)
                if effective_annotations.cylinder is not None
                else None
            ),
        },
        "image": loaded.metadata,
        "crop": {
            "source": crop_source,
            "bbox": crop_bbox.to_list(),
            "shape": list(original_crop.shape),
            "photometric_operations": [],
            "resized": False,
        },
        "embedding": {
            "shape": list(embedding_image.shape),
            "scale": embedding_scale,
            "photometric_operations": embedding_operations,
        },
        "ocr": {
            "shape": list(ocr_image.shape),
            "scale": ocr_scale,
            "photometric_operations": ocr_operations,
        },
        "cylindrical": cylindrical_metadata,
        "dewarpnet": dewarpnet_metadata,
        "timings_ms": timings,
    }
    timings["total"] = _elapsed_ms(total_start)

    return PreprocessingResult(
        original_crop=original_crop,
        embedding_image=embedding_image,
        ocr_image=ocr_image,
        cylindrical_image=cylindrical_image,
        cylindrical_valid_mask=cylindrical_valid_mask,
        dewarpnet_image=dewarpnet_image,
        dewarpnet_valid_mask=dewarpnet_valid_mask,
        metadata=metadata,
    )
