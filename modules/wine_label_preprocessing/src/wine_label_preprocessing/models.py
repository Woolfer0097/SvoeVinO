"""Typed configuration and result contracts for preprocessing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np

ColorOrder: TypeAlias = Literal["RGB", "BGR"]
RGBArray: TypeAlias = np.ndarray


@dataclass(frozen=True, slots=True)
class BBox:
    """Half-open integer bounding box in the EXIF-oriented full image."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min

    def to_list(self) -> list[int]:
        return [self.x_min, self.y_min, self.x_max, self.y_max]


@dataclass(frozen=True, slots=True)
class CylinderGeometry:
    """Visible bottle-body geometry in full oriented-image coordinates."""

    cx: float
    radius: float
    theta_min: float | None = None
    theta_max: float | None = None


@dataclass(slots=True)
class ImageAnnotations:
    """Optional external localization and cylinder information."""

    label_bbox: BBox | None = None
    bottle_bbox: BBox | None = None
    bottle_mask: np.ndarray | None = None
    cylinder: CylinderGeometry | None = None


@dataclass(frozen=True, slots=True)
class PhotometricConfig:
    """Mild, independently switchable luminance operations."""

    brightness_enabled: bool = False
    target_luminance: float = 140.0
    min_gain: float = 0.9
    max_gain: float = 1.1
    clahe_enabled: bool = False
    clahe_clip_limit: float = 1.5
    clahe_grid_size: tuple[int, int] = (8, 8)
    unsharp_enabled: bool = False
    unsharp_sigma: float = 1.0
    unsharp_amount: float = 0.2


@dataclass(frozen=True, slots=True)
class UnwrapConfig:
    """Safety and confidence thresholds for cylindrical unwrapping."""

    enabled: bool = True
    max_stretch: float = 2.5
    angle_margin: float = 0.02
    max_long_side: int | None = 2400
    min_output_width: int = 16
    min_output_height: int = 16
    border_rgb: tuple[int, int, int] = (0, 0, 0)
    mask_min_foreground_width: int = 16
    mask_min_usable_row_fraction: float = 0.7
    mask_max_center_mad_ratio: float = 0.05
    mask_max_radius_cv: float = 0.12
    mask_max_components_per_row: int = 1


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    """Configuration for baseline, OCR, embedding, and geometry branches."""

    embedding_max_long_side: int | None = 2400
    ocr_max_long_side: int | None = 2400
    embedding_photometric: PhotometricConfig = field(
        default_factory=PhotometricConfig
    )
    ocr_photometric: PhotometricConfig = field(default_factory=PhotometricConfig)
    unwrap: UnwrapConfig = field(default_factory=UnwrapConfig)


@dataclass(slots=True)
class LoadedImage:
    """Decoded oriented RGB pixels and JSON-friendly loading metadata."""

    array: RGBArray
    metadata: dict[str, Any]


@dataclass(slots=True)
class PreprocessingResult:
    """All safe preprocessing branches for one source image."""

    original_crop: RGBArray
    embedding_image: RGBArray
    ocr_image: RGBArray
    cylindrical_image: RGBArray | None = None
    cylindrical_valid_mask: np.ndarray | None = None
    dewarpnet_image: RGBArray | None = None
    dewarpnet_valid_mask: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GeometryEstimate:
    """Robust bottle-body estimate and the evidence used to accept it."""

    geometry: CylinderGeometry | None
    reliable: bool
    reason: str | None
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CylindricalResult:
    """Output of an inverse cylindrical map, including its valid pixels."""

    image: RGBArray | None
    valid_mask: np.ndarray | None
    status: Literal["ok", "skipped"]
    reason: str | None
    geometry: CylinderGeometry | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DewarpNetConfig:
    """Filesystem and sampling configuration for an official checkout."""

    official_repo: Path
    wc_checkpoint: Path
    bm_checkpoint: Path
    device: str = "auto"
    align_corners: bool = False
    blur_backward_map: bool = True


@dataclass(slots=True)
class DewarpNetResult:
    """Honest outcome of optional DewarpNet inference."""

    image: RGBArray | None
    valid_mask: np.ndarray | None
    status: Literal["ok", "unavailable", "failed"]
    reason: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OCRTruth:
    """Ground-truth text fields used by the optional benchmark."""

    full_text: str
    name: str
    producer: str
    year: str


@dataclass(frozen=True, slots=True)
class OCRPrediction:
    """Raw OCR output returned by an injected backend."""

    full_text: str
    name: str
    producer: str
    year: str


@dataclass(slots=True)
class ComparisonVariant:
    """One named A-E comparison image or an explicit missing outcome."""

    code: Literal["A", "B", "C", "D", "E"]
    image: RGBArray | None
    status: Literal["ok", "skipped", "unavailable", "failed", "not_run"]
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BenchmarkSample:
    """One query with split, series, optional geometry, and optional truth."""

    sample_id: str
    image: Any
    series_id: str
    split: Literal["tuning", "test"]
    annotations: ImageAnnotations | None = None
    conditions: tuple[str, ...] = ()
    truth: OCRTruth | None = None
    catalog_id: str | None = None
    input_color_order: ColorOrder | None = "RGB"


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Warm benchmark configuration shared by every sample and variant."""

    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    mild_photometric: PhotometricConfig = field(
        default_factory=lambda: PhotometricConfig(
            brightness_enabled=True,
            clahe_enabled=True,
        )
    )
    warmup_runs: int = 1
    measured_runs: int = 5
