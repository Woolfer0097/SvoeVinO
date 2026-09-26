"""Conservative, opt-in luminance processing for OCR experiments."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .errors import PreprocessingError
from .models import PhotometricConfig, RGBArray


def mild_ocr_profile(*, unsharp: bool = False) -> PhotometricConfig:
    """Return the explicit mild profile used by comparison variants B and D."""

    return PhotometricConfig(
        brightness_enabled=True,
        target_luminance=140.0,
        min_gain=0.9,
        max_gain=1.1,
        clahe_enabled=True,
        clahe_clip_limit=1.5,
        clahe_grid_size=(8, 8),
        unsharp_enabled=unsharp,
        unsharp_sigma=1.0,
        unsharp_amount=0.2,
    )


def _validate_config(config: PhotometricConfig) -> None:
    if not (0 < config.min_gain <= config.max_gain):
        raise PreprocessingError("Brightness gain bounds must be positive and ordered")
    if not 0 <= config.target_luminance <= 255:
        raise PreprocessingError("Target luminance must be between 0 and 255")
    if config.clahe_clip_limit <= 0:
        raise PreprocessingError("CLAHE clip limit must be positive")
    if len(config.clahe_grid_size) != 2 or any(
        int(value) < 1 for value in config.clahe_grid_size
    ):
        raise PreprocessingError("CLAHE grid size values must be positive")
    if config.unsharp_sigma <= 0:
        raise PreprocessingError("Unsharp sigma must be positive")
    if config.unsharp_amount < 0:
        raise PreprocessingError("Unsharp amount must not be negative")


def apply_photometric(
    image: RGBArray,
    config: PhotometricConfig,
) -> tuple[RGBArray, list[dict[str, Any]]]:
    """Apply configured mild operations while preserving RGB uint8 shape."""

    _validate_config(config)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise PreprocessingError("Photometric input must be RGB uint8 with shape HxWx3")

    operations: list[dict[str, Any]] = []
    luminance_changed = config.brightness_enabled or config.clahe_enabled
    if luminance_changed:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        luminance = lab[..., 0]

        if config.brightness_enabled:
            median = float(np.median(luminance))
            raw_gain = config.target_luminance / max(median, 1.0)
            gain = float(np.clip(raw_gain, config.min_gain, config.max_gain))
            luminance = np.clip(
                luminance.astype(np.float32) * gain, 0, 255
            ).astype(np.uint8)
            operations.append(
                {
                    "name": "brightness",
                    "median_luminance": median,
                    "target_luminance": config.target_luminance,
                    "gain": gain,
                }
            )

        if config.clahe_enabled:
            clahe = cv2.createCLAHE(
                clipLimit=float(config.clahe_clip_limit),
                tileGridSize=tuple(int(value) for value in config.clahe_grid_size),
            )
            luminance = clahe.apply(luminance)
            operations.append(
                {
                    "name": "clahe",
                    "clip_limit": config.clahe_clip_limit,
                    "grid_size": list(config.clahe_grid_size),
                }
            )

        lab[..., 0] = luminance
        output = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        output = image.copy()

    if config.unsharp_enabled:
        blurred = cv2.GaussianBlur(
            output,
            (0, 0),
            sigmaX=float(config.unsharp_sigma),
            sigmaY=float(config.unsharp_sigma),
        )
        output = cv2.addWeighted(
            output,
            1.0 + float(config.unsharp_amount),
            blurred,
            -float(config.unsharp_amount),
            0,
        )
        operations.append(
            {
                "name": "unsharp",
                "sigma": config.unsharp_sigma,
                "amount": config.unsharp_amount,
            }
        )

    return np.ascontiguousarray(output, dtype=np.uint8), operations

