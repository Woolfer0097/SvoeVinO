"""Lossless image assets and deterministic machine-readable reports."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .models import ComparisonVariant, PreprocessingResult, RGBArray
from .visualization import make_contact_sheet

VARIANT_FILENAMES = {
    "A": "A_original.png",
    "B": "B_photometric.png",
    "C": "C_cylindrical.png",
    "D": "D_cylindrical_photometric.png",
    "E": "E_dewarpnet.png",
}


def to_jsonable(value: Any) -> Any:
    """Convert report metadata without silently embedding image arrays."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        raise TypeError("Image arrays must be saved as assets, not embedded in JSON")
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_jsonable(item) for item in value]
    raise TypeError(f"Unsupported JSON report value: {type(value).__name__}")


def write_json(path: Path, value: Any) -> None:
    """Write stable UTF-8 JSON with sorted keys."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            to_jsonable(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_rgb(path: Path, image: RGBArray) -> None:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Output image must be RGB uint8 with shape HxWx3")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise OSError(f"OpenCV could not write image: {path}")


def _write_mask(path: Path, mask: np.ndarray) -> None:
    if mask.ndim != 2:
        raise ValueError("Output valid mask must be two-dimensional")
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.where(mask, 255, 0).astype(np.uint8)
    if not cv2.imwrite(str(path), pixels):
        raise OSError(f"OpenCV could not write mask: {path}")


def _safe_relative_stem(relative_stem: Path) -> Path:
    if relative_stem.is_absolute() or any(part == ".." for part in relative_stem.parts):
        raise ValueError("Output stem must be a safe relative path")
    if not relative_stem.parts:
        raise ValueError("Output stem must not be empty")
    return relative_stem


def write_comparison(
    output_root: Path,
    relative_stem: Path,
    result: PreprocessingResult,
    variants: Mapping[str, ComparisonVariant],
) -> dict[str, Any]:
    """Write A-E, model branches, masks, contact sheet, and per-image JSON."""

    stem = _safe_relative_stem(relative_stem)
    target = output_root / stem
    target.mkdir(parents=True, exist_ok=True)
    assets: dict[str, str] = {}
    variant_metadata: dict[str, Any] = {}
    for code in "ABCDE":
        variant = variants[code]
        asset = None
        if variant.image is not None:
            filename = VARIANT_FILENAMES[code]
            _write_rgb(target / filename, variant.image)
            asset = filename
            assets[code] = filename
        variant_metadata[code] = {
            "status": variant.status,
            "reason": variant.reason,
            "shape": list(variant.image.shape) if variant.image is not None else None,
            "asset": asset,
            "metadata": variant.metadata,
        }

    _write_rgb(target / "embedding.png", result.embedding_image)
    _write_rgb(target / "ocr.png", result.ocr_image)
    assets["embedding"] = "embedding.png"
    assets["ocr"] = "ocr.png"
    if result.cylindrical_valid_mask is not None:
        _write_mask(
            target / "cylindrical_valid_mask.png", result.cylindrical_valid_mask
        )
        assets["cylindrical_valid_mask"] = "cylindrical_valid_mask.png"
    if result.dewarpnet_valid_mask is not None:
        _write_mask(target / "dewarpnet_valid_mask.png", result.dewarpnet_valid_mask)
        assets["dewarpnet_valid_mask"] = "dewarpnet_valid_mask.png"

    sheet = make_contact_sheet(variants)
    _write_rgb(target / "comparison.png", sheet)
    assets["comparison"] = "comparison.png"
    metadata = {
        "schema_version": "1.0",
        "output_directory": stem.as_posix(),
        "pipeline": result.metadata,
        "variants": variant_metadata,
        "assets": assets,
    }
    write_json(target / "metadata.json", metadata)
    return {
        "status": "ok",
        "output_directory": stem.as_posix(),
        "variants": variant_metadata,
        "assets": assets,
        "pipeline": result.metadata,
    }

