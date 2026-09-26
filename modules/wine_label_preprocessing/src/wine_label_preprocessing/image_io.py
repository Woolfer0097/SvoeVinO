"""EXIF-aware image loading, validated cropping, and proportional limiting."""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .errors import ImageInputError
from .models import BBox, ColorOrder, LoadedImage, RGBArray

ImageInput: TypeAlias = str | Path | Image.Image | np.ndarray


def _validate_rgb_array(image: np.ndarray) -> None:
    if image.dtype != np.uint8:
        raise ImageInputError(f"NumPy image must use uint8, got {image.dtype}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ImageInputError(
            "NumPy image must have shape (height, width, 3), "
            f"got {image.shape}"
        )
    if image.shape[0] < 1 or image.shape[1] < 1:
        raise ImageInputError("Image dimensions must be positive")


def _pillow_to_loaded(source: Image.Image, *, source_name: str) -> LoadedImage:
    encoded_size = source.size
    orientation = int(source.getexif().get(274, 1))
    oriented = ImageOps.exif_transpose(source)
    rgb = np.asarray(oriented.convert("RGB"), dtype=np.uint8)
    array = np.ascontiguousarray(rgb)
    return LoadedImage(
        array=array,
        metadata={
            "source": source_name,
            "input_type": "pillow",
            "input_color_order": "RGB",
            "encoded_size": [encoded_size[0], encoded_size[1]],
            "oriented_size": [array.shape[1], array.shape[0]],
            "exif_orientation": orientation,
            "exif_transposed": orientation not in (0, 1),
        },
    )


def load_rgb(
    image: ImageInput,
    *,
    input_color_order: ColorOrder | None = None,
) -> LoadedImage:
    """Return fully owned, contiguous RGB pixels after EXIF orientation."""

    if isinstance(image, np.ndarray):
        if input_color_order not in ("RGB", "BGR"):
            raise ImageInputError(
                "NumPy input requires an explicit RGB or BGR color order"
            )
        _validate_rgb_array(image)
        rgb = image if input_color_order == "RGB" else image[..., ::-1]
        array = np.ascontiguousarray(rgb)
        return LoadedImage(
            array=array.copy(),
            metadata={
                "source": "numpy",
                "input_type": "numpy",
                "input_color_order": input_color_order,
                "encoded_size": [array.shape[1], array.shape[0]],
                "oriented_size": [array.shape[1], array.shape[0]],
                "exif_orientation": None,
                "exif_transposed": False,
            },
        )

    if isinstance(image, Image.Image):
        return _pillow_to_loaded(image, source_name="PIL.Image")

    path = Path(image)
    try:
        with Image.open(path) as source:
            source.load()
            loaded = _pillow_to_loaded(source, source_name=str(path))
            loaded.metadata["input_type"] = "path"
            return loaded
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ImageInputError(f"Cannot safely decode image: {path}") from exc


def crop_rgb(image: RGBArray, bbox: BBox) -> tuple[RGBArray, BBox]:
    """Clamp a half-open box to the image and return an owned crop."""

    _validate_rgb_array(image)
    height, width = image.shape[:2]
    effective = BBox(
        x_min=min(max(int(bbox.x_min), 0), width),
        y_min=min(max(int(bbox.y_min), 0), height),
        x_max=min(max(int(bbox.x_max), 0), width),
        y_max=min(max(int(bbox.y_max), 0), height),
    )
    if effective.x_min >= effective.x_max or effective.y_min >= effective.y_max:
        raise ImageInputError(
            f"Bounding box is empty after clamping: {effective.to_list()}"
        )
    return np.ascontiguousarray(
        image[effective.y_min : effective.y_max, effective.x_min : effective.x_max]
    ).copy(), effective


def limit_resolution(
    image: RGBArray,
    max_long_side: int | None,
) -> tuple[RGBArray, float]:
    """Downscale once with preserved aspect ratio; never enlarge."""

    _validate_rgb_array(image)
    if max_long_side is not None and max_long_side < 1:
        raise ImageInputError("max_long_side must be positive or None")
    height, width = image.shape[:2]
    if max_long_side is None or max(height, width) <= max_long_side:
        return image.copy(), 1.0
    scale = float(max_long_side) / float(max(height, width))
    output_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    resized = cv2.resize(image, output_size, interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(resized), scale

