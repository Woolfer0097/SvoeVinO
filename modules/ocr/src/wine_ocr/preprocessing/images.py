"""Image loading and conservative OCR preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any

from ..config import OCRConfig
from ..exceptions import ImageDecodeError


@dataclass(frozen=True)
class ImageVariant:
    """Prepared image variant passed to the OCR backend."""

    name: str
    image: Any
    origin_x: int = 0
    origin_y: int = 0


def decode_image(data: bytes, config: OCRConfig) -> Any:
    """Decode uploaded image bytes into an RGB Pillow image."""

    if not data:
        raise ImageDecodeError("Uploaded image is empty")
    if len(data) > config.max_upload_size_bytes:
        raise ImageDecodeError(
            "Uploaded image is too large: "
            f"{len(data)} bytes; maximum is {config.max_upload_size_bytes}"
        )

    try:
        from PIL import Image, ImageOps, UnidentifiedImageError
    except ImportError as exc:
        raise ImageDecodeError("Pillow is required to decode uploaded images") from exc

    try:
        with Image.open(BytesIO(data)) as source:
            image = ImageOps.exif_transpose(source)
            image.load()
            return image.convert("RGB")
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ImageDecodeError("Cannot decode uploaded image") from exc


def build_image_variants(image: Any, config: OCRConfig) -> list[ImageVariant]:
    """Build conservative variants: full image and optional central crop."""

    variants = [ImageVariant(name="full", image=image.copy())]
    if not config.enable_central_crop:
        return variants

    width, height = image.size
    if width <= 1 or height <= 1:
        return variants

    crop_width = max(1, int(width * config.central_crop_fraction))
    crop_height = max(1, int(height * config.central_crop_fraction))
    left = max(0, (width - crop_width) // 2)
    top = max(0, (height - crop_height) // 2)
    right = min(width, left + crop_width)
    bottom = min(height, top + crop_height)

    if right - left == width and bottom - top == height:
        return variants

    variants.append(
        ImageVariant(
            name="central_crop",
            image=image.crop((left, top, right, bottom)),
            origin_x=left,
            origin_y=top,
        )
    )
    return variants
