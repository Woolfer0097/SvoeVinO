"""Decode local images before feature matching."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError


class ImagePreprocessingError(ValueError):
    """Raised when Pillow cannot safely open or decode an image."""


class ImagePreprocessor:
    """Open a decoded RGB image owned by the caller."""

    def open_rgb(self, image_path: Path) -> Image.Image:
        """Return a fully decoded RGB image with EXIF orientation applied."""

        try:
            with Image.open(image_path) as source_image:
                source_image.load()
                oriented = ImageOps.exif_transpose(source_image)
                return oriented.convert("RGB")
        except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
            raise ImagePreprocessingError(
                f"Cannot safely decode image: {image_path}"
            ) from exc
