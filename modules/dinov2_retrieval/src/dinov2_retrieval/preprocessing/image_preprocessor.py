"""Minimal, dependency-light image preparation for future ML stages."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, UnidentifiedImageError


class ImagePreprocessingError(ValueError):
    """Raised when Pillow cannot safely open or decode an image."""


class ImagePreprocessor:
    """Open a decoded image as RGB without retaining a file handle."""

    def open_rgb(self, image_path: Path) -> Image.Image:
        """Return a fully decoded RGB image owned by the caller."""

        try:
            with Image.open(image_path) as source_image:
                source_image.load()
                return source_image.convert("RGB")
        except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
            raise ImagePreprocessingError(
                f"Cannot safely decode image: {image_path}"
            ) from exc
