"""Minimal, dependency-light image preparation shared by reference and query photos."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

# Transparent areas (e.g. catalog product shots) are filled with this color,
# so the result does not depend on the hidden RGB values under alpha = 0.
TRANSPARENT_BACKGROUND = (255, 255, 255)


class ImagePreprocessingError(ValueError):
    """Raised when Pillow cannot safely open or decode an image."""


class ImagePreprocessor:
    """Open a decoded image as upright RGB without retaining a file handle."""

    def open_rgb(self, image_path: Path) -> Image.Image:
        """Return a fully decoded RGB image owned by the caller.

        The EXIF orientation is applied (phones store portrait photos as
        rotated landscape pixels) and transparency is flattened onto white.
        """

        try:
            with Image.open(image_path) as source_image:
                source_image.load()
                upright = ImageOps.exif_transpose(source_image)
                return _to_rgb(upright)
        except (
            OSError,
            ValueError,
            SyntaxError,
            UnidentifiedImageError,
            Image.DecompressionBombError,
        ) as exc:
            raise ImagePreprocessingError(
                f"Cannot safely decode image: {image_path}"
            ) from exc


def _to_rgb(image: Image.Image) -> Image.Image:
    has_alpha = image.mode in ("RGBA", "LA", "PA") or (
        image.mode == "P" and "transparency" in image.info
    )
    if not has_alpha:
        return image.convert("RGB")
    background = Image.new("RGBA", image.size, (*TRANSPARENT_BACKGROUND, 255))
    return Image.alpha_composite(background, image.convert("RGBA")).convert("RGB")
