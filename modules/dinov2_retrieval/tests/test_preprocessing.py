from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from dinov2_retrieval.preprocessing.image_preprocessor import (
    ImagePreprocessingError,
    ImagePreprocessor,
)


def test_preprocessor_decodes_and_converts_to_rgb(tmp_path: Path) -> None:
    image_path = tmp_path / "rgba.png"
    Image.new("RGBA", (7, 5), color=(20, 40, 60, 128)).save(
        image_path, format="PNG"
    )

    with ImagePreprocessor().open_rgb(image_path) as image:
        assert image.mode == "RGB"
        assert image.size == (7, 5)


def test_exif_orientation_is_applied(tmp_path: Path) -> None:
    # Phones save portrait photos as landscape pixels plus EXIF orientation 6.
    image_path = tmp_path / "phone.jpg"
    image = Image.new("RGB", (40, 30), color=(200, 0, 0))
    exif = image.getexif()
    exif[0x0112] = 6
    image.save(image_path, exif=exif)

    with ImagePreprocessor().open_rgb(image_path) as upright:
        assert upright.size == (30, 40)


def test_transparency_is_flattened_onto_white(tmp_path: Path) -> None:
    image_path = tmp_path / "bottle.png"
    image = Image.new("RGBA", (4, 4), color=(0, 0, 0, 0))
    image.putpixel((1, 1), (200, 0, 0, 255))
    image.save(image_path)

    with ImagePreprocessor().open_rgb(image_path) as flattened:
        assert flattened.mode == "RGB"
        assert flattened.getpixel((0, 0)) == (255, 255, 255)
        assert flattened.getpixel((1, 1)) == (200, 0, 0)


def test_palette_transparency_is_flattened(tmp_path: Path) -> None:
    image_path = tmp_path / "logo.gif.png"
    image = Image.new("P", (2, 1))
    image.putpalette([0, 0, 0, 10, 20, 30])
    image.putpixel((1, 0), 1)
    image.info["transparency"] = 0
    image.save(image_path, transparency=0)

    with ImagePreprocessor().open_rgb(image_path) as flattened:
        assert flattened.getpixel((0, 0)) == (255, 255, 255)
        assert flattened.getpixel((1, 0)) == (10, 20, 30)


def test_undecodable_file_raises_preprocessing_error(tmp_path: Path) -> None:
    image_path = tmp_path / "broken.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"garbage" * 10)

    with pytest.raises(ImagePreprocessingError):
        ImagePreprocessor().open_rgb(image_path)
