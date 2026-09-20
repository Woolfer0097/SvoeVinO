from __future__ import annotations

from pathlib import Path

from PIL import Image

from dinov2_retrieval.preprocessing.image_preprocessor import ImagePreprocessor


def test_preprocessor_decodes_and_converts_to_rgb(tmp_path: Path) -> None:
    image_path = tmp_path / "rgba.png"
    Image.new("RGBA", (7, 5), color=(20, 40, 60, 128)).save(
        image_path, format="PNG"
    )

    with ImagePreprocessor().open_rgb(image_path) as image:
        assert image.mode == "RGB"
        assert image.size == (7, 5)
