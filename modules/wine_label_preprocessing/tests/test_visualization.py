from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wine_label_preprocessing.benchmark import create_comparison_variants
from wine_label_preprocessing.models import ComparisonVariant
from wine_label_preprocessing.pipeline import preprocess
from wine_label_preprocessing.reporting import write_comparison, write_json
from wine_label_preprocessing.visualization import make_contact_sheet


def test_contact_sheet_marks_missing_variant_and_preserves_contract() -> None:
    variants = {
        "A": ComparisonVariant(
            "A", np.zeros((20, 30, 3), dtype=np.uint8), "ok"
        ),
        "E": ComparisonVariant(
            "E", None, "unavailable", "DewarpNet not configured"
        ),
    }

    sheet = make_contact_sheet(variants, tile_width=120, tile_height=100)

    assert sheet.shape == (100, 120 * 5, 3)
    assert sheet.dtype == np.uint8
    assert sheet.flags.c_contiguous
    assert np.any(sheet[:, -120:] != 0)


def test_write_comparison_saves_images_masks_and_json(tmp_path: Path) -> None:
    image = np.full((20, 30, 3), 100, dtype=np.uint8)
    result = preprocess(image, input_color_order="RGB")
    variants = create_comparison_variants(result)

    entry = write_comparison(tmp_path, Path("nested/query"), result, variants)

    target = tmp_path / "nested" / "query"
    assert (target / "A_original.png").is_file()
    assert (target / "B_photometric.png").is_file()
    assert (target / "embedding.png").is_file()
    assert (target / "ocr.png").is_file()
    assert (target / "comparison.png").is_file()
    assert (target / "metadata.json").is_file()
    assert entry["variants"]["C"]["status"] == "skipped"
    metadata = json.loads((target / "metadata.json").read_text())
    assert "original_crop" not in metadata
    assert metadata["variants"]["E"]["status"] == "not_run"


def test_json_writer_converts_numpy_scalars_and_paths(tmp_path: Path) -> None:
    path = tmp_path / "report.json"

    write_json(path, {"number": np.float32(1.5), "path": Path("a/b")})

    assert json.loads(path.read_text()) == {"number": 1.5, "path": "a/b"}
