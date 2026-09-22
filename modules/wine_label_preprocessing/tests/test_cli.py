from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from wine_label_preprocessing.cli import main


def write_test_png(path: Path, size: tuple[int, int] = (30, 20)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(
        np.full((size[1], size[0], 3), 127, dtype=np.uint8), mode="RGB"
    ).save(path)
    return path


def test_process_file_writes_variants_and_report(tmp_path: Path) -> None:
    input_path = write_test_png(tmp_path / "input.png")
    output = tmp_path / "out"

    exit_code = main(["process", str(input_path), "--output-dir", str(output)])

    assert exit_code == 0
    assert (output / "input" / "A_original.png").is_file()
    assert (output / "input" / "B_photometric.png").is_file()
    assert (output / "input" / "comparison.png").is_file()
    report = json.loads((output / "report.json").read_text())
    assert report["schema_version"] == "1.0"
    assert report["images"][0]["variants"]["C"]["status"] == "skipped"
    assert report["images"][0]["variants"]["E"]["status"] == "not_run"


def test_process_folder_preserves_relative_paths(tmp_path: Path) -> None:
    input_dir = tmp_path / "images"
    write_test_png(input_dir / "front.png")
    write_test_png(input_dir / "nested" / "side.jpg")
    (input_dir / "notes.txt").write_text("ignored")
    output = tmp_path / "out"

    exit_code = main(["process", str(input_dir), "--output-dir", str(output)])

    assert exit_code == 0
    assert (output / "front" / "A_original.png").is_file()
    assert (output / "nested" / "side" / "A_original.png").is_file()
    report = json.loads((output / "report.json").read_text())
    assert [item["input"] for item in report["images"]] == [
        "front.png",
        "nested/side.jpg",
    ]


def test_process_with_explicit_geometry_writes_cylindrical_variant(
    tmp_path: Path,
) -> None:
    input_path = write_test_png(tmp_path / "bottle.png", size=(100, 60))
    output = tmp_path / "out"

    exit_code = main(
        [
            "process",
            str(input_path),
            "--output-dir",
            str(output),
            "--label-bbox",
            "20",
            "10",
            "80",
            "50",
            "--cx",
            "50",
            "--radius",
            "42",
            "--theta-min",
            "-0.6",
            "--theta-max",
            "0.6",
        ]
    )

    assert exit_code == 0
    assert (output / "bottle" / "C_cylindrical.png").is_file()
    assert (output / "bottle" / "cylindrical_valid_mask.png").is_file()


def test_benchmark_manifest_writes_not_run_model_metrics(tmp_path: Path) -> None:
    write_test_png(tmp_path / "query.png", size=(100, 60))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "query-1",
                "image": "query.png",
                "series_id": "series-1",
                "split": "test",
                "conditions": ["frontal"],
                "annotations": {
                    "label_bbox": [20, 10, 80, 50],
                    "cylinder": {
                        "cx": 50,
                        "radius": 42,
                        "theta_min": -0.6,
                        "theta_max": 0.6,
                    },
                },
            }
        )
        + "\n"
    )
    output = tmp_path / "benchmark"

    exit_code = main(
        [
            "benchmark",
            str(manifest),
            "--output-dir",
            str(output),
            "--warmup-runs",
            "1",
            "--measured-runs",
            "2",
        ]
    )

    assert exit_code == 0
    report = json.loads((output / "report.json").read_text())
    assert report["metrics"]["ocr"]["status"] == "not_run"
    assert report["metrics"]["retrieval"]["status"] == "not_run"
    assert report["timing"]["measurement_count"] == 2
    assert (output / "query-1" / "comparison.png").is_file()


def test_benchmark_rejects_series_leakage(tmp_path: Path, capsys) -> None:
    write_test_png(tmp_path / "a.png")
    write_test_png(tmp_path / "b.png")
    manifest = tmp_path / "leak.jsonl"
    rows = [
        {
            "sample_id": "a",
            "image": "a.png",
            "series_id": "same",
            "split": "tuning",
        },
        {
            "sample_id": "b",
            "image": "b.png",
            "series_id": "same",
            "split": "test",
        },
    ]
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    exit_code = main(
        ["benchmark", str(manifest), "--output-dir", str(tmp_path / "out")]
    )

    assert exit_code == 2
    assert "series" in capsys.readouterr().err.lower()
