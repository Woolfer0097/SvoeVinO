from __future__ import annotations

import numpy as np
import pytest

from wine_label_preprocessing.benchmark import (
    create_comparison_variants,
    run_benchmark,
    validate_samples,
)
from wine_label_preprocessing.errors import ManifestError
from wine_label_preprocessing.models import (
    BBox,
    BenchmarkConfig,
    BenchmarkSample,
    CylinderGeometry,
    ImageAnnotations,
    OCRPrediction,
    OCRTruth,
)
from wine_label_preprocessing.pipeline import preprocess


class CountingOCRBackend:
    name = "counting-ocr"

    def __init__(self) -> None:
        self.calls = 0

    def recognize(self, image: np.ndarray) -> OCRPrediction:
        self.calls += 1
        return OCRPrediction(
            full_text="Example Wine 2020",
            name="Example Wine",
            producer="Example Winery",
            year="2020",
        )


class CountingEmbeddingBackend:
    name = "counting-embedding"

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, image: np.ndarray) -> np.ndarray:
        self.calls += 1
        return np.array([1.0, 0.0], dtype=np.float32)


def benchmark_sample(
    *,
    split: str = "test",
    series_id: str = "series-1",
    sample_id: str = "query-1",
) -> BenchmarkSample:
    image = np.full((60, 100, 3), 100, dtype=np.uint8)
    return BenchmarkSample(
        sample_id=sample_id,
        image=image,
        series_id=series_id,
        split=split,
        annotations=ImageAnnotations(
            label_bbox=BBox(20, 10, 80, 50),
            cylinder=CylinderGeometry(50.0, 42.0, -0.6, 0.6),
        ),
        conditions=("frontal",),
        truth=OCRTruth(
            full_text="Example Wine 2020",
            name="Example Wine",
            producer="Example Winery",
            year="2020",
        ),
        catalog_id="wine-1",
    )


def test_series_cannot_cross_tuning_and_test() -> None:
    samples = [
        benchmark_sample(split="tuning", series_id="shared", sample_id="tuning-1"),
        benchmark_sample(split="test", series_id="shared", sample_id="test-1"),
    ]

    with pytest.raises(ManifestError, match="series"):
        validate_samples(samples)


def test_comparison_variants_are_a_through_e_and_never_fake_missing() -> None:
    result = preprocess(
        np.full((20, 30, 3), 80, dtype=np.uint8), input_color_order="RGB"
    )

    variants = create_comparison_variants(result)

    assert list(variants) == ["A", "B", "C", "D", "E"]
    assert variants["A"].status == "ok"
    assert variants["B"].status == "ok"
    assert variants["C"].status == "skipped"
    assert variants["D"].status == "skipped"
    assert variants["E"].status == "not_run"
    assert variants["E"].image is None


def test_one_backend_instance_is_reused_for_available_variants() -> None:
    ocr = CountingOCRBackend()
    embedder = CountingEmbeddingBackend()
    config = BenchmarkConfig(warmup_runs=1, measured_runs=3)

    report = run_benchmark(
        [benchmark_sample()],
        config,
        ocr_backend=ocr,
        embedding_backend=embedder,
        gallery={"wine-1": np.array([1.0, 0.0], dtype=np.float32)},
    )

    assert set(report["variants"]) == {"A", "B", "C", "D", "E"}
    assert report["variants"]["E"]["status"] == "not_run"
    assert ocr.calls == 4
    assert embedder.calls == 4
    assert report["ocr_backend"]["instance_id"] == id(ocr)
    assert report["embedding_backend"]["instance_id"] == id(embedder)
    assert report["metrics"]["ocr"]["status"] == "measured"
    assert report["metrics"]["retrieval"]["status"] == "measured"
    assert report["metrics"]["retrieval"]["by_variant"]["A"]["recall_at_1"] == 1.0
    assert report["metrics"]["by_condition"]["frontal"]["sample_count"] == 1
    assert report["timing"]["warmup_runs"] == 1
    assert report["timing"]["measured_runs"] == 3
    assert report["timing"]["p95_ms"] >= report["timing"]["p50_ms"]


def test_missing_backends_are_explicitly_not_run() -> None:
    report = run_benchmark(
        [benchmark_sample()], BenchmarkConfig(warmup_runs=0, measured_runs=1)
    )

    assert report["metrics"]["ocr"] == {
        "status": "not_run",
        "reason": "ocr_backend_not_supplied",
    }
    assert report["metrics"]["retrieval"] == {
        "status": "not_run",
        "reason": "embedding_backend_not_supplied",
    }
    assert report["gallery"]["status"] == "not_supplied"
