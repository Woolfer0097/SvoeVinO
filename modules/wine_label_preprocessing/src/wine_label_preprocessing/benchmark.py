"""A-E comparison runner with injected OCR/embedding backends."""

from __future__ import annotations

import hashlib
import os
import platform
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from time import perf_counter_ns
from typing import Any, Protocol

import cv2
import numpy as np

from .errors import ManifestError
from .image_io import load_rgb
from .metrics import (
    character_error_rate,
    exact_field_matches,
    normalize_text,
    paired_degradation_rate,
    recall_at_k,
    retrieval_rank,
)
from .models import (
    BenchmarkConfig,
    BenchmarkSample,
    ComparisonVariant,
    OCRPrediction,
    PreprocessingResult,
)
from .photometry import apply_photometric
from .pipeline import preprocess


class OCRBackend(Protocol):
    name: str

    def recognize(self, image: np.ndarray) -> OCRPrediction:
        """Recognize all benchmark text fields from an RGB image."""


class EmbeddingBackend(Protocol):
    name: str

    def embed(self, image: np.ndarray) -> np.ndarray:
        """Return one query embedding from an RGB image."""


def validate_samples(samples: Sequence[BenchmarkSample]) -> None:
    """Reject duplicate IDs and photo-series leakage across splits."""

    seen_ids: set[str] = set()
    series_splits: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        if not sample.sample_id:
            raise ManifestError("Benchmark sample_id must not be empty")
        if sample.sample_id in seen_ids:
            raise ManifestError(f"Duplicate benchmark sample_id: {sample.sample_id}")
        seen_ids.add(sample.sample_id)
        if sample.split not in {"tuning", "test"}:
            raise ManifestError(
                f"Benchmark split must be tuning or test: {sample.sample_id}"
            )
        if not sample.series_id:
            raise ManifestError(
                f"Benchmark series_id must not be empty: {sample.sample_id}"
            )
        series_splits[sample.series_id].add(sample.split)
    leaked = sorted(series for series, splits in series_splits.items() if len(splits) > 1)
    if leaked:
        raise ManifestError(
            "Photo series must not cross tuning and test splits: " + ", ".join(leaked)
        )


def create_comparison_variants(
    result: PreprocessingResult,
    mild_config: Any | None = None,
) -> dict[str, ComparisonVariant]:
    """Create honest A-E variants without substituting missing geometry."""

    if mild_config is None:
        from .photometry import mild_ocr_profile

        mild_config = mild_ocr_profile()
    corrected_a, operations_a = apply_photometric(result.original_crop, mild_config)
    variants: dict[str, ComparisonVariant] = {
        "A": ComparisonVariant(
            "A", result.original_crop, "ok", metadata={"name": "original_crop"}
        ),
        "B": ComparisonVariant(
            "B",
            corrected_a,
            "ok",
            metadata={
                "name": "original_crop_mild_photometric",
                "photometric_operations": operations_a,
            },
        ),
    }

    cylinder_metadata = result.metadata.get("cylindrical", {})
    if result.cylindrical_image is None:
        cylinder_status = cylinder_metadata.get("status", "skipped")
        if cylinder_status not in {"skipped", "failed", "unavailable"}:
            cylinder_status = "skipped"
        reason = cylinder_metadata.get("reason", "cylindrical_variant_missing")
        variants["C"] = ComparisonVariant("C", None, cylinder_status, reason)
        variants["D"] = ComparisonVariant("D", None, cylinder_status, reason)
    else:
        corrected_c, operations_c = apply_photometric(
            result.cylindrical_image, mild_config
        )
        variants["C"] = ComparisonVariant(
            "C",
            result.cylindrical_image,
            "ok",
            metadata={"name": "cylindrical"},
        )
        variants["D"] = ComparisonVariant(
            "D",
            corrected_c,
            "ok",
            metadata={
                "name": "cylindrical_mild_photometric",
                "photometric_operations": operations_c,
            },
        )

    dewarp_metadata = result.metadata.get("dewarpnet", {})
    if result.dewarpnet_image is None:
        raw_status = dewarp_metadata.get("status", "not_configured")
        status = "not_run" if raw_status == "not_configured" else raw_status
        if status not in {"not_run", "unavailable", "failed", "skipped"}:
            status = "not_run"
        variants["E"] = ComparisonVariant(
            "E",
            None,
            status,
            dewarp_metadata.get("reason", "dewarpnet_variant_missing"),
        )
    else:
        variants["E"] = ComparisonVariant(
            "E", result.dewarpnet_image, "ok", metadata={"name": "dewarpnet"}
        )
    return variants


def _gallery_metadata(gallery: Mapping[str, np.ndarray] | None) -> dict[str, Any]:
    if gallery is None:
        return {"status": "not_supplied"}
    digest = hashlib.sha256()
    dimensions: set[int] = set()
    for catalog_id in sorted(gallery):
        vector = np.asarray(gallery[catalog_id], dtype=np.float32).reshape(-1)
        dimensions.add(vector.size)
        digest.update(catalog_id.encode("utf-8"))
        digest.update(vector.shape[0].to_bytes(8, "little"))
        digest.update(vector.tobytes(order="C"))
    if len(dimensions) > 1:
        raise ValueError("Fixed gallery embeddings must share one dimension")
    return {
        "status": "fixed",
        "item_count": len(gallery),
        "dimension": next(iter(dimensions), 0),
        "sha256": digest.hexdigest(),
        "catalog_ids": sorted(gallery),
    }


def environment_metadata() -> dict[str, Any]:
    """Return enough hardware/runtime context to interpret latency numbers."""

    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unreported",
        "logical_cpu_count": os.cpu_count(),
        "executable": sys.executable,
    }


def _variant_summary(
    samples: Sequence[dict[str, Any]], code: str
) -> dict[str, Any]:
    statuses = [sample["variants"][code]["status"] for sample in samples]
    available = statuses.count("ok")
    if available == len(statuses):
        status = "ok"
    elif available:
        status = "partial"
    elif len(set(statuses)) == 1:
        status = statuses[0]
    else:
        status = "not_run"
    return {
        "status": status,
        "sample_count": len(statuses),
        "available_count": available,
        "status_counts": {
            key: statuses.count(key) for key in sorted(set(statuses))
        },
    }


def _aggregate_ocr(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, Any] = {}
    for code in "ABCDE":
        selected = [record for record in records if record["variant"] == code]
        if not selected:
            continue
        by_variant[code] = {
            "count": len(selected),
            "cer": float(np.mean([record["cer"] for record in selected])),
            **{
                f"exact_{field}": float(
                    np.mean([record["exact"][field] for record in selected])
                )
                for field in ("name", "producer", "year", "all")
            },
        }
    return by_variant


def _aggregate_retrieval(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, Any] = {}
    for code in "ABCDE":
        selected = [record for record in records if record["variant"] == code]
        if not selected:
            continue
        by_variant[code] = {
            "count": len(selected),
            "mean_rank": float(np.mean([record["rank"] for record in selected])),
            "recall_at_1": float(np.mean([record["recall"][1] for record in selected])),
            "recall_at_5": float(np.mean([record["recall"][5] for record in selected])),
        }
    return by_variant


def _paired_degradation(
    ocr_records: list[dict[str, Any]],
    retrieval_records: list[dict[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for code in "BCDE":
        by_sample_ocr = {
            (record["sample_id"], record["variant"]): record for record in ocr_records
        }
        ocr_pairs = [
            (by_sample_ocr[(sample_id, "A")], by_sample_ocr[(sample_id, code)])
            for sample_id in sorted({record["sample_id"] for record in ocr_records})
            if (sample_id, "A") in by_sample_ocr and (sample_id, code) in by_sample_ocr
        ]
        candidate_output: dict[str, Any] = {}
        if ocr_pairs:
            candidate_output["cer"] = paired_degradation_rate(
                [pair[0]["cer"] for pair in ocr_pairs],
                [pair[1]["cer"] for pair in ocr_pairs],
                lower_is_better=True,
            )
            for field in ("name", "producer", "year", "all"):
                candidate_output[f"exact_{field}"] = paired_degradation_rate(
                    [pair[0]["exact"][field] for pair in ocr_pairs],
                    [pair[1]["exact"][field] for pair in ocr_pairs],
                    lower_is_better=False,
                )
        by_sample_retrieval = {
            (record["sample_id"], record["variant"]): record
            for record in retrieval_records
        }
        retrieval_pairs = [
            (
                by_sample_retrieval[(sample_id, "A")],
                by_sample_retrieval[(sample_id, code)],
            )
            for sample_id in sorted(
                {record["sample_id"] for record in retrieval_records}
            )
            if (sample_id, "A") in by_sample_retrieval
            and (sample_id, code) in by_sample_retrieval
        ]
        if retrieval_pairs:
            for cutoff in (1, 5):
                candidate_output[f"recall_at_{cutoff}"] = paired_degradation_rate(
                    [pair[0]["recall"][cutoff] for pair in retrieval_pairs],
                    [pair[1]["recall"][cutoff] for pair in retrieval_pairs],
                    lower_is_better=False,
                )
        if candidate_output:
            output[code] = candidate_output
    return output


def run_benchmark(
    samples: Sequence[BenchmarkSample],
    config: BenchmarkConfig | None = None,
    *,
    ocr_backend: OCRBackend | None = None,
    embedding_backend: EmbeddingBackend | None = None,
    gallery: Mapping[str, np.ndarray] | None = None,
    dewarpnet: Any | None = None,
) -> dict[str, Any]:
    """Run warm in-memory A-E comparisons with shared evaluator instances."""

    validate_samples(samples)
    effective_config = config or BenchmarkConfig()
    if effective_config.warmup_runs < 0:
        raise ValueError("warmup_runs must not be negative")
    if effective_config.measured_runs < 1:
        raise ValueError("measured_runs must be positive")
    gallery_info = _gallery_metadata(gallery)

    sample_reports: list[dict[str, Any]] = []
    all_timings: list[float] = []
    ocr_records: list[dict[str, Any]] = []
    retrieval_records: list[dict[str, Any]] = []

    for sample in samples:
        decode_start = perf_counter_ns()
        loaded = load_rgb(
            sample.image,
            input_color_order=sample.input_color_order,
        )
        decode_ms = (perf_counter_ns() - decode_start) / 1_000_000.0

        def process_once() -> tuple[PreprocessingResult, dict[str, ComparisonVariant]]:
            result = preprocess(
                loaded.array,
                annotations=sample.annotations,
                config=effective_config.preprocessing,
                input_color_order="RGB",
                dewarpnet=dewarpnet,
            )
            return result, create_comparison_variants(
                result, effective_config.mild_photometric
            )

        for _ in range(effective_config.warmup_runs):
            process_once()

        measured: list[float] = []
        result: PreprocessingResult | None = None
        variants: dict[str, ComparisonVariant] | None = None
        for _ in range(effective_config.measured_runs):
            started = perf_counter_ns()
            result, variants = process_once()
            measured.append((perf_counter_ns() - started) / 1_000_000.0)
        assert result is not None and variants is not None
        all_timings.extend(measured)

        variant_report: dict[str, Any] = {}
        for code, variant in variants.items():
            item: dict[str, Any] = {
                "status": variant.status,
                "reason": variant.reason,
                "shape": list(variant.image.shape) if variant.image is not None else None,
                "metadata": variant.metadata,
            }
            if variant.image is not None and ocr_backend is not None:
                prediction = ocr_backend.recognize(variant.image)
                item["ocr_prediction"] = {
                    **asdict(prediction),
                    "normalized": {
                        field: normalize_text(getattr(prediction, field))
                        for field in ("full_text", "name", "producer", "year")
                    },
                }
                if sample.truth is not None:
                    cer = character_error_rate(
                        sample.truth.full_text, prediction.full_text
                    )
                    exact = exact_field_matches(sample.truth, prediction)
                    item["ocr_metrics"] = {"cer": cer, "exact": exact}
                    ocr_records.append(
                        {
                            "sample_id": sample.sample_id,
                            "variant": code,
                            "conditions": sample.conditions,
                            "cer": cer,
                            "exact": exact,
                        }
                    )
            if (
                variant.image is not None
                and embedding_backend is not None
                and gallery is not None
                and sample.catalog_id is not None
            ):
                embedding = np.asarray(embedding_backend.embed(variant.image))
                rank = retrieval_rank(embedding, sample.catalog_id, gallery)
                recall = recall_at_k(rank, (1, 5))
                item["retrieval_metrics"] = {
                    "rank": rank,
                    "recall_at_1": recall[1],
                    "recall_at_5": recall[5],
                }
                retrieval_records.append(
                    {
                        "sample_id": sample.sample_id,
                        "variant": code,
                        "conditions": sample.conditions,
                        "rank": rank,
                        "recall": recall,
                    }
                )
            variant_report[code] = item

        sample_reports.append(
            {
                "sample_id": sample.sample_id,
                "series_id": sample.series_id,
                "split": sample.split,
                "conditions": list(sample.conditions),
                "catalog_id": sample.catalog_id,
                "decoded_shape": list(loaded.array.shape),
                "decode_ms": decode_ms,
                "preprocessing_timing_ms": {
                    "p50": float(np.percentile(measured, 50)),
                    "p95": float(np.percentile(measured, 95)),
                    "values": measured,
                },
                "pipeline_metadata": result.metadata,
                "variants": variant_report,
            }
        )

    if ocr_backend is None:
        ocr_metrics: dict[str, Any] = {
            "status": "not_run",
            "reason": "ocr_backend_not_supplied",
        }
    elif not ocr_records:
        ocr_metrics = {"status": "not_run", "reason": "ocr_truth_not_supplied"}
    else:
        ocr_metrics = {"status": "measured", "by_variant": _aggregate_ocr(ocr_records)}

    if embedding_backend is None:
        retrieval_metrics: dict[str, Any] = {
            "status": "not_run",
            "reason": "embedding_backend_not_supplied",
        }
    elif gallery is None:
        retrieval_metrics = {"status": "not_run", "reason": "gallery_not_supplied"}
    elif not retrieval_records:
        retrieval_metrics = {
            "status": "not_run",
            "reason": "catalog_truth_not_supplied",
        }
    else:
        retrieval_metrics = {
            "status": "measured",
            "by_variant": _aggregate_retrieval(retrieval_records),
        }

    conditions = sorted(
        {condition for sample in samples for condition in sample.conditions}
    )
    by_condition: dict[str, Any] = {}
    for condition in conditions:
        condition_ocr = [
            record for record in ocr_records if condition in record["conditions"]
        ]
        condition_retrieval = [
            record for record in retrieval_records if condition in record["conditions"]
        ]
        by_condition[condition] = {
            "sample_count": sum(condition in sample.conditions for sample in samples),
            "ocr_by_variant": _aggregate_ocr(condition_ocr),
            "retrieval_by_variant": _aggregate_retrieval(condition_retrieval),
        }

    timing = {
        "warmup_runs": effective_config.warmup_runs,
        "measured_runs": effective_config.measured_runs,
        "measurement_count": len(all_timings),
        "p50_ms": float(np.percentile(all_timings, 50)) if all_timings else None,
        "p95_ms": float(np.percentile(all_timings, 95)) if all_timings else None,
        "included": ["in_memory_preprocess", "A-E_variant_photometry"],
        "excluded": ["decode", "model_construction", "visualization", "file_writes"],
        "model_loading_excluded_by_warmup": effective_config.warmup_runs > 0,
    }
    return {
        "schema_version": "1.0",
        "sample_count": len(samples),
        "config": {
            "warmup_runs": effective_config.warmup_runs,
            "measured_runs": effective_config.measured_runs,
        },
        "ocr_backend": (
            {"name": ocr_backend.name, "instance_id": id(ocr_backend)}
            if ocr_backend is not None
            else None
        ),
        "embedding_backend": (
            {"name": embedding_backend.name, "instance_id": id(embedding_backend)}
            if embedding_backend is not None
            else None
        ),
        "gallery": gallery_info,
        "variants": {
            code: _variant_summary(sample_reports, code) for code in "ABCDE"
        },
        "samples": sample_reports,
        "metrics": {
            "ocr": ocr_metrics,
            "retrieval": retrieval_metrics,
            "degradation_relative_to_A": _paired_degradation(
                ocr_records, retrieval_records
            ),
            "by_condition": by_condition,
        },
        "timing": timing,
        "environment": environment_metadata(),
    }

