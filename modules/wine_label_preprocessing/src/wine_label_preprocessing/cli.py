"""CLI for file/folder preprocessing and model-independent benchmarking."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from .benchmark import (
    create_comparison_variants,
    environment_metadata,
    run_benchmark,
)
from .dewarpnet import DewarpNetAdapter
from .errors import ManifestError, PreprocessingError
from .models import (
    BBox,
    BenchmarkConfig,
    BenchmarkSample,
    CylinderGeometry,
    DewarpNetConfig,
    ImageAnnotations,
    OCRTruth,
    PhotometricConfig,
    PreprocessingConfig,
    UnwrapConfig,
)
from .photometry import mild_ocr_profile
from .pipeline import preprocess
from .reporting import write_comparison, write_json

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _resolution_limit(value: str) -> int | None:
    if value.lower() == "none":
        return None
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("resolution limit must be positive or 'none'")
    return parsed


def _bbox(value: Any, *, field: str) -> BBox | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ManifestError(f"{field} must contain four integer coordinates")
    if any(isinstance(item, bool) or not isinstance(item, (int, np.integer)) for item in value):
        raise ManifestError(f"{field} coordinates must be integers")
    return BBox(*(int(item) for item in value))


def _load_mask(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as source:
            oriented = ImageOps.exif_transpose(source)
            return np.asarray(oriented.convert("L"), dtype=np.uint8) != 0
    except OSError as exc:
        raise ManifestError(f"Cannot decode bottle mask: {path}") from exc


def _annotations_from_mapping(
    value: dict[str, Any] | None,
    *,
    base_directory: Path,
) -> ImageAnnotations:
    data = value or {}
    cylinder_data = data.get("cylinder")
    cylinder = None
    if cylinder_data is not None:
        if not isinstance(cylinder_data, dict):
            raise ManifestError("cylinder must be an object")
        try:
            cylinder = CylinderGeometry(
                cx=float(cylinder_data["cx"]),
                radius=float(cylinder_data["radius"]),
                theta_min=(
                    float(cylinder_data["theta_min"])
                    if cylinder_data.get("theta_min") is not None
                    else None
                ),
                theta_max=(
                    float(cylinder_data["theta_max"])
                    if cylinder_data.get("theta_max") is not None
                    else None
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ManifestError("cylinder requires numeric cx and radius") from exc
    mask = None
    mask_value = data.get("bottle_mask")
    if mask_value is not None:
        mask_path = Path(mask_value)
        if not mask_path.is_absolute():
            mask_path = base_directory / mask_path
        mask = _load_mask(mask_path)
    return ImageAnnotations(
        label_bbox=_bbox(data.get("label_bbox"), field="label_bbox"),
        bottle_bbox=_bbox(data.get("bottle_bbox"), field="bottle_bbox"),
        bottle_mask=mask,
        cylinder=cylinder,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ManifestError(
                    f"Manifest line {line_number} must be a JSON object"
                )
            rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"Cannot read JSONL manifest {path}: {exc}") from exc
    return rows


def _dewarpnet_from_args(args: argparse.Namespace) -> DewarpNetAdapter | None:
    values = [
        args.dewarpnet_repo,
        args.dewarpnet_wc_checkpoint,
        args.dewarpnet_bm_checkpoint,
    ]
    if not any(values):
        return None
    if not all(values):
        raise ManifestError(
            "DewarpNet requires --dewarpnet-repo, --dewarpnet-wc-checkpoint, "
            "and --dewarpnet-bm-checkpoint together"
        )
    return DewarpNetAdapter(
        DewarpNetConfig(
            official_repo=Path(args.dewarpnet_repo),
            wc_checkpoint=Path(args.dewarpnet_wc_checkpoint),
            bm_checkpoint=Path(args.dewarpnet_bm_checkpoint),
            device=args.dewarpnet_device,
            align_corners=args.dewarpnet_align_corners,
        )
    )


def _preprocessing_config(args: argparse.Namespace) -> PreprocessingConfig:
    if args.ocr_mild:
        ocr_profile = mild_ocr_profile(unsharp=args.ocr_unsharp)
    elif args.ocr_unsharp:
        ocr_profile = PhotometricConfig(unsharp_enabled=True)
    else:
        ocr_profile = PhotometricConfig()
    return PreprocessingConfig(
        embedding_max_long_side=args.embedding_max_long_side,
        ocr_max_long_side=args.ocr_max_long_side,
        ocr_photometric=ocr_profile,
        unwrap=UnwrapConfig(
            enabled=not args.disable_cylindrical,
            max_stretch=args.max_stretch,
            max_long_side=args.unwrap_max_long_side,
        ),
    )


def _direct_annotations(args: argparse.Namespace) -> ImageAnnotations:
    angle_values = (args.theta_min, args.theta_max)
    geometry_values = (args.cx, args.radius)
    if any(value is not None for value in geometry_values + angle_values):
        if args.cx is None or args.radius is None:
            raise ManifestError("Explicit cylinder requires both --cx and --radius")
        if (args.theta_min is None) != (args.theta_max is None):
            raise ManifestError("Explicit angles require both --theta-min and --theta-max")
        cylinder = CylinderGeometry(
            args.cx, args.radius, args.theta_min, args.theta_max
        )
    else:
        cylinder = None
    mask = _load_mask(Path(args.bottle_mask)) if args.bottle_mask else None
    return ImageAnnotations(
        label_bbox=_bbox(args.label_bbox, field="label_bbox"),
        bottle_bbox=_bbox(args.bottle_bbox, field="bottle_bbox"),
        bottle_mask=mask,
        cylinder=cylinder,
    )


def _has_direct_annotations(args: argparse.Namespace) -> bool:
    return any(
        value is not None
        for value in (
            args.label_bbox,
            args.bottle_bbox,
            args.bottle_mask,
            args.cx,
            args.radius,
            args.theta_min,
            args.theta_max,
        )
    )


def _annotation_manifest(path: Path) -> dict[str, ImageAnnotations]:
    annotations: dict[str, ImageAnnotations] = {}
    for row in _read_jsonl(path):
        image_value = row.get("image")
        if not isinstance(image_value, str) or not image_value:
            raise ManifestError("Annotation manifest rows require a nonempty image")
        key = Path(image_value).as_posix()
        if key in annotations:
            raise ManifestError(f"Duplicate annotation manifest image: {key}")
        annotation_data = row.get("annotations", row)
        annotations[key] = _annotations_from_mapping(
            annotation_data, base_directory=path.parent
        )
    return annotations


def _discover_images(input_path: Path) -> list[tuple[Path, Path, str]]:
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ManifestError(f"Unsupported input image extension: {input_path.suffix}")
        return [(input_path, Path(input_path.stem), input_path.name)]
    if not input_path.is_dir():
        raise ManifestError(f"Input path does not exist: {input_path}")
    discovered = sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not discovered:
        raise ManifestError(f"No supported images found in: {input_path}")
    return [
        (path, path.relative_to(input_path).with_suffix(""), path.relative_to(input_path).as_posix())
        for path in discovered
    ]


def _run_process(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    output_root = Path(args.output_dir)
    items = _discover_images(input_path)
    if input_path.is_dir() and _has_direct_annotations(args):
        raise ManifestError(
            "Folder processing requires --annotations for per-image geometry"
        )
    direct = _direct_annotations(args)
    manifest = (
        _annotation_manifest(Path(args.annotations)) if args.annotations else {}
    )
    config = _preprocessing_config(args)
    dewarpnet = _dewarpnet_from_args(args)
    report_items: list[dict[str, Any]] = []
    failed = False
    used_stems: set[str] = set()
    for path, relative_stem, input_name in items:
        if relative_stem.as_posix() in used_stems:
            raise ManifestError(
                f"Output collision for input stem: {relative_stem.as_posix()}"
            )
        used_stems.add(relative_stem.as_posix())
        annotations = manifest.get(input_name, direct if input_path.is_file() else ImageAnnotations())
        try:
            result = preprocess(
                path,
                annotations=annotations,
                config=config,
                dewarpnet=dewarpnet,
            )
            variants = create_comparison_variants(
                result, mild_ocr_profile(unsharp=args.ocr_unsharp)
            )
            entry = write_comparison(output_root, relative_stem, result, variants)
            entry["input"] = input_name
            report_items.append(entry)
        except (OSError, ValueError, PreprocessingError) as exc:
            failed = True
            print(f"processing failed for {input_name}: {exc}", file=sys.stderr)
            report_items.append(
                {"input": input_name, "status": "failed", "reason": str(exc)}
            )
    report = {
        "schema_version": "1.0",
        "command": "process",
        "input": str(input_path),
        "config": asdict(config),
        "environment": environment_metadata(),
        "images": report_items,
    }
    write_json(output_root / "report.json", report)
    return 1 if failed else 0


def _benchmark_samples(path: Path) -> list[BenchmarkSample]:
    samples: list[BenchmarkSample] = []
    for row in _read_jsonl(path):
        required = ("sample_id", "image", "series_id", "split")
        missing = [field for field in required if not row.get(field)]
        if missing:
            raise ManifestError(
                "Benchmark manifest row is missing: " + ", ".join(missing)
            )
        sample_id = str(row["sample_id"])
        sample_path = Path(sample_id)
        if sample_path.name != sample_id or sample_id in {".", ".."}:
            raise ManifestError("Benchmark sample_id must be a safe filename")
        image_path = Path(str(row["image"]))
        if not image_path.is_absolute():
            image_path = path.parent / image_path
        truth_data = row.get("truth")
        truth = None
        if truth_data is not None:
            if not isinstance(truth_data, dict):
                raise ManifestError("Benchmark truth must be an object")
            truth = OCRTruth(
                full_text=str(truth_data.get("full_text", "")),
                name=str(truth_data.get("name", "")),
                producer=str(truth_data.get("producer", "")),
                year=str(truth_data.get("year", "")),
            )
        samples.append(
            BenchmarkSample(
                sample_id=sample_id,
                image=image_path,
                series_id=str(row["series_id"]),
                split=str(row["split"]),
                annotations=_annotations_from_mapping(
                    row.get("annotations"), base_directory=path.parent
                ),
                conditions=tuple(str(value) for value in row.get("conditions", [])),
                truth=truth,
                catalog_id=(
                    str(row["catalog_id"])
                    if row.get("catalog_id") is not None
                    else None
                ),
                input_color_order=None,
            )
        )
    return samples


def _run_benchmark(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    output_root = Path(args.output_dir)
    samples = _benchmark_samples(manifest_path)
    config = _preprocessing_config(args)
    benchmark_config = BenchmarkConfig(
        preprocessing=config,
        mild_photometric=mild_ocr_profile(unsharp=args.ocr_unsharp),
        warmup_runs=args.warmup_runs,
        measured_runs=args.measured_runs,
    )
    dewarpnet = _dewarpnet_from_args(args)
    report = run_benchmark(samples, benchmark_config, dewarpnet=dewarpnet)
    assets_by_id: dict[str, Any] = {}
    for sample in samples:
        result = preprocess(
            sample.image,
            annotations=sample.annotations,
            config=config,
            input_color_order=sample.input_color_order,
            dewarpnet=dewarpnet,
        )
        variants = create_comparison_variants(
            result, benchmark_config.mild_photometric
        )
        assets_by_id[sample.sample_id] = write_comparison(
            output_root, Path(sample.sample_id), result, variants
        )
    for sample_report in report["samples"]:
        sample_report["assets"] = assets_by_id[sample_report["sample_id"]]["assets"]
    report["command"] = "benchmark"
    report["manifest"] = str(manifest_path)
    write_json(output_root / "report.json", report)
    return 0


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--embedding-max-long-side", type=_resolution_limit, default=2400
    )
    parser.add_argument("--ocr-max-long-side", type=_resolution_limit, default=2400)
    parser.add_argument("--unwrap-max-long-side", type=_resolution_limit, default=2400)
    parser.add_argument("--max-stretch", type=float, default=2.5)
    parser.add_argument("--disable-cylindrical", action="store_true")
    parser.add_argument("--ocr-mild", action="store_true")
    parser.add_argument("--ocr-unsharp", action="store_true")
    parser.add_argument("--dewarpnet-repo")
    parser.add_argument("--dewarpnet-wc-checkpoint")
    parser.add_argument("--dewarpnet-bm-checkpoint")
    parser.add_argument(
        "--dewarpnet-device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--dewarpnet-align-corners", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wine-label-preprocess")
    subparsers = parser.add_subparsers(dest="command", required=True)

    process_parser = subparsers.add_parser(
        "process", help="process one image or every supported image in a folder"
    )
    process_parser.add_argument("input")
    process_parser.add_argument("--annotations")
    process_parser.add_argument("--label-bbox", nargs=4, type=int)
    process_parser.add_argument("--bottle-bbox", nargs=4, type=int)
    process_parser.add_argument("--bottle-mask")
    process_parser.add_argument("--cx", type=float)
    process_parser.add_argument("--radius", type=float)
    process_parser.add_argument("--theta-min", type=float)
    process_parser.add_argument("--theta-max", type=float)
    _add_common_options(process_parser)

    benchmark_parser = subparsers.add_parser(
        "benchmark", help="run A-E preprocessing comparisons from a JSONL manifest"
    )
    benchmark_parser.add_argument("manifest")
    benchmark_parser.add_argument("--warmup-runs", type=int, default=1)
    benchmark_parser.add_argument("--measured-runs", type=int, default=5)
    _add_common_options(benchmark_parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "process":
            return _run_process(args)
        if args.command == "benchmark":
            return _run_benchmark(args)
    except (ManifestError, PreprocessingError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2

