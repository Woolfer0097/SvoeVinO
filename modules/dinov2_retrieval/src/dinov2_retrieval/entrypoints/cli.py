"""CLI entrypoint; every command prints exactly one JSON document to stdout.

The JSON is indented on a terminal and printed on one line otherwise, so the
last stdout line is the result that Apache Airflow pushes to XCom. Logs go to
stderr. Exit codes: 0 success, 1 invalid input, file or configuration,
2 invalid CLI arguments, 3 model error, 4 database error, 5 indexing or
evaluation finished but some photos failed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from typing import Any, NoReturn

from pydantic import ValidationError

from ..application.check_health import DATABASE_CHECKS, check_health
from ..application.create_embedding import create_embedding
from ..application.evaluate_retrieval import evaluate_retrieval
from ..application.index_reference_images import index_reference_images
from ..application.search_similar_wines import search_similar_wines
from ..application.validate_image import validate_image
from ..config import ConfigurationError, get_default_top_k, resolve_data_path
from ..contracts import HealthReport, ReferenceImage, RetrievalRequest, SearchRequest
from ..embedding.base import Embedder, EmbeddingError
from ..infrastructure.csv_manifest import ManifestError
from ..infrastructure.evaluation_manifest import read_evaluation_queries
from ..infrastructure.reference_manifest import load_references
from ..infrastructure.storage.local_storage import ImageValidationError
from ..preprocessing.image_preprocessor import ImagePreprocessingError
from ..retrieval.reference_repository import ReferenceRepository, RepositoryError

EXIT_OK = 0
EXIT_INPUT_ERROR = 1
EXIT_USAGE_ERROR = 2
EXIT_MODEL_ERROR = 3
EXIT_DATABASE_ERROR = 4
EXIT_PARTIAL_FAILURE = 5

RUN_STATUS_EXIT_CODES = {
    "ok": EXIT_OK,
    "partial": EXIT_PARTIAL_FAILURE,
    "failed": EXIT_INPUT_ERROR,
}
FAILURE_LABELS = {
    "validate": "validation failed",
    "embed": "embedding failed",
    "index": "indexing failed",
    "search": "search failed",
    "evaluate": "evaluation failed",
    "health": "health check failed",
}


class UsageError(Exception):
    """Invalid command-line arguments."""


class RequestFileError(ValueError):
    """Raised when a --request-json file cannot be read or is invalid."""


class _JsonArgumentParser(argparse.ArgumentParser):
    """Raise instead of exiting, so usage errors are reported as JSON too."""

    def error(self, message: str) -> NoReturn:
        raise UsageError(message)


def build_parser() -> argparse.ArgumentParser:
    parser = _JsonArgumentParser(prog="dinov2-retrieval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate", help="validate a local image without creating an embedding"
    )
    validate_parser.add_argument("--image-uri", required=True)
    validate_parser.add_argument("--job-id", default="local-validation")
    validate_parser.add_argument("--query-id", default="local-validation")
    validate_parser.add_argument("--top-k", type=_positive_int, default=20)

    embed_parser = subparsers.add_parser(
        "embed", help="create a DINOv2 embedding for a local image"
    )
    embed_parser.add_argument("--image-uri", required=True)

    index_parser = subparsers.add_parser(
        "index",
        help="embed reference photos and save them to PostgreSQL; without "
        "--manifest/--reference-dir uses DATA_ROOT/reference",
    )
    source = index_parser.add_mutually_exclusive_group()
    source.add_argument("--manifest", help="CSV with wine_id,slug,image_uri columns")
    source.add_argument(
        "--reference-dir",
        help="folder with photos: <dir>/<photo> is one wine named after the file, "
        "<dir>/<wine>/<photo> is one wine named after the subfolder",
    )
    index_parser.add_argument(
        "--prune",
        action="store_true",
        help="also delete indexed photos of this model that are not in the source",
    )

    search_parser = subparsers.add_parser(
        "search", help="find the Top-K wines most similar to a photo"
    )
    query = search_parser.add_mutually_exclusive_group(required=True)
    query.add_argument("--image-uri")
    query.add_argument(
        "--request-json",
        help="JSON file with request_id, image_uri and optional top_k",
    )
    search_parser.add_argument(
        "--top-k", type=_positive_int, help="default: DEFAULT_TOP_K"
    )
    search_parser.add_argument("--request-id", help="default: a generated UUID")

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="measure Recall@K on photos with a known wine_id"
    )
    evaluate_parser.add_argument(
        "--queries",
        required=True,
        help="CSV with query_image_uri,wine_id columns",
    )
    evaluate_parser.add_argument(
        "--top-k",
        type=_positive_int,
        help="K of the main Recall@K; Recall@1, @5 and @20 are always reported. "
        "Default: DEFAULT_TOP_K",
    )

    subparsers.add_parser(
        "health", help="check configuration, PostgreSQL, schema and model"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.WARNING,
        stream=sys.stderr,
        format="%(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("dinov2_retrieval").setLevel(logging.INFO)

    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except UsageError as exc:
        parser.print_usage(sys.stderr)
        return _report_failure(None, exc, "usage", EXIT_USAGE_ERROR)

    try:
        return COMMANDS[args.command](args)
    except Exception as exc:  # every failure must end with JSON and an exit code
        category, exit_code = _classify(exc)
        return _report_failure(args.command, exc, category, exit_code)


def _run_validate(args: argparse.Namespace) -> int:
    request = RetrievalRequest(
        job_id=args.job_id,
        query_id=args.query_id,
        image_uri=args.image_uri,
        top_k=args.top_k,
    )
    validated_image = validate_image(request)
    _print_json(validated_image.model_dump(mode="json"))
    return EXIT_OK


def _run_embed(args: argparse.Namespace) -> int:
    result = create_embedding(args.image_uri)
    _print_json(
        {
            "path": result.path,
            "model": result.model,
            "dimension": result.dimension,
            "device": result.device,
            "embedding_preview": result.embedding[:5],
        }
    )
    return EXIT_OK


def _run_index(args: argparse.Namespace) -> int:
    references, source = _load_references(args)
    logging.getLogger(__name__).info(
        "indexing %d reference photos from %s", len(references), source
    )
    with closing(_connect_repository()) as repository:
        stats = index_reference_images(
            references, repository, prune=args.prune, embedder=_create_embedder()
        )
    stats = stats.model_copy(update={"source": source})
    _print_json(stats.model_dump(mode="json"))
    return RUN_STATUS_EXIT_CODES[stats.status]


def _run_search(args: argparse.Namespace) -> int:
    request = _build_search_request(args)
    with closing(_connect_repository()) as repository:
        response = search_similar_wines(
            request, repository, embedder=_create_embedder()
        )
    _print_json(response.model_dump(mode="json", exclude_none=True))
    return EXIT_OK


def _run_evaluate(args: argparse.Namespace) -> int:
    queries = read_evaluation_queries(resolve_data_path(args.queries))
    with closing(_connect_repository()) as repository:
        report = evaluate_retrieval(
            queries,
            repository,
            top_k=args.top_k or get_default_top_k(),
            embedder=_create_embedder(),
        )
    _print_json(report.model_dump(mode="json"))
    return RUN_STATUS_EXIT_CODES[report.status]


def _run_health(args: argparse.Namespace) -> int:
    report = check_health()
    _print_json(report.model_dump(mode="json", exclude_none=True))
    return _health_exit_code(report)


COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "validate": _run_validate,
    "embed": _run_embed,
    "index": _run_index,
    "search": _run_search,
    "evaluate": _run_evaluate,
    "health": _run_health,
}


def _load_references(
    args: argparse.Namespace,
) -> tuple[list[ReferenceImage], str]:
    return load_references(args.manifest, args.reference_dir)


def _build_search_request(args: argparse.Namespace) -> SearchRequest:
    if args.request_json is None:
        return SearchRequest(
            request_id=args.request_id or uuid.uuid4().hex,
            image_uri=args.image_uri,
            top_k=args.top_k or get_default_top_k(),
        )
    if args.top_k is not None or args.request_id is not None:
        raise UsageError(
            "--top-k and --request-id cannot be combined with --request-json"
        )
    return _read_request_file(resolve_data_path(args.request_json))


def _read_request_file(path: Path) -> SearchRequest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RequestFileError(f"Request file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RequestFileError(f"Request file {path} is not valid JSON: {exc}") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise RequestFileError(f"Cannot read request file {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise RequestFileError(f"Request file {path} must contain a JSON object")
    payload.setdefault("top_k", get_default_top_k())
    try:
        return SearchRequest.model_validate(payload)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'request'}: {error['msg']}"
            for error in exc.errors()
        )
        raise RequestFileError(f"Invalid search request in {path}: {problems}") from exc


def _connect_repository() -> ReferenceRepository:
    try:
        from ..infrastructure.database.postgres_reference_repository import (
            PostgresReferenceRepository,
        )
    except ImportError as exc:
        raise RepositoryError(
            f"PostgreSQL support is not installed (the db extra): {exc}"
        ) from exc
    return PostgresReferenceRepository.connect()


def _create_embedder() -> Embedder:
    from ..embedding.dinov2_embedder import DinoV2Embedder

    return DinoV2Embedder()


def _health_exit_code(report: HealthReport) -> int:
    failed = {name for name, check in report.checks.items() if check.status == "error"}
    if not failed:
        return EXIT_OK
    if "config" in failed:
        return EXIT_INPUT_ERROR
    if failed.intersection(DATABASE_CHECKS):
        return EXIT_DATABASE_ERROR
    return EXIT_MODEL_ERROR


def _classify(exc: Exception) -> tuple[str, int]:
    if isinstance(exc, UsageError):
        return "usage", EXIT_USAGE_ERROR
    if isinstance(exc, RepositoryError):
        return "database", EXIT_DATABASE_ERROR
    if isinstance(exc, (EmbeddingError, ImportError)):
        return "model", EXIT_MODEL_ERROR
    if isinstance(
        exc,
        (
            ImageValidationError,
            ImagePreprocessingError,
            ManifestError,
            RequestFileError,
            ConfigurationError,
            ValidationError,
        ),
    ):
        return "input", EXIT_INPUT_ERROR
    return "internal", EXIT_INPUT_ERROR


def _report_failure(
    command: str | None, exc: Exception, category: str, exit_code: int
) -> int:
    label = FAILURE_LABELS.get(command or "", "usage error")
    print(f"{label}: {exc}", file=sys.stderr)
    _print_json(
        {
            "status": "error",
            "command": command,
            "error": {
                "category": category,
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
    )
    return exit_code


def _print_json(payload: Any) -> None:
    indent = 2 if sys.stdout.isatty() else None
    print(json.dumps(payload, ensure_ascii=False, indent=indent))


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        number = 0
    if number <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value!r}")
    return number
