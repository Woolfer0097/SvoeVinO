"""CLI entrypoint for local image validation."""

from __future__ import annotations

import argparse
import json
import sys

from ..application.create_embedding import create_embedding
from ..application.validate_image import validate_image
from ..contracts import RetrievalRequest
from ..infrastructure.storage.local_storage import ImageValidationError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dinov2-retrieval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate", help="validate a local image without creating an embedding"
    )
    validate_parser.add_argument("--image-uri", required=True)
    validate_parser.add_argument("--job-id", default="local-validation")
    validate_parser.add_argument("--query-id", default="local-validation")
    validate_parser.add_argument("--top-k", type=int, default=20)

    embed_parser = subparsers.add_parser(
        "embed", help="create a DINOv2 embedding for a local image"
    )
    embed_parser.add_argument("--image-uri", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "validate":
        request = RetrievalRequest(
            job_id=args.job_id,
            query_id=args.query_id,
            image_uri=args.image_uri,
            top_k=args.top_k,
        )
        try:
            validated_image = validate_image(request)
        except (ImageValidationError, ValueError) as exc:
            print(f"validation failed: {exc}", file=sys.stderr)
            return 1

        print(validated_image.model_dump_json(indent=2))
        return 0

    if args.command == "embed":
        try:
            result = create_embedding(args.image_uri)
        except (ImageValidationError, ImportError, RuntimeError, ValueError) as exc:
            print(f"embedding failed: {exc}", file=sys.stderr)
            return 1

        print(
            json.dumps(
                {
                    "path": result.path,
                    "model": result.model,
                    "dimension": result.dimension,
                    "device": result.device,
                    "embedding_preview": result.embedding[:5],
                },
                indent=2,
            )
        )
        return 0

    return 2
