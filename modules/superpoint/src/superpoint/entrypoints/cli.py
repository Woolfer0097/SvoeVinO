"""CLI entrypoint for SuperPoint+LightGlue photo verification."""

from __future__ import annotations

import argparse
import sys

from ..application.validate_image import validate_image
from ..application.verify_photos import verify_photos
from ..contracts import VerificationRequest
from ..infrastructure.storage.local_storage import ImageValidationError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="superpoint")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate",
        help="validate a local image without matching it",
    )
    validate_parser.add_argument("--image-uri", required=True)

    verify_parser = subparsers.add_parser(
        "verify",
        help="verify a query photo against a reference photo",
    )
    verify_parser.add_argument("--query-uri", required=True)
    verify_parser.add_argument("--reference-uri", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "validate":
        try:
            validated_image = validate_image(args.image_uri)
        except (ImageValidationError, ValueError) as exc:
            print(f"validation failed: {exc}", file=sys.stderr)
            return 1
        print(validated_image.model_dump_json(indent=2))
        return 0

    if args.command == "verify":
        request = VerificationRequest(
            query_uri=args.query_uri,
            reference_uri=args.reference_uri,
        )
        try:
            result = verify_photos(request)
        except (ImageValidationError, ImportError, RuntimeError, ValueError, OSError) as exc:
            print(f"verification failed: {exc}", file=sys.stderr)
            return 1
        print(result.model_dump_json(indent=2))
        return 0

    return 2
