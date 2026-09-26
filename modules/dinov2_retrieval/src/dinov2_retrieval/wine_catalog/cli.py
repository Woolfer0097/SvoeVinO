"""Command line entry point for CSV import and wine-card scraping."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ..config import get_database_url
from .importer import import_csv
from .migrations import apply_migrations
from .scraper import scrape_wines
from .embeddings import KINDS, generate_embeddings
from .photo_manifest import build_photo_manifest, stage_manifest_photos, write_photo_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wine-pipeline")
    commands = parser.add_subparsers(dest="command", required=True)

    importer = commands.add_parser("import-csv", help="import every CSV row into PostgreSQL")
    importer.add_argument("csv_path", type=Path)
    importer.add_argument("--source-id", help="stable import identity; defaults to the resolved CSV path")
    importer.add_argument("--database-url", help="overrides DATABASE_URL")

    scraper = commands.add_parser("scrape", help="scrape cards for imported unique slugs")
    scraper.add_argument("--database-url", help="overrides DATABASE_URL")
    scraper.add_argument("--slugs", help="comma-separated sample of imported slugs")
    scraper.add_argument("--limit", type=_positive_int, help="process at most this many cards")
    scraper.add_argument("--retry-failed", action="store_true", help="process only slugs marked failed")
    scraper.add_argument(
        "--photos-dir", type=Path,
        default=Path(os.getenv("WINE_PHOTOS_DIR", "data/web_photos")),
    )
    scraper.add_argument("--timeout", type=float, default=20.0)
    scraper.add_argument("--delay", type=float, default=1.0, help="minimum seconds between HTTP requests")
    scraper.add_argument("--retries", type=int, default=3)
    scraper.add_argument("--interactive", action="store_true", help="open a persistent Playwright browser for manual CAPTCHA/age checks")
    scraper.add_argument("--cookie-jar", type=Path, help="defaults beside --photos-dir")
    scraper.add_argument("--browser-profile", type=Path, help="defaults beside --photos-dir")

    manifest = commands.add_parser("photo-manifest", help="match CSV photo names to Strapi uploads")
    manifest.add_argument("--database-url", help="overrides DATABASE_URL")
    manifest.add_argument("--uploads-root", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)

    stage = commands.add_parser("stage-photos", help="copy referenced CSV photos for transfer")
    stage.add_argument("--uploads-root", type=Path, required=True)
    stage.add_argument("--photo-manifest", type=Path, required=True)
    stage.add_argument("--output", type=Path, required=True)

    embedder = commands.add_parser("embed", help="generate missing wine vectors on a GPU machine")
    embedder.add_argument("--database-url", help="overrides DATABASE_URL")
    embedder.add_argument("--kind", choices=("all", *KINDS), default="all")
    embedder.add_argument("--uploads-root", type=Path)
    embedder.add_argument("--photo-manifest", type=Path)
    embedder.add_argument("--web-photos-root", type=Path, default=Path(os.getenv("WINE_PHOTOS_DIR", "data/web_photos")))
    embedder.add_argument("--image-model", default="facebook/dinov2-small")
    embedder.add_argument("--text-model", default="intfloat/multilingual-e5-base")
    embedder.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    embedder.add_argument("--text-batch-size", type=_positive_int, default=16)
    embedder.add_argument("--limit", type=_positive_int)
    embedder.add_argument("--retry-failed", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "import-csv":
            result = _run_import(args)
        elif args.command == "scrape":
            result = _run_scrape(args)
        elif args.command == "photo-manifest":
            result = _run_photo_manifest(args)
        elif args.command == "stage-photos":
            result = {"command": "stage-photos", "status": "ok", **stage_manifest_photos(
                args.photo_manifest, args.uploads_root, args.output,
            )}
        else:
            result = _run_embed(args)
    except Exception as exc:
        print(json.dumps({"command": args.command, "status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("errors", 0):
        return 5
    return 0


def _run_import(args: argparse.Namespace) -> dict[str, int | str]:
    with _connect(args.database_url) as connection:
        applied = apply_migrations(connection)
        summary = import_csv(connection, args.csv_path, source_id=args.source_id)
        imported_rows = connection.execute(
            "SELECT COUNT(*) FROM wines WHERE import_source = %s",
            (summary["source_id"],),
        ).fetchone()[0]
        if imported_rows != summary["rows"]:
            raise RuntimeError(
                f"Import count mismatch: CSV has {summary['rows']} records but "
                f"PostgreSQL has {imported_rows} records for this source"
            )
    return {
        "command": "import-csv",
        "status": "ok",
        "csv_path": str(args.csv_path.resolve()),
        "source_id": summary["source_id"],
        "csv_rows": summary["rows"],
        "imported_records": imported_rows,
        "unique_slugs": summary["unique_slugs"],
        "expected_records": 2109,
        "row_difference": int(summary["rows"]) - 2109,
        "unique_slug_difference": int(summary["unique_slugs"]) - 2109,
        "migrations_applied": applied,
    }


def _run_scrape(args: argparse.Namespace) -> dict[str, object]:
    slugs = [slug.strip() for slug in args.slugs.split(",") if slug.strip()] if args.slugs else None
    with _connect(args.database_url) as connection:
        summary = scrape_wines(
            connection,
            slugs=slugs,
            retry_failed=args.retry_failed,
            limit=args.limit,
            photos_dir=args.photos_dir,
            timeout_seconds=args.timeout,
            min_delay_seconds=args.delay,
            retries=args.retries,
            interactive=args.interactive,
            cookie_jar_path=args.cookie_jar,
            browser_profile=args.browser_profile,
        )
    return {"command": "scrape", "status": "ok" if not summary["errors"] else "partial", **summary}


def _run_photo_manifest(args: argparse.Namespace) -> dict[str, object]:
    with _connect(args.database_url) as connection:
        apply_migrations(connection)
        names = [
            row[0] for row in connection.execute(
                "SELECT DISTINCT dataset_photo FROM wines "
                "WHERE dataset_photo IS NOT NULL ORDER BY dataset_photo"
            ).fetchall()
        ]
    rows = build_photo_manifest(names, args.uploads_root)
    write_photo_manifest(rows, args.output)
    counts = {status: sum(row["status"] == status for row in rows)
              for status in ("matched", "ambiguous", "missing")}
    return {"command": "photo-manifest", "status": "ok", "output": str(args.output),
            "unique_photos": len(rows), **counts}


def _run_embed(args: argparse.Namespace) -> dict[str, object]:
    kinds = KINDS if args.kind == "all" else (args.kind,)
    with _connect(args.database_url) as connection:
        summary = generate_embeddings(
            connection, kinds=kinds, uploads_root=args.uploads_root,
            manifest_path=args.photo_manifest, web_photos_root=args.web_photos_root,
            image_model=args.image_model, text_model=args.text_model,
            device=args.device, text_batch_size=args.text_batch_size,
            limit=args.limit, retry_failed=args.retry_failed,
        )
    errors = sum(item["errors"] for item in summary.values())
    return {"command": "embed", "status": "partial" if errors else "ok",
            "results": summary, "errors": errors}


def _connect(database_url: str | None):
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError("PostgreSQL support is optional; install with `pip install -e '.[db,catalog]'`") from exc
    return psycopg.connect(get_database_url(database_url), autocommit=True, connect_timeout=10)


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


if __name__ == "__main__":
    sys.exit(main())
