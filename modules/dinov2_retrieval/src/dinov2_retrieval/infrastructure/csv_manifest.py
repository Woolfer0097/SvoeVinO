"""Shared reading and validation of CSV manifests (reference photos, evaluation queries)."""

from __future__ import annotations

import csv
import posixpath
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from ..config import get_data_root

MAX_REPORTED_PROBLEMS = 20

ManifestRow = tuple[int, dict[str, str]]


class ManifestError(ValueError):
    """Raised when a manifest, a queries file or a photo folder cannot be used."""


def read_csv_manifest(
    manifest_path: str | Path,
    required_columns: Sequence[str],
    *,
    kind: str = "Manifest",
) -> tuple[list[ManifestRow], list[str]]:
    """Return ``(line, values)`` of every row with all required columns filled.

    The file must be UTF-8 (a BOM is allowed) with a header row. Extra columns
    are ignored, values are stripped and blank rows are skipped. Rows with
    problems are left out and described in the returned problem list, so the
    caller can add its own checks and report everything at once.
    """

    path = Path(manifest_path)
    try:
        with path.open(encoding="utf-8-sig", newline="") as manifest_file:
            reader = csv.reader(manifest_file)
            header = next(reader, None)
            if header is None:
                raise ManifestError(f"{kind} is empty: {path}")
            positions = _column_positions(header, required_columns, path, kind)
            return _read_rows(reader, positions, len(header))
    except FileNotFoundError as exc:
        raise ManifestError(f"{kind} does not exist: {path}") from exc
    except IsADirectoryError as exc:
        raise ManifestError(f"{kind} path is a directory: {path}") from exc
    except UnicodeDecodeError as exc:
        raise ManifestError(f"{kind} must be UTF-8 encoded: {path}") from exc
    except csv.Error as exc:
        raise ManifestError(f"Cannot parse {kind.lower()} {path}: {exc}") from exc
    except OSError as exc:
        raise ManifestError(f"Cannot read {kind.lower()} {path}: {exc}") from exc


def container_path_problem(image_uri: str, data_root: Path) -> str | None:
    """Explain why ``image_uri`` is not an absolute path inside DATA_ROOT, if it is not."""

    if not image_uri.startswith("/"):
        return f"path must be absolute and start with {data_root}: {image_uri}"
    root = PurePosixPath(data_root.as_posix())
    normalized = PurePosixPath(posixpath.normpath(image_uri))
    if normalized.is_relative_to(root):
        return None
    # DATA_ROOT is resolved, so a path through a symlink to it is still inside.
    if Path(image_uri).resolve(strict=False).is_relative_to(data_root):
        return None
    return f"path must be inside DATA_ROOT {data_root}: {image_uri}"


def resolve_data_root(data_root: str | Path | None) -> Path:
    """Use an explicit root as is, otherwise the validated ``DATA_ROOT``."""

    return Path(data_root) if data_root is not None else get_data_root()


def raise_for_problems(problems: list[str], path: Path, kind: str) -> None:
    if not problems:
        return
    shown = problems[:MAX_REPORTED_PROBLEMS]
    hidden = len(problems) - len(shown)
    suffix = f"; ... and {hidden} more" if hidden else ""
    raise ManifestError(f"Invalid {kind.lower()} {path}: " + "; ".join(shown) + suffix)


def _column_positions(
    header: list[str], required_columns: Sequence[str], path: Path, kind: str
) -> dict[str, int]:
    columns = [column.strip() for column in header]
    missing = [name for name in required_columns if name not in columns]
    if missing:
        raise ManifestError(
            f"{kind} {path} is missing required columns: {', '.join(missing)}. "
            f"Found columns: {', '.join(columns) or '(none)'}"
        )
    repeated = [name for name in required_columns if columns.count(name) > 1]
    if repeated:
        raise ManifestError(f"{kind} {path} repeats columns: {', '.join(repeated)}")
    return {name: columns.index(name) for name in required_columns}


def _read_rows(
    reader, positions: dict[str, int], column_count: int
) -> tuple[list[ManifestRow], list[str]]:
    rows: list[ManifestRow] = []
    problems: list[str] = []
    for row in reader:
        line = reader.line_num
        if not any(cell.strip() for cell in row):
            continue
        if len(row) > column_count:
            problems.append(f"line {line}: {len(row)} values for {column_count} columns")
            continue

        values = {
            name: row[index].strip() if index < len(row) else ""
            for name, index in positions.items()
        }
        empty = [name for name, value in values.items() if not value]
        if empty:
            problems.append(f"line {line}: empty {', '.join(empty)}")
            continue
        rows.append((line, values))
    return rows, problems
