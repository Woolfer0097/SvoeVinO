"""Evaluation queries: user photos with the wine_id that is known to be correct."""

from __future__ import annotations

from pathlib import Path

from ..contracts import EvaluationQuery
from .csv_manifest import (
    ManifestError,
    container_path_problem,
    raise_for_problems,
    read_csv_manifest,
    resolve_data_root,
)

REQUIRED_COLUMNS = ("query_image_uri", "wine_id")
KIND = "Queries file"


def read_evaluation_queries(
    queries_path: str | Path, data_root: str | Path | None = None
) -> list[EvaluationQuery]:
    """Read and validate ``queries.csv`` with query_image_uri and wine_id columns.

    query_image_uri must be an absolute container path inside DATA_ROOT and
    unique, so one photo is not counted twice. Whether the image itself can be
    read is checked per query during evaluation.
    """

    path = Path(queries_path)
    root = resolve_data_root(data_root)
    rows, problems = read_csv_manifest(path, REQUIRED_COLUMNS, kind=KIND)

    queries: list[EvaluationQuery] = []
    line_by_image_uri: dict[str, int] = {}
    for line, values in rows:
        image_uri = values["query_image_uri"]
        path_problem = container_path_problem(image_uri, root)
        if path_problem:
            problems.append(f"line {line}: query_image_uri {path_problem}")
            continue
        if image_uri in line_by_image_uri:
            problems.append(
                f"line {line}: duplicate query_image_uri {image_uri} "
                f"(first on line {line_by_image_uri[image_uri]})"
            )
            continue
        line_by_image_uri[image_uri] = line
        queries.append(EvaluationQuery(**values))

    raise_for_problems(problems, path, KIND)
    if not queries:
        raise ManifestError(f"Queries file has no queries: {path}")
    return queries
