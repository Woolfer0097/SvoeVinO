"""Idempotent import of the wine CSV while retaining every source column."""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .migrations import apply_migrations

FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "name": ("название вина", "название", "name", "wine name"),
    "category": ("категория", "category"),
    "color": ("цвет", "color"),
    "region": ("регион", "region"),
    "grape_variety": (
        "сорт винограда", "сорта винограда", "grape variety", "grape_variety"
    ),
    "description": ("описание", "description"),
    "winery": ("винодельня", "производитель", "winery", "producer"),
    "slug": ("slug", "слаг"),
    "dataset_photo": (
        "название фото", "фото", "dataset photo", "dataset_photo", "photo filename"
    ),
}
INSERT_SQL = """
INSERT INTO wines (
    name, category, color, region, grape_variety, description, winery, slug,
    dataset_photo, source_data, import_source, source_row_number
) VALUES (
    %(name)s, %(category)s, %(color)s, %(region)s, %(grape_variety)s,
    %(description)s, %(winery)s, %(slug)s, %(dataset_photo)s, %(source_data)s,
    %(import_source)s, %(source_row_number)s
)
ON CONFLICT (import_source, source_row_number) DO UPDATE SET
    name = EXCLUDED.name,
    category = EXCLUDED.category,
    color = EXCLUDED.color,
    region = EXCLUDED.region,
    grape_variety = EXCLUDED.grape_variety,
    description = EXCLUDED.description,
    winery = EXCLUDED.winery,
    slug = EXCLUDED.slug,
    dataset_photo = EXCLUDED.dataset_photo,
    source_data = EXCLUDED.source_data,
    dataset_photo_embedding = CASE WHEN wines.dataset_photo IS DISTINCT FROM EXCLUDED.dataset_photo
        THEN NULL ELSE wines.dataset_photo_embedding END,
    dataset_photo_embedding_model = CASE WHEN wines.dataset_photo IS DISTINCT FROM EXCLUDED.dataset_photo
        THEN NULL ELSE wines.dataset_photo_embedding_model END,
    description_text_embedding = CASE WHEN
        (wines.name, wines.category, wines.color, wines.region, wines.grape_variety,
         wines.description, wines.winery)
        IS DISTINCT FROM
        (EXCLUDED.name, EXCLUDED.category, EXCLUDED.color, EXCLUDED.region,
         EXCLUDED.grape_variety, EXCLUDED.description, EXCLUDED.winery)
        THEN NULL ELSE wines.description_text_embedding END,
    description_text_embedding_model = CASE WHEN
        (wines.name, wines.category, wines.color, wines.region, wines.grape_variety,
         wines.description, wines.winery)
        IS DISTINCT FROM
        (EXCLUDED.name, EXCLUDED.category, EXCLUDED.color, EXCLUDED.region,
         EXCLUDED.grape_variety, EXCLUDED.description, EXCLUDED.winery)
        THEN NULL ELSE wines.description_text_embedding_model END,
    web_photo = CASE WHEN wines.slug IS DISTINCT FROM EXCLUDED.slug
        THEN NULL ELSE wines.web_photo END,
    web_photo_url = CASE WHEN wines.slug IS DISTINCT FROM EXCLUDED.slug
        THEN NULL ELSE wines.web_photo_url END,
    web_photo_embedding = CASE WHEN wines.slug IS DISTINCT FROM EXCLUDED.slug
        THEN NULL ELSE wines.web_photo_embedding END,
    web_photo_embedding_model = CASE WHEN wines.slug IS DISTINCT FROM EXCLUDED.slug
        THEN NULL ELSE wines.web_photo_embedding_model END
"""


def iter_csv_rows(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield source record numbers and raw DictReader rows (including extras)."""

    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        for source_row_number, row in enumerate(reader, start=2):
            raw = dict(row)
            extras = raw.pop(None, None)
            if extras is not None:
                raw["_extra_columns"] = extras
            yield source_row_number, raw


def import_csv(connection, path: Path, *, source_id: str | None = None) -> dict[str, int | str]:
    """Upsert all source rows by source identity and CSV record number."""

    path = path.expanduser().resolve(strict=True)
    source_id = source_id or f"file:{path}"
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        columns = _map_columns(reader.fieldnames)
        count = 0
        slugs: set[str] = set()
        with connection.transaction():
            apply_migrations(connection)
            for source_row_number, row in enumerate(reader, start=2):
                raw: dict[str, Any] = dict(row)
                extras = raw.pop(None, None)
                if extras is not None:
                    raw["_extra_columns"] = extras
                record = {
                    name: _clean_value(row.get(source_column))
                    if source_column is not None
                    else None
                    for name, source_column in columns.items()
                }
                record.update(
                    source_data=_jsonb(raw),
                    import_source=source_id,
                    source_row_number=source_row_number,
                )
                connection.execute(INSERT_SQL, record)
                count += 1
                if record["slug"]:
                    slugs.add(record["slug"])

            # Remove trailing rows from the same source if it was shortened.
            connection.execute(
                "DELETE FROM wines WHERE import_source = %s AND source_row_number > %s",
                (source_id, count + 1),
            )

    return {"source_id": source_id, "rows": count, "unique_slugs": len(slugs)}


def _map_columns(headers: list[str]) -> dict[str, str | None]:
    normalized = {header.strip().casefold(): header for header in headers}
    mapping: dict[str, str | None] = {}
    for field, aliases in FIELD_ALIASES.items():
        mapping[field] = next(
            (normalized[alias.casefold()] for alias in aliases if alias.casefold() in normalized),
            None,
        )
    if mapping["slug"] is None:
        raise ValueError("CSV must have a slug column (Slug or slug)")
    if mapping["dataset_photo"] is None:
        raise ValueError("CSV must have a photo name column (Название фото or dataset_photo)")
    return mapping


def _clean_value(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _jsonb(value: dict[str, Any]):
    """Wrap a Python mapping for psycopg's JSONB adapter."""

    from psycopg.types.json import Jsonb

    return Jsonb(value)
