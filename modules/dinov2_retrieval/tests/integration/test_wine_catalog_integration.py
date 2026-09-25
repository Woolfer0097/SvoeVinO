"""Manual PostgreSQL/pgvector check for the CSV and shared photo pipeline."""

from __future__ import annotations

import os
import uuid
from io import BytesIO
from pathlib import Path

import pytest

psycopg = pytest.importorskip("psycopg")
Image = pytest.importorskip("PIL.Image")

from dinov2_retrieval.wine_catalog.importer import import_csv
from dinov2_retrieval.wine_catalog.scraper import WinePage, _save_success, _save_verified_image


@pytest.mark.integration
def test_duplicate_slugs_reimport_and_one_photo_for_all_rows(tmp_path: Path) -> None:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        pytest.skip("DATABASE_URL is not set")

    suffix = uuid.uuid4().hex
    slug = f"pipeline-check-{suffix}"
    csv_path = tmp_path / "wines.csv"
    csv_path.write_text(
        "Название вина,Slug,Название фото,extra\n"
        f"Первое,{slug},dataset-1.webp,one\n"
        f"Второе,{slug},dataset-2.webp,two\n",
        encoding="utf-8",
    )
    source_id = f"integration:{suffix}"
    buffer = BytesIO()
    Image.new("RGB", (160, 160), color=(123, 45, 67)).save(buffer, format="JPEG")

    with psycopg.connect(database_url, autocommit=True) as connection:
        connection.execute("BEGIN")
        try:
            first = import_csv(connection, csv_path, source_id=source_id)
            second = import_csv(connection, csv_path, source_id=source_id)
            assert first["rows"] == second["rows"] == 2
            assert first["unique_slugs"] == second["unique_slugs"] == 1
            assert connection.execute(
                "SELECT COUNT(*) FROM wines WHERE import_source = %s", (source_id,)
            ).fetchone()[0] == 2

            connection.execute(
                "INSERT INTO wine_scrape_jobs (slug) VALUES (%s) ON CONFLICT DO NOTHING",
                (slug,),
            )
            saved_image = _save_verified_image(slug, buffer.getvalue(), tmp_path)
            wine = WinePage(image_url="https://vino-svoe.ru/media/reference.jpg")
            _save_success(connection, slug, wine, saved_image.name)
            import_csv(connection, csv_path, source_id=source_id)

            rows = connection.execute(
                "SELECT name, dataset_photo, web_photo, web_photo_url, source_data, "
                "dataset_photo_embedding, description_text_embedding, web_photo_embedding "
                "FROM wines WHERE import_source = %s ORDER BY source_row_number",
                (source_id,),
            ).fetchall()
            assert len(rows) == 2
            assert [row[1] for row in rows] == ["dataset-1.webp", "dataset-2.webp"]
            assert [row[0] for row in rows] == ["Первое", "Второе"]
            assert {row[2] for row in rows} == {saved_image.name}
            assert {row[3] for row in rows} == {wine.image_url}
            assert [row[4]["extra"] for row in rows] == ["one", "two"]
            assert all(row[5:] == (None, None, None) for row in rows)
        finally:
            connection.execute("ROLLBACK")
