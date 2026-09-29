"""Public catalog cards, without exposing embeddings or import internals."""
from __future__ import annotations

import re
from pathlib import PurePath

import psycopg

FIELDS = ("name", "category", "color", "region", "grape_variety", "description", "winery")
KNOWN_LABELS = {
    "Slug", "Название фото", "Название вина", "Категория", "Цвет", "Регион",
    "Сорт винограда", "Описание", "Винодельня",
}
YEAR = re.compile(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)")


def make_card(row: tuple) -> dict:
    slug, *values, photo, source = row
    card = dict(zip(FIELDS, (str(value).strip() if value is not None else "" for value in values)))
    explicit = {int(year) for year in YEAR.findall(card["name"] + " " + slug)}
    card["vintage"] = next(iter(explicit)) if len(explicit) == 1 else None
    card["web_photo_uri"] = (
        "/data/web_photos/" + PurePath(photo).name
        if photo and PurePath(photo).name == photo and "\\" not in photo else None
    )
    card["attributes"] = [
        {"label": key, "value": str(value).strip()}
        for key, value in (source or {}).items()
        if key not in KNOWN_LABELS and isinstance(value, (str, int, float))
        and str(value).strip()
    ]
    return card


class Catalog:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    async def _query(self, sql, params):
        async with await psycopg.AsyncConnection.connect(
            self.database_url, connect_timeout=5,
            options="-c default_transaction_read_only=on",
        ) as connection:
            cursor = await connection.execute(sql, params)
            return await cursor.fetchall()

    async def slugs(self, wine_ids: list[str]) -> dict[str, str]:
        if not wine_ids:
            return {}
        if any(not value.isdecimal() for value in wine_ids):
            raise ValueError("OCR returned invalid catalog IDs")
        rows = await self._query(
            "SELECT id, slug FROM wines WHERE id = ANY(%s)",
            ([int(value) for value in wine_ids],),
        )
        return {str(wine_id): slug for wine_id, slug in rows}

    async def cards(self, slugs: list[str]) -> dict[str, dict]:
        rows = await self._query("""
            SELECT DISTINCT ON (slug) slug,
                name,category,color,region,grape_variety,description,winery,
                web_photo,source_data
            FROM wines WHERE slug = ANY(%s)
            ORDER BY slug, (web_photo IS NOT NULL) DESC, id
        """, (slugs,))
        return {row[0]: make_card(row) for row in rows}

    async def references(self, slugs: list[str], model_name: str) -> list[dict]:
        """Canonical visual references for text hits missed by DINO's Top-20."""
        if not slugs:
            return []
        rows = await self._query("""
            SELECT DISTINCT ON (slug) slug, wine_id, image_uri
            FROM reference_images WHERE slug = ANY(%s) AND model_name = %s
            ORDER BY slug, (image_uri LIKE '%%/web-%%') DESC, id
        """, (slugs, model_name))
        by_slug = {slug: {"slug": slug, "wine_id": wine_id,
                         "best_image_uri": uri, "score": None,
                         "retrieval_sources": ["ocr"]}
                   for slug, wine_id, uri in rows}
        return [by_slug[slug] for slug in slugs if slug in by_slug]


def apply_vintage(ranked: list[dict], years: list[int]):
    """Check an explicit catalog vintage; never infer a year from a description."""
    years = sorted(set(years))
    if len(years) != 1:
        return ranked, {
            "detected_year": None,
            "state": "ambiguous" if years else "not_read",
        }
    detected = years[0]
    # The list has already been shortlisted/fused; year is a discriminative
    # reranker, not permission to select an unrelated catalog wine globally.
    def priority(candidate):
        vintage = candidate.get("vintage")
        return 0 if vintage == detected else 1 if vintage is None else 2
    ranked = sorted(ranked, key=priority)  # stable within each class
    chosen = ranked[0].get("vintage") if ranked else None
    return ranked, {
        "detected_year": detected,
        "state": "matched" if chosen == detected else "unverified" if chosen is None else "mismatch",
    }
