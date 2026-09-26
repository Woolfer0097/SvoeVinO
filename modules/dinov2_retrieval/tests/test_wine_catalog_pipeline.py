from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from dinov2_retrieval.wine_catalog import importer
from dinov2_retrieval.wine_catalog.importer import import_csv
from dinov2_retrieval.wine_catalog.scraper import WinePage, _save_success, parse_wine_page


class _Result:
    def __init__(self, row=None, rows=None):
        self._row = row
        self._rows = rows or []

    def fetchone(self):
        return self._row

    def fetchall(self):
        return self._rows


class _FakeConnection:
    def __init__(self):
        self.migrations = set()
        self.wines = {}
        self.statements = []

    @contextmanager
    def transaction(self):
        yield

    def execute(self, query, params=None):
        query = query.strip()
        params = params or {}
        self.statements.append((query, params))
        lowered = query.lower()
        if lowered.startswith("select 1 from schema_migrations"):
            return _Result((1,) if params[0] in self.migrations else None)
        if lowered.startswith("insert into schema_migrations"):
            self.migrations.add(params[0])
            return _Result()
        if lowered.startswith("insert into wines"):
            key = (params["import_source"], params["source_row_number"])
            self.wines[key] = dict(params)
            return _Result()
        if lowered.startswith("delete from wines"):
            source, highest_row = params
            self.wines = {
                key: row for key, row in self.wines.items()
                if row["import_source"] != source or row["source_row_number"] <= highest_row
            }
            return _Result()
        if lowered.startswith("update wines"):
            for row in self.wines.values():
                if row.get("slug") == params["slug"]:
                    row["web_photo"] = params["web_photo"]
                    row["web_photo_url"] = params["web_photo_url"]
            return _Result()
        return _Result()


def test_import_is_idempotent_and_preserves_duplicate_slug_rows(
    tmp_path: Path, monkeypatch
) -> None:
    csv_path = tmp_path / "wines.csv"
    csv_path.write_text(
        "Название вина,Категория,Цвет,Регион,Сорт винограда,Описание,Винодельня,Slug,Название фото,extra\n"
        "Wine A,Белое,золотистый,Крым,Алиготе,Текст,Хозяйство,shared-slug,a.webp,first\n"
        "Wine A,Белое,золотистый,Крым,Алиготе,Текст,Хозяйство,shared-slug,b.webp,second\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(importer, "_jsonb", lambda value: value)
    connection = _FakeConnection()

    first = import_csv(connection, csv_path)
    second = import_csv(connection, csv_path)

    assert first["rows"] == second["rows"] == 2
    assert first["unique_slugs"] == second["unique_slugs"] == 1
    assert len(connection.wines) == 2
    assert {row["dataset_photo"] for row in connection.wines.values()} == {"a.webp", "b.webp"}
    assert {row["source_data"]["extra"] for row in connection.wines.values()} == {"first", "second"}
    assert {row["source_row_number"] for row in connection.wines.values()} == {2, 3}


def test_page_parser_reads_product_photo_without_metadata() -> None:
    page_html = """
    <html><head>
      <meta property="og:image" content="/media/orange.webp">
      <meta name="description" content="Ароматное сухое вино">
    </head><body>
      <h1>Orange, 2023</h1><h2>WINEMAFIA</h2>
      <span>Регион</span><span>Кубань</span>
      <span>Сорт винограда</span><span>Рислинг</span>
      <span>Категория и цвет</span><span>Белое сухое</span><span>Сухой насыщенной соломы</span>
      <span>Температура подачи</span>
    </body></html>
    """

    wine = parse_wine_page(page_html, "https://vino-svoe.ru/wines/orange")

    assert wine.image_url == "https://vino-svoe.ru/media/orange.webp"


def test_page_parser_prefers_structured_product_image() -> None:
    page_html = """
    <html><head><meta property="og:image" content="/media/site-logo.png">
      <script type="application/ld+json">
        {"@type": "Product", "name": "Русское Игристое", "image": "/media/abrau.webp"}
      </script></head><body>
      <h1>Русское Игристое</h1>
    </body></html>
    """

    wine = parse_wine_page(page_html, "https://vino-svoe.ru/wines/abrau-dyurso")

    assert wine.image_url == "https://vino-svoe.ru/media/abrau.webp"


def test_page_parser_prefers_observed_bottle_image_over_social_preview() -> None:
    page_html = """
    <html><head><meta property="og:image" content="/media/social-preview.webp"></head>
    <body><h1>Другое написание имени</h1>
      <img class="wine-hero-block__bottle" alt="Вино на фото"
           src="https://api.vino-svoe.ru/v1/img/str-api/1160/1160/resize/uploads/wine.webp">
    </body></html>
    """
    wine = parse_wine_page(page_html, "https://vino-svoe.ru/wines/example")
    assert wine.image_url.endswith("/uploads/wine.webp")


def test_scraped_photo_updates_every_slug_match_without_touching_source_photos() -> None:
    connection = _FakeConnection()
    connection.wines = {
        ("source", 2): {"slug": "same", "name": "Первое", "dataset_photo": "one.webp", "source_data": {"row": 1}},
        ("source", 3): {"slug": "same", "name": "Второе", "dataset_photo": "two.webp", "source_data": {"row": 2}},
        ("source", 4): {"slug": "other", "dataset_photo": "other.webp", "source_data": {"row": 3}},
    }
    wine = WinePage(image_url="https://cdn.example/photo.jpg")

    _save_success(connection, "same", wine, "data/web_photos/same.jpg")

    matches = [row for row in connection.wines.values() if row["slug"] == "same"]
    other = next(row for row in connection.wines.values() if row["slug"] == "other")
    assert [row["web_photo"] for row in matches] == [
        "data/web_photos/same.jpg", "data/web_photos/same.jpg"
    ]
    assert [row["dataset_photo"] for row in matches] == ["one.webp", "two.webp"]
    assert [row["name"] for row in matches] == ["Первое", "Второе"]
    assert [row["source_data"] for row in matches] == [{"row": 1}, {"row": 2}]
    assert other.get("web_photo") is None
