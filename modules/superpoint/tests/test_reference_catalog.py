from __future__ import annotations

from superpoint.infrastructure.database.reference_catalog import (
    PostgresReferenceCatalog,
)


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return self.rows


class FakeConnection:
    def __init__(self, rows):
        self.rows = rows
        self.query = ""
        self.params = None

    def execute(self, query, params):
        self.query = query
        self.params = params
        return FakeCursor(self.rows)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_catalog_groups_paths_by_wine_id() -> None:
    connection = FakeConnection(
        [
            ("shepot", "/data/reference/shepot.webp"),
            ("shepot", "/data/reference/shepot-2.webp"),
            ("other", "/data/reference/other.webp"),
        ]
    )

    photos = PostgresReferenceCatalog(
        database_url="postgresql://dinov2:dinov2@127.0.0.1:5433/dinov2",
        connect=lambda url: connection,
    ).image_uris(["shepot", "other"])

    assert photos == {
        "shepot": [
            "/data/reference/shepot.webp",
            "/data/reference/shepot-2.webp",
        ],
        "other": ["/data/reference/other.webp"],
    }
    assert "reference_images" in connection.query
    assert connection.params == (["shepot", "other"],)
