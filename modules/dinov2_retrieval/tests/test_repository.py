from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest

psycopg = pytest.importorskip("psycopg")
np = pytest.importorskip("numpy")
pytest.importorskip("pgvector")

from dinov2_retrieval.contracts import ReferenceImageRecord, ReferenceMatch
from dinov2_retrieval.infrastructure.database import connection as connection_module
from dinov2_retrieval.infrastructure.database.connection import (
    SCHEMA_PATH,
    DatabaseConnectionError,
    initialize_schema,
    inspect_schema,
)
from dinov2_retrieval.infrastructure.database.postgres_reference_repository import (
    PostgresReferenceRepository,
)
from dinov2_retrieval.retrieval.reference_repository import RepositoryError

MODULE_ROOT = Path(__file__).resolve().parents[1]


class FakeCursor:
    def __init__(self, rows: list[tuple]) -> None:
        self.rows = rows
        self.rowcount = len(rows)

    def fetchone(self) -> tuple | None:
        return self.rows[0] if self.rows else None

    def fetchall(self) -> list[tuple]:
        return self.rows


class FakeConnection:
    """Records queries; answers with rows chosen by a substring of the SQL."""

    def __init__(self, answers: dict[str, list[tuple]] | None = None) -> None:
        self.answers = answers or {}
        self.queries: list[tuple[str, object]] = []
        self.transactions = 0
        self.closed = False
        self.error: Exception | None = None

    def execute(self, query: str, params: object = None) -> FakeCursor:
        if self.error is not None:
            raise self.error
        self.queries.append((query, params))
        for fragment, rows in self.answers.items():
            if fragment in query:
                return FakeCursor(rows)
        return FakeCursor([])

    @contextmanager
    def transaction(self):
        self.transactions += 1
        yield

    def close(self) -> None:
        self.closed = True


def record(image_uri: str = "/data/reference/wine-001/photo-1.jpg") -> ReferenceImageRecord:
    return ReferenceImageRecord(
        wine_id="wine-001",
        slug="cabernet-2020",
        image_uri=image_uri,
        model_name="facebook/dinov2-small",
        embedding=[0.6, 0.8] + [0.0] * 382,
    )


@pytest.mark.parametrize(("inserted", "outcome"), [(True, "inserted"), (False, "updated")])
def test_upsert_reports_insert_or_update(inserted: bool, outcome: str) -> None:
    connection = FakeConnection({"INSERT INTO reference_images": [(inserted,)]})

    assert PostgresReferenceRepository(connection).upsert_reference_embedding(record()) == outcome

    query, params = connection.queries[0]
    assert "ON CONFLICT (image_uri) DO UPDATE" in query
    assert "updated_at = NOW()" in query
    assert params["image_uri"] == "/data/reference/wine-001/photo-1.jpg"
    assert params["model_name"] == "facebook/dinov2-small"
    assert params["embedding"].dtype == np.float32
    assert params["embedding"].shape == (384,)
    assert params["embedding"][:2].tolist() == pytest.approx([0.6, 0.8])


def test_search_uses_cosine_distance_and_maps_rows() -> None:
    connection = FakeConnection(
        {
            "embedding <=>": [
                ("wine-001", "cabernet-2020", "/data/reference/wine-001/photo-1.jpg", 0.06),
                ("wine-002", "merlot-2021", "/data/reference/wine-002/photo-1.jpg", 0.25),
            ]
        }
    )

    matches = PostgresReferenceRepository(connection).search_similar(
        [0.6, 0.8], limit=100, model_name="facebook/dinov2-small"
    )

    assert matches == [
        ReferenceMatch(
            wine_id="wine-001",
            slug="cabernet-2020",
            image_uri="/data/reference/wine-001/photo-1.jpg",
            distance=0.06,
        ),
        ReferenceMatch(
            wine_id="wine-002",
            slug="merlot-2021",
            image_uri="/data/reference/wine-002/photo-1.jpg",
            distance=0.25,
        ),
    ]
    query, params = connection.queries[0]
    assert "ORDER BY embedding <=> %(query)s" in query
    assert "WHERE model_name = %(model_name)s" in query
    assert params["limit"] == 100
    assert params["model_name"] == "facebook/dinov2-small"
    assert params["query"].tolist() == pytest.approx([0.6, 0.8])


def test_count_all_and_per_model() -> None:
    connection = FakeConnection({"COUNT(*)": [(7,)]})
    repository = PostgresReferenceRepository(connection)

    assert repository.get_reference_count() == 7
    assert repository.get_reference_count("facebook/dinov2-small") == 7
    assert connection.queries[0][1] is None
    assert connection.queries[1][1] == {"model_name": "facebook/dinov2-small"}


def test_delete_references_except_keeps_listed_photos_of_the_model() -> None:
    connection = FakeConnection({"DELETE FROM reference_images": [(1,), (2,)]})

    deleted = PostgresReferenceRepository(connection).delete_references_except(
        "facebook/dinov2-small", {"/data/reference/b.webp", "/data/reference/a.webp"}
    )

    assert deleted == 2
    query, params = connection.queries[0]
    assert "WHERE model_name = %(model_name)s AND NOT (image_uri = ANY(%(keep)s))" in query
    assert params == {
        "model_name": "facebook/dinov2-small",
        "keep": ["/data/reference/a.webp", "/data/reference/b.webp"],
    }


def test_database_errors_become_repository_errors() -> None:
    connection = FakeConnection()
    connection.error = psycopg.OperationalError("server closed the connection")
    repository = PostgresReferenceRepository(connection)

    with pytest.raises(RepositoryError, match="Cannot save reference embedding"):
        repository.upsert_reference_embedding(record())
    with pytest.raises(RepositoryError, match="Cannot search"):
        repository.search_similar([0.1], limit=1, model_name="m")
    with pytest.raises(RepositoryError, match="Cannot count"):
        repository.get_reference_count()
    with pytest.raises(RepositoryError, match="Cannot delete stale"):
        repository.delete_references_except("m", [])


def test_close_closes_the_connection() -> None:
    connection = FakeConnection()

    PostgresReferenceRepository(connection).close()

    assert connection.closed


def test_connection_errors_hide_the_password(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args, **kwargs):
        raise psycopg.OperationalError("connection refused")

    monkeypatch.setattr(connection_module.psycopg, "connect", refuse)

    with pytest.raises(DatabaseConnectionError) as error:
        connection_module.connect("postgresql://dinov2:secret@postgres:5432/dinov2")

    assert "dinov2:***@postgres" in str(error.value)
    assert "secret" not in str(error.value)
    assert isinstance(error.value, RepositoryError)


def test_initialize_schema_runs_schema_under_advisory_lock() -> None:
    connection = FakeConnection()

    initialize_schema(connection)

    assert connection.transactions == 1
    assert "pg_advisory_xact_lock" in connection.queries[0][0]
    assert connection.queries[1][0] == SCHEMA_PATH.read_text(encoding="utf-8")


def test_inspect_schema_reports_dimension_and_counts() -> None:
    connection = FakeConnection(
        {
            "server_version": [("16.4",)],
            "pg_extension": [("0.8.0",)],
            "to_regclass": [(True,)],
            "format_type": [("vector(384)",)],
            "FILTER": [(5, 3)],
        }
    )

    inspection = inspect_schema(connection, "facebook/dinov2-small")

    assert inspection.server_version == "16.4"
    assert inspection.pgvector_version == "0.8.0"
    assert inspection.table_exists
    assert inspection.embedding_dimension == 384
    assert (inspection.reference_count, inspection.model_reference_count) == (5, 3)


def test_inspect_schema_without_table() -> None:
    connection = FakeConnection(
        {"server_version": [("16.4",)], "to_regclass": [(False,)]}
    )

    inspection = inspect_schema(connection, "facebook/dinov2-small")

    assert inspection.pgvector_version is None
    assert not inspection.table_exists
    assert inspection.embedding_dimension is None


def test_schema_files_are_identical_and_match_the_spec() -> None:
    schema = SCHEMA_PATH.read_text(encoding="utf-8")

    assert (MODULE_ROOT / "database" / "init.sql").read_text(encoding="utf-8") == schema
    assert "CREATE EXTENSION IF NOT EXISTS vector;" in schema
    assert "image_uri TEXT NOT NULL UNIQUE" in schema
    assert "embedding VECTOR(384) NOT NULL" in schema
