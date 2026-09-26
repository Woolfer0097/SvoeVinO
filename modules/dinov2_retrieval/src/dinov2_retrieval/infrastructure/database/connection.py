"""PostgreSQL connection, schema initialization and schema inspection."""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import psycopg

from ...config import get_database_url, redact_database_url
from ...contracts import DatabaseInspection
from ...retrieval.reference_repository import RepositoryError

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
DEFAULT_CONNECT_TIMEOUT_SECONDS = 10
# Serializes concurrent schema creation, e.g. parallel Airflow tasks.
SCHEMA_LOCK_KEY = 0x44494E4F

_VECTOR_TYPE_PATTERN = re.compile(r"^vector\((\d+)\)$")


class DatabaseConnectionError(RepositoryError):
    """Raised when PostgreSQL is unreachable or rejects the connection."""


def connect(
    database_url: str | None = None,
    *,
    connect_timeout: int = DEFAULT_CONNECT_TIMEOUT_SECONDS,
) -> psycopg.Connection:
    """Open an autocommit connection to the database from ``DATABASE_URL``."""

    url = get_database_url(database_url)
    try:
        return psycopg.connect(url, autocommit=True, connect_timeout=connect_timeout)
    except psycopg.Error as exc:
        raise DatabaseConnectionError(
            f"Cannot connect to PostgreSQL at {redact_database_url(url)}: {exc}"
        ) from exc


@contextmanager
def open_connection(database_url: str | None = None) -> Iterator[psycopg.Connection]:
    """Yield a connection and always close it afterwards."""

    connection = connect(database_url)
    try:
        yield connection
    finally:
        connection.close()


def initialize_schema(connection: psycopg.Connection) -> None:
    """Create the pgvector extension and the reference_images table if missing."""

    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    try:
        with connection.transaction():
            connection.execute("SELECT pg_advisory_xact_lock(%s)", (SCHEMA_LOCK_KEY,))
            connection.execute(schema_sql)
    except psycopg.Error as exc:
        raise RepositoryError(f"Cannot initialize database schema: {exc}") from exc


def inspect_database(
    model_name: str, database_url: str | None = None
) -> DatabaseInspection:
    """Connect without changing anything and describe the schema state."""

    with open_connection(database_url) as connection:
        return inspect_schema(connection, model_name)


def inspect_schema(
    connection: psycopg.Connection, model_name: str
) -> DatabaseInspection:
    """Describe pgvector, the reference_images table and its embedding column."""

    try:
        server_version = connection.execute("SHOW server_version").fetchone()[0]
        extension = connection.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
        table_exists = connection.execute(
            "SELECT to_regclass('reference_images') IS NOT NULL"
        ).fetchone()[0]

        embedding_dimension = None
        reference_count = None
        model_reference_count = None
        if table_exists:
            column = connection.execute(
                "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                "WHERE attrelid = 'reference_images'::regclass "
                "AND attname = 'embedding' AND NOT attisdropped"
            ).fetchone()
            embedding_dimension = _parse_vector_dimension(column[0] if column else None)
            reference_count, model_reference_count = connection.execute(
                "SELECT COUNT(*), COUNT(*) FILTER (WHERE model_name = %s) "
                "FROM reference_images",
                (model_name,),
            ).fetchone()
    except psycopg.Error as exc:
        raise RepositoryError(f"Cannot inspect database schema: {exc}") from exc

    return DatabaseInspection(
        server_version=server_version,
        pgvector_version=extension[0] if extension else None,
        table_exists=bool(table_exists),
        embedding_dimension=embedding_dimension,
        reference_count=reference_count,
        model_reference_count=model_reference_count,
    )


def _parse_vector_dimension(column_type: str | None) -> int | None:
    match = _VECTOR_TYPE_PATTERN.match(column_type or "")
    return int(match.group(1)) if match else None
