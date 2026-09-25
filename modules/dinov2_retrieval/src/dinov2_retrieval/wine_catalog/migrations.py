"""Small ordered SQL migration runner for the wine catalog pipeline."""

from __future__ import annotations

from pathlib import Path

MIGRATIONS_DIR = Path(__file__).with_name("migrations")
MIGRATION_LOCK_KEY = 0x57494E45


def apply_migrations(connection) -> list[str]:
    """Apply each migration once and return the newly applied migration names."""

    connection.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        "version TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW())"
    )
    applied: list[str] = []
    for migration_path in sorted(MIGRATIONS_DIR.glob("[0-9]*.sql")):
        with connection.transaction():
            connection.execute("SELECT pg_advisory_xact_lock(%s)", (MIGRATION_LOCK_KEY,))
            exists = connection.execute(
                "SELECT 1 FROM schema_migrations WHERE version = %s",
                (migration_path.name,),
            ).fetchone()
            if exists:
                continue
            connection.execute(migration_path.read_text(encoding="utf-8"))
            connection.execute(
                "INSERT INTO schema_migrations (version) VALUES (%s)",
                (migration_path.name,),
            )
            applied.append(migration_path.name)
    return applied
