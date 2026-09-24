from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from dinov2_retrieval.application.check_health import check_health
from dinov2_retrieval.config import Settings
from dinov2_retrieval.contracts import DatabaseInspection
from dinov2_retrieval.retrieval.reference_repository import RepositoryError

HEALTHY_DATABASE = DatabaseInspection(
    server_version="16.4",
    pgvector_version="0.8.0",
    table_exists=True,
    embedding_dimension=384,
    reference_count=5,
    model_reference_count=5,
)


class FakeEmbedder:
    model_name = "facebook/dinov2-small"
    device = "cpu"

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def embed(self, image: Image.Image) -> list[float]:
        return [0.0] * self.dimension


@pytest.fixture(autouse=True)
def configured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("DATABASE_URL", "postgresql://dinov2:secret@postgres:5432/dinov2")


def run_health(inspection=HEALTHY_DATABASE, embedder=None):
    def inspect(settings: Settings) -> DatabaseInspection:
        if isinstance(inspection, Exception):
            raise inspection
        return inspection

    def create(settings: Settings):
        if isinstance(embedder, Exception):
            raise embedder
        return embedder or FakeEmbedder()

    return check_health(inspect_database=inspect, create_embedder=create)


def test_everything_ok() -> None:
    report = run_health()

    assert report.status == "ok"
    assert list(report.checks) == [
        "config",
        "database",
        "pgvector",
        "reference_table",
        "model",
    ]
    config = report.checks["config"].details
    assert config["database_url"] == "postgresql://dinov2:***@postgres:5432/dinov2"
    assert config["dino_embedding_dimension"] == 384
    assert report.checks["reference_table"].details == {
        "embedding_dimension": 384,
        "reference_images": 5,
        "model_reference_images": 5,
    }
    assert report.checks["model"].details["embedding_dimension"] == 384


def test_invalid_config_skips_other_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL")

    report = run_health()

    assert report.status == "error"
    assert report.checks["config"].status == "error"
    assert "DATABASE_URL must be set" in report.checks["config"].message
    assert {report.checks[name].status for name in report.checks if name != "config"} == {
        "skipped"
    }


def test_unreachable_database_is_reported_and_model_still_checked() -> None:
    report = run_health(inspection=RepositoryError("Cannot connect to PostgreSQL"))

    assert report.status == "error"
    assert report.checks["database"].status == "error"
    assert "secret" not in str(report.model_dump())
    assert report.checks["pgvector"].status == "skipped"
    assert report.checks["reference_table"].status == "skipped"
    assert report.checks["model"].status == "ok"


@pytest.mark.parametrize(
    ("changes", "failed_check", "message"),
    [
        ({"pgvector_version": None}, "pgvector", "extension vector is not installed"),
        (
            {"table_exists": False, "embedding_dimension": None},
            "reference_table",
            "does not exist",
        ),
        ({"embedding_dimension": 768}, "reference_table", "dimension 768"),
    ],
)
def test_schema_problems(changes: dict, failed_check: str, message: str) -> None:
    report = run_health(inspection=HEALTHY_DATABASE.model_copy(update=changes))

    assert report.status == "error"
    assert report.checks[failed_check].status == "error"
    assert message in report.checks[failed_check].message


def test_model_problems() -> None:
    failed_load = run_health(embedder=ImportError("No module named 'torch'"))
    wrong_dimension = run_health(embedder=FakeEmbedder(dimension=768))

    assert failed_load.checks["model"].status == "error"
    assert "ImportError" in failed_load.checks["model"].message
    assert wrong_dimension.checks["model"].status == "error"
    assert "768" in wrong_dimension.checks["model"].message
