from __future__ import annotations

from pathlib import Path

import pytest

from dinov2_retrieval.config import (
    ConfigurationError,
    get_database_url,
    get_default_top_k,
    get_dino_embedding_dimension,
    get_dino_model_name,
    get_raw_retrieval_limit,
    load_settings,
    redact_database_url,
)

DATABASE_URL = "postgresql://dinov2:secret@postgres:5432/dinov2"


def test_retrieval_settings_have_defaults() -> None:
    assert get_dino_model_name() == "facebook/dinov2-with-registers-giant"
    assert get_dino_embedding_dimension() == 1536
    assert get_default_top_k() == 20
    assert get_raw_retrieval_limit() == 100


def test_retrieval_settings_are_read_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DINO_MODEL_NAME", "facebook/dinov2-base")
    monkeypatch.setenv("DINO_EMBEDDING_DIMENSION", "768")
    monkeypatch.setenv("DEFAULT_TOP_K", "5")
    monkeypatch.setenv("RAW_RETRIEVAL_LIMIT", "50")

    assert get_dino_model_name() == "facebook/dinov2-base"
    assert get_dino_embedding_dimension() == 768
    assert get_default_top_k() == 5
    assert get_raw_retrieval_limit() == 50


@pytest.mark.parametrize(
    "name", ["DINO_EMBEDDING_DIMENSION", "DEFAULT_TOP_K", "RAW_RETRIEVAL_LIMIT"]
)
@pytest.mark.parametrize("value", ["0", "-3", "twenty"])
def test_integer_settings_must_be_positive(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ConfigurationError, match=f"{name} must be a positive integer"):
        load_settings_value(name)


def load_settings_value(name: str) -> int:
    getters = {
        "DINO_EMBEDDING_DIMENSION": get_dino_embedding_dimension,
        "DEFAULT_TOP_K": get_default_top_k,
        "RAW_RETRIEVAL_LIMIT": get_raw_retrieval_limit,
    }
    return getters[name]()


def test_database_url_is_required() -> None:
    with pytest.raises(ConfigurationError, match="DATABASE_URL must be set"):
        get_database_url()


def test_database_url_must_be_postgresql(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "mysql://user:pass@host/db")

    with pytest.raises(ConfigurationError, match="postgresql://"):
        get_database_url()


def test_redact_database_url_hides_only_the_password() -> None:
    assert (
        redact_database_url(DATABASE_URL)
        == "postgresql://dinov2:***@postgres:5432/dinov2"
    )
    assert redact_database_url("postgresql://postgres/db") == "postgresql://postgres/db"


def test_load_settings_reads_everything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("DATABASE_URL", DATABASE_URL)

    settings = load_settings()

    assert settings.data_root == tmp_path.resolve()
    assert settings.database_url == DATABASE_URL
    assert settings.dino_model_name == "facebook/dinov2-with-registers-giant"
    assert settings.dino_embedding_dimension == 1536
    assert settings.default_top_k == 20
    assert settings.raw_retrieval_limit == 100
