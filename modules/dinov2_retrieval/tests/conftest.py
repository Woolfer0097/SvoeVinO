from __future__ import annotations

import pytest

# Settings the Docker image passes in; unit tests must not depend on them.
MODULE_ENV_VARS = (
    "DATA_ROOT",
    "MAX_IMAGE_SIZE_BYTES",
    "SUPPORTED_IMAGE_EXTENSIONS",
    "DATABASE_URL",
    "DINO_MODEL_NAME",
    "DINO_EMBEDDING_DIMENSION",
    "DEFAULT_TOP_K",
    "RAW_RETRIEVAL_LIMIT",
)


@pytest.fixture(autouse=True)
def isolated_environment(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    if request.node.get_closest_marker("integration"):
        return
    for name in MODULE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
