"""Manual integration tests against real PostgreSQL/pgvector and DINOv2.

Run inside Docker Compose (see README):

    docker compose up -d postgres
    docker compose run --rm dinov2-retrieval pytest -m integration -q

Test rows are removed afterwards; photos are generated in a temporary folder
because /data is mounted read-only.
"""

from __future__ import annotations

import math
import os
import uuid
from contextlib import closing
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

pytestmark = pytest.mark.integration

psycopg = pytest.importorskip("psycopg")
pytest.importorskip("pgvector")

from dinov2_retrieval.application.evaluate_retrieval import evaluate_retrieval
from dinov2_retrieval.application.index_reference_images import index_reference_images
from dinov2_retrieval.application.search_similar_wines import search_similar_wines
from dinov2_retrieval.contracts import (
    EvaluationQuery,
    ReferenceImage,
    ReferenceImageRecord,
    SearchRequest,
)
from dinov2_retrieval.infrastructure.database.connection import (
    inspect_database,
    open_connection,
)
from dinov2_retrieval.infrastructure.database.postgres_reference_repository import (
    PostgresReferenceRepository,
)
from dinov2_retrieval.infrastructure.storage.local_storage import LocalImageStorage


@pytest.fixture(scope="module", autouse=True)
def database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL is not set")
    return url


@pytest.fixture
def repository():
    with closing(PostgresReferenceRepository.connect()) as repository:
        yield repository


def unit_vector(index: int, dimension: int = 384) -> list[float]:
    values = [0.0] * dimension
    values[index] = 1.0
    return values


def delete_rows(where: str, value: str) -> None:
    with open_connection() as connection:
        connection.execute(f"DELETE FROM reference_images WHERE {where}", (value,))


def test_schema_matches_configuration() -> None:
    inspection = inspect_database("facebook/dinov2-small")

    assert inspection.pgvector_version
    assert inspection.table_exists
    assert inspection.embedding_dimension == 384


def test_upsert_count_and_cosine_search(repository: PostgresReferenceRepository) -> None:
    model_name = f"integration-test/{uuid.uuid4().hex}"
    records = [
        ReferenceImageRecord(
            wine_id=wine_id,
            slug=f"{wine_id}-slug",
            image_uri=f"/integration/{model_name}/{wine_id}/{photo}.jpg",
            model_name=model_name,
            embedding=unit_vector(axis),
        )
        for wine_id, photo, axis in [
            ("wine-a", "photo-1", 0),
            ("wine-a", "photo-2", 1),
            ("wine-b", "photo-1", 2),
        ]
    ]
    try:
        assert [repository.upsert_reference_embedding(r) for r in records] == [
            "inserted"
        ] * 3
        assert repository.upsert_reference_embedding(records[0]) == "updated"
        assert repository.get_reference_count(model_name) == 3

        query = unit_vector(0)
        query[2] = 0.5
        matches = repository.search_similar(query, limit=10, model_name=model_name)

        assert [m.image_uri for m in matches] == [
            records[0].image_uri,
            records[2].image_uri,
            records[1].image_uri,
        ]
        assert matches[0].distance == pytest.approx(1 - 1 / math.sqrt(1.25), abs=1e-6)
        assert matches[2].distance == pytest.approx(1.0, abs=1e-6)

        assert repository.delete_references_except(model_name, [records[0].image_uri]) == 2
        assert repository.get_reference_count(model_name) == 1
    finally:
        delete_rows("model_name = %s", model_name)


def test_index_and_search_with_real_model(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    from dinov2_retrieval.embedding.dinov2_embedder import DinoV2Embedder

    data_root = tmp_path.resolve()
    references = []
    for index, color in enumerate([(170, 20, 40), (30, 120, 60), (40, 60, 160)]):
        wine_id = f"it-wine-{index}"
        path = data_root / wine_id / "photo-1.jpg"
        path.parent.mkdir()
        image = Image.new("RGB", (320, 240), color=(235, 230, 220))
        ImageDraw.Draw(image).rectangle((60 + 20 * index, 40, 260, 200), fill=color)
        image.save(path)
        references.append(
            ReferenceImage(wine_id=wine_id, slug=f"{wine_id}-slug", image_uri=str(path))
        )

    storage = LocalImageStorage(data_root=data_root)
    embedder = DinoV2Embedder()
    try:
        with closing(PostgresReferenceRepository.connect()) as repository:
            first = index_reference_images(
                references, repository, storage=storage, embedder=embedder
            )
            second = index_reference_images(
                references, repository, storage=storage, embedder=embedder
            )
            response = search_similar_wines(
                SearchRequest(request_id="integration", image_uri=references[1].image_uri),
                repository,
                storage=storage,
                embedder=embedder,
            )
            reference_count = repository.get_reference_count(embedder.model_name)
            evaluation = evaluate_retrieval(
                [
                    EvaluationQuery(query_image_uri=r.image_uri, wine_id=r.wine_id)
                    for r in references
                ],
                repository,
                top_k=1,
                storage=storage,
                embedder=embedder,
            )
            reference_count_after = repository.get_reference_count(embedder.model_name)
    finally:
        delete_rows("image_uri LIKE %s", f"{data_root}/%")

    assert (first.status, first.inserted, first.updated) == ("ok", 3, 0)
    assert (second.inserted, second.updated) == (0, 3)
    assert response.query_embedding_dimension == 384
    assert response.candidates[0].wine_id == "it-wine-1"
    assert response.candidates[0].score == pytest.approx(1.0, abs=1e-4)
    assert (evaluation.status, evaluation.queries_processed) == ("ok", 3)
    assert evaluation.recall_at_k == 1.0
    assert reference_count_after == reference_count
