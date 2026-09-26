from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from PIL import Image

from dinov2_retrieval.application.search_similar_wines import (
    search_similar_wines,
    select_top_wines,
)
from dinov2_retrieval.contracts import ReferenceMatch, SearchRequest
from dinov2_retrieval.infrastructure.storage.local_storage import (
    ImageNotFoundError,
    LocalImageStorage,
)


class FakeEmbedder:
    model_name = "fake/model"
    device = "cpu"

    def embed(self, image: Image.Image) -> list[float]:
        return [0.6, 0.8, 0.0]


class FakeRetriever:
    def __init__(self, matches: list[ReferenceMatch]) -> None:
        self.matches = matches
        self.calls: list[tuple[list[float], int, str]] = []

    def search_similar(
        self, query_embedding: Sequence[float], limit: int, model_name: str
    ) -> list[ReferenceMatch]:
        self.calls.append((list(query_embedding), limit, model_name))
        return self.matches[:limit]


def match(wine_id: str, photo: str, distance: float) -> ReferenceMatch:
    return ReferenceMatch(
        wine_id=wine_id,
        slug=f"{wine_id}-slug",
        image_uri=f"/data/reference/{wine_id}/{photo}.jpg",
        distance=distance,
    )


@pytest.fixture
def query_image(tmp_path: Path) -> Path:
    path = tmp_path / "queries" / "test.jpg"
    path.parent.mkdir()
    Image.new("RGB", (16, 8), color=(10, 20, 30)).save(path)
    return path


def run_search(
    data_root: Path, retriever: FakeRetriever, top_k: int = 20, raw_limit: int | None = None
):
    return search_similar_wines(
        SearchRequest(request_id="request-001", image_uri="queries/test.jpg", top_k=top_k),
        retriever,
        storage=LocalImageStorage(data_root=data_root),
        embedder=FakeEmbedder(),
        raw_limit=raw_limit,
    )


def test_groups_photos_by_wine_and_keeps_the_best_one() -> None:
    matches = [
        match("wine-001", "photo-1", 0.30),
        match("wine-002", "photo-1", 0.10),
        match("wine-001", "photo-2", 0.05),
        match("wine-003", "photo-1", 0.20),
        match("wine-002", "photo-2", 0.40),
    ]

    candidates = select_top_wines(matches, top_k=20)

    assert [(c.wine_id, c.best_image_uri) for c in candidates] == [
        ("wine-001", "/data/reference/wine-001/photo-2.jpg"),
        ("wine-002", "/data/reference/wine-002/photo-1.jpg"),
        ("wine-003", "/data/reference/wine-003/photo-1.jpg"),
    ]
    assert [c.distance for c in candidates] == [0.05, 0.10, 0.20]
    assert [c.score for c in candidates] == pytest.approx([0.95, 0.90, 0.80])
    assert candidates[0].slug == "wine-001-slug"


def test_result_is_limited_to_top_k_distinct_wines() -> None:
    matches = [
        match(f"wine-{index:03d}", f"photo-{photo}", index / 100 + photo / 1000)
        for index in range(30)
        for photo in range(3)
    ]

    candidates = select_top_wines(matches, top_k=20)

    assert len(candidates) == 20
    assert len({c.wine_id for c in candidates}) == 20
    assert [c.wine_id for c in candidates] == [f"wine-{index:03d}" for index in range(20)]
    assert all(c.best_image_uri.endswith("photo-0.jpg") for c in candidates)


def test_equal_distances_are_ordered_by_wine_id() -> None:
    matches = [match("wine-b", "photo", 0.1), match("wine-a", "photo", 0.1)]

    assert [c.wine_id for c in select_top_wines(matches, top_k=5)] == ["wine-a", "wine-b"]


def test_search_embeds_query_and_returns_response(query_image: Path) -> None:
    data_root = query_image.parent.parent
    retriever = FakeRetriever(
        [match("wine-001", "photo-1", 0.06), match("wine-002", "photo-1", 0.25)]
    )

    response = run_search(data_root, retriever, top_k=20)

    assert response.request_id == "request-001"
    assert response.status == "ok"
    assert response.model_name == "fake/model"
    assert response.query_embedding_dimension == 3
    assert [c.wine_id for c in response.candidates] == ["wine-001", "wine-002"]
    assert response.candidates[0].score == pytest.approx(0.94)
    assert response.message is None
    assert retriever.calls == [([0.6, 0.8, 0.0], 100, "fake/model")]


def test_raw_limit_is_read_from_config_and_never_below_top_k(
    query_image: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = query_image.parent.parent
    monkeypatch.setenv("RAW_RETRIEVAL_LIMIT", "30")
    retriever = FakeRetriever([])

    run_search(data_root, retriever, top_k=5)
    run_search(data_root, retriever, top_k=50)

    assert [limit for _, limit, _ in retriever.calls] == [30, 50]


def test_limit_grows_until_top_k_wines_are_found(query_image: Path) -> None:
    # 30 wines with 6 photos each: the first 100 photos cover only 17 wines
    matches = [
        match(f"wine-{wine:03d}", f"photo-{photo}", wine / 100 + photo / 10000)
        for wine in range(30)
        for photo in range(6)
    ]
    retriever = FakeRetriever(matches)

    response = run_search(query_image.parent.parent, retriever, top_k=20)

    assert len(response.candidates) == 20
    assert response.candidates[-1].wine_id == "wine-019"
    assert [limit for _, limit, _ in retriever.calls] == [100, 200]


def test_limit_stops_growing_when_index_is_exhausted(query_image: Path) -> None:
    matches = [
        match(f"wine-{wine:03d}", f"photo-{photo}", wine / 100 + photo / 10000)
        for wine in range(3)
        for photo in range(50)
    ]
    retriever = FakeRetriever(matches)

    response = run_search(query_image.parent.parent, retriever, top_k=20)

    assert len(response.candidates) == 3
    assert [limit for _, limit, _ in retriever.calls] == [100, 200]


def test_empty_index_gives_no_results_status(query_image: Path) -> None:
    response = run_search(query_image.parent.parent, FakeRetriever([]))

    assert response.status == "no_results"
    assert response.candidates == []
    assert "run the index command" in response.message


def test_invalid_query_photo_is_rejected_before_search(tmp_path: Path) -> None:
    retriever = FakeRetriever([])

    with pytest.raises(ImageNotFoundError):
        run_search(tmp_path, retriever)
    assert retriever.calls == []
