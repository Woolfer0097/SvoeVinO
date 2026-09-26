from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from PIL import Image

from dinov2_retrieval.application.evaluate_retrieval import (
    NOT_IN_TOP_K,
    compute_recall,
    evaluate_retrieval,
)
from dinov2_retrieval.contracts import EvaluationQuery, ReferenceImageRecord, ReferenceMatch
from dinov2_retrieval.infrastructure.evaluation_manifest import (
    ManifestError,
    read_evaluation_queries,
)
from dinov2_retrieval.infrastructure.storage.local_storage import LocalImageStorage
from dinov2_retrieval.retrieval.reference_repository import RepositoryError

# --- queries.csv ---------------------------------------------------------------

VALID_QUERIES = """query_image_uri,wine_id
/data/evaluation/images/query-001.jpg,wine-001
/data/evaluation/images/query-002.jpg,wine-002
"""


def write_queries(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "queries.csv"
    path.write_text(content, encoding="utf-8")
    return path


def read(path: Path) -> list[EvaluationQuery]:
    return read_evaluation_queries(path, data_root="/data")


def test_reads_evaluation_queries(tmp_path: Path) -> None:
    assert read(write_queries(tmp_path, VALID_QUERIES)) == [
        EvaluationQuery(
            query_image_uri="/data/evaluation/images/query-001.jpg", wine_id="wine-001"
        ),
        EvaluationQuery(
            query_image_uri="/data/evaluation/images/query-002.jpg", wine_id="wine-002"
        ),
    ]


def test_extra_columns_and_blank_rows_are_ignored(tmp_path: Path) -> None:
    content = "note,wine_id,query_image_uri\nfront,wine-001,/data/q.jpg\n\n"

    assert read(write_queries(tmp_path, content)) == [
        EvaluationQuery(query_image_uri="/data/q.jpg", wine_id="wine-001")
    ]


@pytest.mark.parametrize(
    ("header", "missing"),
    [
        ("image_uri,wine_id", "query_image_uri"),
        ("query_image_uri,wine", "wine_id"),
        ("path;wine_id", "query_image_uri, wine_id"),
    ],
)
def test_required_columns(tmp_path: Path, header: str, missing: str) -> None:
    path = write_queries(tmp_path, f"{header}\n/data/q.jpg,wine-001\n")

    with pytest.raises(ManifestError, match=f"missing required columns: {missing}"):
        read(path)


def test_row_problems_are_reported_together(tmp_path: Path) -> None:
    content = (
        "query_image_uri,wine_id\n"
        "/data/q1.jpg,\n"
        "evaluation/images/q2.jpg,wine-002\n"
        "/data/q3.jpg,wine-003\n"
        "/data/q3.jpg,wine-004\n"
        "/home/user/q5.jpg,wine-005\n"
    )

    with pytest.raises(ManifestError) as error:
        read(write_queries(tmp_path, content))

    message = str(error.value)
    assert "line 2: empty wine_id" in message
    assert "line 3: query_image_uri path must be absolute" in message
    assert "line 5: duplicate query_image_uri /data/q3.jpg (first on line 4)" in message
    assert "line 6: query_image_uri path must be inside DATA_ROOT /data" in message


@pytest.mark.parametrize(
    ("content", "message"),
    [("", "Queries file is empty"), ("query_image_uri,wine_id\n", "has no queries")],
)
def test_queries_file_without_queries(tmp_path: Path, content: str, message: str) -> None:
    with pytest.raises(ManifestError, match=message):
        read(write_queries(tmp_path, content))


def test_missing_queries_file(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="Queries file does not exist"):
        read(tmp_path / "missing.csv")


# --- Recall@K -------------------------------------------------------------------


def test_recall_at_1_5_and_20() -> None:
    ranks = [1, 1, 3, 5, 6, 20, 21, None]

    recall_at, correct_at = compute_recall(ranks, [1, 5, 20])

    assert correct_at == {1: 2, 5: 4, 20: 6}
    assert recall_at == {1: 0.25, 5: 0.5, 20: 0.75}


def test_recall_is_rounded_and_none_without_queries() -> None:
    assert compute_recall([1, None, None], [1])[0] == {1: 0.3333}
    assert compute_recall([], [1, 5]) == ({1: None, 5: None}, {1: 0, 5: 0})


# --- evaluation -----------------------------------------------------------------


class ColorEmbedder:
    """Embeds the red channel of the first pixel; counts calls."""

    model_name = "fake/model"
    device = "cpu"

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, image: Image.Image) -> list[float]:
        assert image.mode == "RGB"
        self.calls += 1
        return [float(image.getpixel((0, 0))[0])]


class ScriptedRepository:
    """Returns prepared nearest reference photos for each query color."""

    def __init__(self, results: dict[int, list[ReferenceMatch]]) -> None:
        self.results = results
        self.limits: list[int] = []
        self.upserts: list[ReferenceImageRecord] = []

    def get_wine_ids(self, model_name: str) -> set[str]:
        assert model_name == "fake/model"
        return {
            match.wine_id for matches in self.results.values() for match in matches
        }

    def search_similar(
        self, query_embedding: Sequence[float], limit: int, model_name: str
    ) -> list[ReferenceMatch]:
        self.limits.append(limit)
        return self.results[int(query_embedding[0])][:limit]

    def upsert_reference_embedding(self, record: ReferenceImageRecord) -> str:
        self.upserts.append(record)
        return "inserted"

    def get_reference_count(self, model_name: str | None = None) -> int:
        return 0

    def close(self) -> None:
        pass


def photo(wine_id: str, number: int, distance: float) -> ReferenceMatch:
    return ReferenceMatch(
        wine_id=wine_id,
        slug=f"{wine_id}-slug",
        image_uri=f"/data/reference/{wine_id}/photo-{number}.jpg",
        distance=distance,
    )


def ranking(*wine_ids: str) -> list[ReferenceMatch]:
    return [photo(wine_id, 1, 0.1 + index / 100) for index, wine_id in enumerate(wine_ids)]


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    images = tmp_path / "evaluation" / "images"
    images.mkdir(parents=True)
    for red in (10, 20, 30):
        Image.new("RGB", (16, 8), color=(red, 0, 0)).save(images / f"query-{red}.jpg")
    (images / "broken.jpg").write_bytes(b"not an image")
    return tmp_path.resolve()


def query(data_root: Path, name: str, wine_id: str) -> EvaluationQuery:
    return EvaluationQuery(
        query_image_uri=str(data_root / "evaluation" / "images" / name), wine_id=wine_id
    )


def run(data_root, queries, repository, embedder=None, top_k=20):
    return evaluate_retrieval(
        queries,
        repository,
        top_k=top_k,
        storage=LocalImageStorage(data_root=data_root),
        embedder=embedder or ColorEmbedder(),
    )


WINES = [f"wine-{index:03d}" for index in range(1, 31)]


def test_correct_wine_found(data_root: Path) -> None:
    repository = ScriptedRepository({10: ranking("wine-001", "wine-002")})

    report = run(data_root, [query(data_root, "query-10.jpg", "wine-001")], repository)

    assert report.status == "ok"
    assert (report.queries_total, report.queries_processed, report.queries_with_errors) == (1, 1, 0)
    assert (report.recall_at_k, report.correct_queries) == (1.0, 1)
    assert report.recall_at == {1: 1.0, 5: 1.0, 20: 1.0}
    assert report.incorrect_queries == []
    assert report.errors == []


def test_correct_wine_not_found(data_root: Path) -> None:
    others = [wine for wine in WINES if wine != "wine-001"]
    # wine-001 is indexed (key 20) but not among the wines found for query 10
    repository = ScriptedRepository(
        {10: ranking(*others), 20: [photo("wine-001", 1, 0.9)]}
    )

    report = run(data_root, [query(data_root, "query-10.jpg", "wine-001")], repository)

    assert report.status == "ok"
    assert (report.recall_at_k, report.correct_queries) == (0.0, 0)
    [incorrect] = report.incorrect_queries
    assert incorrect.query_image_uri.endswith("query-10.jpg")
    assert incorrect.expected_wine_id == "wine-001"
    assert incorrect.predicted_wine_ids == others[:20]
    assert incorrect.expected_rank is None
    assert incorrect.reason == NOT_IN_TOP_K
    assert "embedding" not in incorrect.model_dump()


def test_recall_levels_and_top_k(data_root: Path) -> None:
    # correct wine at rank 1, 3 and 12 for three queries
    repository = ScriptedRepository(
        {
            10: ranking("wine-001", *WINES[5:25]),
            20: ranking("wine-005", "wine-006", "wine-002", *WINES[10:25]),
            30: ranking(*WINES[3:14], "wine-003", *WINES[14:25]),
        }
    )
    queries = [
        query(data_root, "query-10.jpg", "wine-001"),
        query(data_root, "query-20.jpg", "wine-002"),
        query(data_root, "query-30.jpg", "wine-003"),
    ]

    report = run(data_root, queries, repository, top_k=10)

    assert report.top_k == 10
    assert report.recall_at == {1: 0.3333, 5: 0.6667, 10: 0.6667, 20: 1.0}
    assert report.correct_at == {1: 1, 5: 2, 10: 2, 20: 3}
    assert (report.recall_at_k, report.correct_queries) == (0.6667, 2)
    assert repository.limits == [100, 100, 100]
    [incorrect] = report.incorrect_queries
    assert (incorrect.expected_wine_id, incorrect.expected_rank) == ("wine-003", 12)
    assert len(incorrect.predicted_wine_ids) == 10


def test_several_photos_of_one_wine_take_one_rank(data_root: Path) -> None:
    repository = ScriptedRepository(
        {
            10: [
                photo("wine-002", 1, 0.10),
                photo("wine-002", 2, 0.11),
                photo("wine-002", 3, 0.12),
                photo("wine-003", 1, 0.13),
                photo("wine-003", 2, 0.14),
                photo("wine-001", 1, 0.15),
            ]
        }
    )

    report = run(
        data_root, [query(data_root, "query-10.jpg", "wine-001")], repository, top_k=3
    )

    assert report.recall_at[1] == 0.0
    assert report.recall_at_k == 1.0
    assert report.correct_at == {1: 0, 3: 1, 5: 1, 20: 1}


def test_broken_images_do_not_stop_evaluation(data_root: Path) -> None:
    repository = ScriptedRepository({10: ranking("wine-001"), 20: ranking("wine-002")})
    queries = [
        query(data_root, "broken.jpg", "wine-001"),
        query(data_root, "missing.jpg", "wine-002"),
        query(data_root, "query-10.jpg", "wine-001"),
        query(data_root, "query-20.jpg", "wine-002"),
    ]

    report = run(data_root, queries, repository)

    assert report.status == "partial"
    assert (report.queries_total, report.queries_processed, report.queries_with_errors) == (4, 2, 2)
    assert (report.recall_at_k, report.correct_queries) == (1.0, 2)
    assert [(e.expected_wine_id, e.error_type) for e in report.errors] == [
        ("wine-001", "CorruptedImageError"),
        ("wine-002", "ImageNotFoundError"),
    ]
    assert report.errors[0].query_image_uri.endswith("broken.jpg")


def test_wine_without_reference_photos_is_an_error(data_root: Path) -> None:
    repository = ScriptedRepository({10: ranking("wine-001")})
    embedder = ColorEmbedder()

    report = run(
        data_root, [query(data_root, "query-10.jpg", "wine-999")], repository, embedder
    )

    assert report.status == "failed"
    assert report.recall_at_k is None
    assert report.errors[0].error_type == "WineNotIndexedError"
    assert "wine_id wine-999 has no reference photos" in report.errors[0].reason
    assert embedder.calls == 0


def test_empty_index(data_root: Path) -> None:
    report = run(
        data_root, [query(data_root, "query-10.jpg", "wine-001")], ScriptedRepository({})
    )

    assert report.status == "failed"
    assert report.indexed_wines == 0
    assert "run the index command" in report.message


def test_one_embedder_for_all_queries_and_nothing_is_stored(data_root: Path) -> None:
    repository = ScriptedRepository(
        {10: ranking("wine-001"), 20: ranking("wine-002"), 30: ranking("wine-003")}
    )
    embedder = ColorEmbedder()
    queries = [
        query(data_root, f"query-{red}.jpg", wine_id)
        for red, wine_id in [(10, "wine-001"), (20, "wine-002"), (30, "wine-003")]
    ]

    run(data_root, queries, repository, embedder)

    assert embedder.calls == 3
    assert repository.upserts == []


def test_database_errors_stop_evaluation(data_root: Path) -> None:
    class BrokenRepository(ScriptedRepository):
        def search_similar(self, query_embedding, limit, model_name):
            raise RepositoryError("connection lost")

    with pytest.raises(RepositoryError, match="connection lost"):
        run(
            data_root,
            [query(data_root, "query-10.jpg", "wine-001")],
            BrokenRepository({10: ranking("wine-001")}),
        )
