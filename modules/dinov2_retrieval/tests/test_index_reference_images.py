from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from PIL import Image

from dinov2_retrieval.application.index_reference_images import index_reference_images
from dinov2_retrieval.contracts import ReferenceImage, ReferenceImageRecord, ReferenceMatch
from dinov2_retrieval.infrastructure.storage.local_storage import LocalImageStorage
from dinov2_retrieval.retrieval.reference_repository import RepositoryError


class FakeEmbedder:
    model_name = "fake/model"
    device = "cpu"

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, image: Image.Image) -> list[float]:
        assert image.mode == "RGB"
        self.calls += 1
        red, green, blue = image.getpixel((0, 0))
        return [red / 255, green / 255, blue / 255]


class FakeRepository:
    """In-memory stand-in for PostgresReferenceRepository."""

    def __init__(self) -> None:
        self.records: dict[str, ReferenceImageRecord] = {}

    def upsert_reference_embedding(self, record: ReferenceImageRecord) -> str:
        outcome = "updated" if record.image_uri in self.records else "inserted"
        self.records[record.image_uri] = record
        return outcome

    def get_reference_count(self, model_name: str | None = None) -> int:
        return sum(
            1
            for record in self.records.values()
            if model_name is None or record.model_name == model_name
        )

    def search_similar(
        self, query_embedding: Sequence[float], limit: int, model_name: str
    ) -> list[ReferenceMatch]:
        raise AssertionError("indexing must not search")

    def close(self) -> None:
        pass


class PruningRepository(FakeRepository):
    def __init__(self, stored_uris: list[str]) -> None:
        super().__init__()
        for uri in stored_uris:
            self.records[uri] = ReferenceImageRecord(
                wine_id="old", slug="old", image_uri=uri, model_name="fake/model", embedding=[0.0]
            )
        self.keep_calls: list[set[str]] = []

    def delete_references_except(self, model_name: str, keep_image_uris) -> int:
        assert model_name == "fake/model"
        keep = set(keep_image_uris)
        self.keep_calls.append(keep)
        stale = [uri for uri in self.records if uri not in keep]
        for uri in stale:
            del self.records[uri]
        return len(stale)


class BrokenRepository(FakeRepository):
    def upsert_reference_embedding(self, record: ReferenceImageRecord) -> str:
        raise RepositoryError("connection lost")


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    for relative, color in [
        ("reference/wine-001/photo-1.jpg", (200, 0, 0)),
        ("reference/wine-001/photo-2.jpg", (180, 20, 0)),
        ("reference/wine-002/photo-1.png", (0, 0, 200)),
    ]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 8), color=color).save(path)
    (tmp_path / "reference/broken.jpg").write_bytes(b"not an image")
    return tmp_path.resolve()


def reference(data_root: Path, wine_id: str, relative: str) -> ReferenceImage:
    return ReferenceImage(
        wine_id=wine_id, slug=f"{wine_id}-slug", image_uri=str(data_root / relative)
    )


def valid_references(data_root: Path) -> list[ReferenceImage]:
    return [
        reference(data_root, "wine-001", "reference/wine-001/photo-1.jpg"),
        reference(data_root, "wine-001", "reference/wine-001/photo-2.jpg"),
        reference(data_root, "wine-002", "reference/wine-002/photo-1.png"),
    ]


def run_index(
    data_root: Path,
    references: list[ReferenceImage],
    repository: FakeRepository,
    embedder: FakeEmbedder,
    prune: bool = False,
):
    return index_reference_images(
        references,
        repository,
        prune=prune,
        storage=LocalImageStorage(data_root=data_root),
        embedder=embedder,
    )


def test_indexes_every_photo_with_one_embedder(data_root: Path) -> None:
    repository = FakeRepository()
    embedder = FakeEmbedder()

    stats = run_index(data_root, valid_references(data_root), repository, embedder)

    assert stats.status == "ok"
    assert (stats.total, stats.processed, stats.inserted, stats.updated) == (3, 3, 3, 0)
    assert (stats.skipped, stats.errors, stats.error_details) == (0, 0, [])
    assert stats.model_name == "fake/model"
    assert stats.reference_count == 3
    assert embedder.calls == 3

    record = repository.records[str(data_root / "reference/wine-002/photo-1.png")]
    assert record.wine_id == "wine-002"
    assert record.slug == "wine-002-slug"
    assert record.model_name == "fake/model"
    assert record.embedding == [0.0, 0.0, pytest.approx(200 / 255)]


def test_rerun_updates_instead_of_duplicating(data_root: Path) -> None:
    repository = FakeRepository()
    run_index(data_root, valid_references(data_root), repository, FakeEmbedder())

    stats = run_index(data_root, valid_references(data_root), repository, FakeEmbedder())

    assert (stats.inserted, stats.updated, stats.processed) == (0, 3, 3)
    assert len(repository.records) == 3
    assert stats.reference_count == 3


def test_broken_photos_do_not_stop_indexing(data_root: Path) -> None:
    references = [
        reference(data_root, "wine-003", "reference/missing.jpg"),
        reference(data_root, "wine-004", "reference/broken.jpg"),
        *valid_references(data_root),
    ]
    repository = FakeRepository()

    stats = run_index(data_root, references, repository, FakeEmbedder())

    assert stats.status == "partial"
    assert (stats.total, stats.processed, stats.errors) == (5, 3, 2)
    assert [(error.wine_id, error.error_type) for error in stats.error_details] == [
        ("wine-003", "ImageNotFoundError"),
        ("wine-004", "CorruptedImageError"),
    ]
    assert stats.error_details[0].image_uri == str(data_root / "reference/missing.jpg")
    assert len(repository.records) == 3


def test_run_without_a_single_valid_photo_fails(data_root: Path) -> None:
    references = [reference(data_root, "wine-004", "reference/broken.jpg")]

    stats = run_index(data_root, references, FakeRepository(), FakeEmbedder())

    assert stats.status == "failed"
    assert (stats.processed, stats.errors, stats.reference_count) == (0, 1, 0)


def test_one_file_listed_twice_is_indexed_once(data_root: Path) -> None:
    references = [
        reference(data_root, "wine-001", "reference/wine-001/photo-1.jpg"),
        ReferenceImage(
            wine_id="wine-001",
            slug="wine-001-slug",
            image_uri="reference/wine-001/../wine-001/photo-1.jpg",
        ),
    ]
    repository = FakeRepository()

    stats = run_index(data_root, references, repository, FakeEmbedder())

    assert (stats.processed, stats.skipped, stats.errors) == (1, 1, 0)
    assert list(repository.records) == [str(data_root / "reference/wine-001/photo-1.jpg")]


def test_repository_errors_stop_indexing(data_root: Path) -> None:
    with pytest.raises(RepositoryError, match="connection lost"):
        run_index(
            data_root, valid_references(data_root), BrokenRepository(), FakeEmbedder()
        )


def test_prune_deletes_stored_photos_missing_from_the_list(data_root: Path) -> None:
    stale = str(data_root / "reference/old-wine/photo.jpg")
    kept = str(data_root / "reference/wine-001/photo-1.jpg")
    repository = PruningRepository([stale, kept])

    stats = run_index(
        data_root, valid_references(data_root), repository, FakeEmbedder(), prune=True
    )

    assert (stats.deleted, stats.inserted, stats.updated) == (1, 2, 1)
    assert stale not in repository.records
    assert stats.reference_count == 3


def test_prune_keeps_photos_that_failed_in_this_run(data_root: Path) -> None:
    broken = str(data_root / "reference/broken.jpg")
    repository = PruningRepository([broken])
    references = [
        reference(data_root, "wine-004", "reference/broken.jpg"),
        *valid_references(data_root),
    ]

    stats = run_index(data_root, references, repository, FakeEmbedder(), prune=True)

    assert (stats.status, stats.deleted) == ("partial", 0)
    assert broken in repository.records


def test_prune_is_skipped_when_nothing_was_processed(data_root: Path) -> None:
    repository = PruningRepository([str(data_root / "reference/wine-001/photo-1.jpg")])
    references = [reference(data_root, "wine-003", "reference/missing.jpg")]

    stats = run_index(data_root, references, repository, FakeEmbedder(), prune=True)

    assert (stats.status, stats.deleted) == ("failed", 0)
    assert repository.keep_calls == []
    assert len(repository.records) == 1


def test_without_prune_nothing_is_deleted(data_root: Path) -> None:
    repository = PruningRepository([str(data_root / "reference/old-wine/photo.jpg")])

    stats = run_index(data_root, valid_references(data_root), repository, FakeEmbedder())

    assert stats.deleted == 0
    assert repository.keep_calls == []
