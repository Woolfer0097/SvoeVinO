from __future__ import annotations

from pathlib import Path

from PIL import Image

from dinov2_retrieval.application.create_embedding import create_embedding
from dinov2_retrieval.application.validate_image import validate_image
from dinov2_retrieval.contracts import RetrievalRequest, ValidatedImage


class FakeImageStorage:
    def __init__(self, result: ValidatedImage) -> None:
        self.result = result
        self.received_uri: str | None = None

    def validate(self, image_uri: str) -> ValidatedImage:
        self.received_uri = image_uri
        return self.result


class FakeEmbedder:
    model_name = "fake/model"
    device = "cpu"

    def embed(self, image: Image.Image) -> list[float]:
        assert image.mode == "RGB"
        return [0.5, 0.5]


def test_validate_image_coordinates_storage() -> None:
    request = RetrievalRequest(
        job_id="job-1",
        query_id="query-1",
        image_uri="/data/queries/test.jpeg",
    )
    expected = ValidatedImage(
        path="/data/queries/test.jpeg",
        width=256,
        height=192,
        mime_type="image/jpeg",
    )
    storage = FakeImageStorage(expected)

    result = validate_image(request, storage=storage)

    assert result == expected
    assert storage.received_uri == request.image_uri


def test_create_embedding_returns_full_embedding_and_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    image_path = tmp_path / "test.jpg"
    Image.new("RGB", (8, 4), color=(20, 40, 60)).save(image_path)

    result = create_embedding(str(image_path), embedder=FakeEmbedder())

    assert result.path == str(image_path.resolve())
    assert result.model == "fake/model"
    assert result.dimension == 2
    assert result.device == "cpu"
    assert result.embedding == [0.5, 0.5]
