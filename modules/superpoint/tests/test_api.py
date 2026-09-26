from __future__ import annotations

import io
import tempfile
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from superpoint.contracts import MatchPrediction
from superpoint.entrypoints.api import (
    create_app,
    max_upload_body_bytes,
    parse_candidate_ids,
)
from superpoint.infrastructure.storage.local_storage import LocalImageStorage


class FakeMatcher:
    model_name = "fake-matcher"
    device = "cpu"

    def __init__(self, prediction: MatchPrediction) -> None:
        self.prediction = prediction

    def match(self, query_image: Image.Image, reference_image: Image.Image) -> MatchPrediction:
        assert query_image.mode == "RGB"
        assert reference_image.mode == "RGB"
        return self.prediction


class FakeGeometry:
    def count_inliers(self, query_points, reference_points) -> int:
        return len(query_points)


def prediction() -> MatchPrediction:
    return MatchPrediction(
        query_points=[(1.0, 2.0), (3.0, 4.0)],
        reference_points=[(5.0, 6.0), (7.0, 8.0)],
        scores=[0.9, 0.7],
        num_keypoints_query=4,
        num_keypoints_reference=5,
    )


def photo_bytes(image_format: str = "JPEG") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 16), color=(120, 20, 60)).save(buffer, format=image_format)
    return buffer.getvalue()


def upload_dirs() -> set[str]:
    return {path.name for path in Path(tempfile.gettempdir()).glob("superpoint-upload-*")}


@pytest.fixture
def client() -> TestClient:
    return TestClient(
        create_app(matcher=FakeMatcher(prediction()), geometry=FakeGeometry())
    )


def test_health(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model": "fake-matcher",
        "device": "cpu",
    }


def test_root_opens_swagger(client: TestClient) -> None:
    response = client.get("/", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


def test_swagger_is_grouped_and_ready_to_try(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    tags = {
        path: next(iter(operation.values()))["tags"][0]
        for path, operation in schema["paths"].items()
    }

    assert schema["info"]["title"] == "SuperPoint — проверка кандидатов"
    assert [tag["name"] for tag in schema["tags"]] == ["Проверка", "Служебное"]
    assert tags == {
        "/verify": "Проверка",
        "/health": "Служебное",
    }
    docs = client.get("/docs").text
    assert '"tryItOutEnabled": true' in docs
    assert '"displayRequestDuration": true' in docs


class ScriptedGeometry:
    def __init__(self, inliers: list[int]) -> None:
        self.inliers = list(inliers)

    def count_inliers(self, query_points, reference_points) -> int:
        return self.inliers.pop(0)


class FakeCatalog:
    def __init__(self, photos: dict[str, list[str]]) -> None:
        self.photos = photos
        self.calls: list[list[str]] = []

    def image_uris(self, wine_ids):
        self.calls.append(list(wine_ids))
        return {
            wine_id: self.photos[wine_id]
            for wine_id in wine_ids
            if wine_id in self.photos
        }


def reference_tree(tmp_path: Path) -> tuple[Path, dict[str, list[str]]]:
    root = tmp_path / "data"
    folder = root / "reference"
    folder.mkdir(parents=True)
    photos: dict[str, list[str]] = {}
    for wine_id, name in (("low", "low.jpg"), ("high", "high.jpg"), ("extra", "extra.jpg")):
        path = folder / name
        path.write_bytes(photo_bytes())
        photos[wine_id] = [str(path)]
    return root, photos


def ranking_client(tmp_path: Path, inliers: list[int]) -> tuple[TestClient, FakeCatalog]:
    root, photos = reference_tree(tmp_path)
    catalog = FakeCatalog(photos)
    app = create_app(
        matcher=FakeMatcher(prediction()),
        geometry=ScriptedGeometry(inliers),
        catalog=catalog,
        reference_storage=LocalImageStorage(root),
    )
    return TestClient(app), catalog


def test_verify_ranks_candidates_by_inlier_ratio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MIN_MATCHES", "2")
    monkeypatch.setenv("MIN_INLIERS", "2")
    monkeypatch.setenv("MIN_INLIER_RATIO", "0.5")
    client, catalog = ranking_client(tmp_path, inliers=[1, 2])
    before = upload_dirs()

    response = client.post(
        "/verify",
        files={"query": ("photo.jpg", photo_bytes(), "image/jpeg")},
        data={"candidates": '["low", "high"]'},
    )

    assert response.status_code == 200
    assert response.json() == [
        {"id": "high", "score": 1.0},
        {"id": "low", "score": 0.0},
    ]
    assert catalog.calls == [["low", "high"]]
    assert upload_dirs() == before


def test_verify_uses_the_best_photo_of_one_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MIN_MATCHES", "2")
    monkeypatch.setenv("MIN_INLIERS", "2")
    monkeypatch.setenv("MIN_INLIER_RATIO", "0.5")
    root, photos = reference_tree(tmp_path)
    second = root / "reference" / "low-2.jpg"
    second.write_bytes(photo_bytes())
    photos["low"] = [photos["low"][0], str(second)]
    from superpoint.infrastructure.storage.local_storage import LocalImageStorage

    client = TestClient(
        create_app(
            matcher=FakeMatcher(prediction()),
            geometry=ScriptedGeometry([0, 2]),
            catalog=FakeCatalog(photos),
            reference_storage=LocalImageStorage(root),
        )
    )

    response = client.post(
        "/verify",
        files={"query": ("photo.jpg", photo_bytes(), "image/jpeg")},
        data={"candidates": '["low"]'},
    )

    assert response.status_code == 200
    assert response.json() == [{"id": "low", "score": 1.0}]


def test_verify_rejects_unknown_ids(tmp_path: Path) -> None:
    client, _catalog = ranking_client(tmp_path, inliers=[])

    response = client.post(
        "/verify",
        files={"query": ("photo.jpg", photo_bytes(), "image/jpeg")},
        data={"candidates": '["missing", "low"]'},
    )

    assert response.status_code == 404
    assert "missing" in response.json()["detail"]


def test_verify_reports_catalog_failure(tmp_path: Path) -> None:
    from superpoint.infrastructure.database.reference_catalog import CatalogError

    class BrokenCatalog:
        def image_uris(self, wine_ids):
            raise CatalogError("Cannot read reference photos: connection refused")

    root, _photos = reference_tree(tmp_path)
    client = TestClient(
        create_app(
            matcher=FakeMatcher(prediction()),
            geometry=FakeGeometry(),
            catalog=BrokenCatalog(),
            reference_storage=LocalImageStorage(root),
        )
    )

    response = client.post(
        "/verify",
        files={"query": ("photo.jpg", photo_bytes(), "image/jpeg")},
        data={"candidates": '["low"]'},
    )

    assert response.status_code == 503
    assert "connection refused" in response.json()["detail"]


def test_parse_candidate_ids_accepts_swagger_wrapped_array() -> None:
    pasted = """[
  "shepot",
  "Tamara_22_9ef262fd0c"
]"""

    assert parse_candidate_ids(pasted) == ["shepot", "Tamara_22_9ef262fd0c"]
    assert parse_candidate_ids(f'"{pasted}"') == ["shepot", "Tamara_22_9ef262fd0c"]
    assert parse_candidate_ids("shepot, Tamara_22_9ef262fd0c") == [
        "shepot",
        "Tamara_22_9ef262fd0c",
    ]


@pytest.mark.parametrize(
    "candidates",
    ["[]", "null", '["ok", "ok"]', "[" + ",".join(f'"{index}"' for index in range(21)) + "]"],
)
def test_verify_rejects_bad_candidate_lists(tmp_path: Path, candidates: str) -> None:
    client, _catalog = ranking_client(tmp_path, inliers=[])

    response = client.post(
        "/verify",
        files={"query": ("photo.jpg", photo_bytes(), "image/jpeg")},
        data={"candidates": candidates},
    )

    assert response.status_code == 422


def test_verify_requires_query_and_candidates(client: TestClient) -> None:
    response = client.post(
        "/verify",
        files={"query": ("photo.jpg", photo_bytes(), "image/jpeg")},
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    ("query_upload", "status_code"),
    [
        (("notes.txt", b"text", "text/plain"), 415),
        (("broken.jpg", b"not an image", "image/jpeg"), 422),
        (("empty.jpg", b"", "image/jpeg"), 422),
    ],
)
def test_verify_upload_errors(
    client: TestClient,
    query_upload: tuple[str, bytes, str],
    status_code: int,
) -> None:
    before = upload_dirs()

    response = client.post(
        "/verify",
        files={"query": query_upload},
        data={"candidates": '["shepot"]'},
    )

    assert response.status_code == status_code
    assert response.json()["detail"]
    assert upload_dirs() == before, "failed uploads must not be kept"


def test_verify_upload_too_large(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MAX_IMAGE_SIZE_BYTES", "100")
    before = upload_dirs()

    response = client.post(
        "/verify",
        files={"query": ("photo.jpg", photo_bytes(), "image/jpeg")},
        data={"candidates": '["shepot"]'},
    )

    assert response.status_code == 413
    assert upload_dirs() == before


def test_verify_rejects_body_larger_than_one_image(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MAX_IMAGE_SIZE_BYTES", "100")
    before = upload_dirs()

    response = client.post(
        "/verify",
        content=b"x" * (max_upload_body_bytes() + 1),
        headers={"content-type": "multipart/form-data; boundary=bound"},
    )

    assert response.status_code == 413
    assert response.json()["detail"]
    assert upload_dirs() == before
