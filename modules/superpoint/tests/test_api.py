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
from superpoint.entrypoints.api import create_app, max_upload_body_bytes


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

    assert schema["info"]["title"] == "SuperPoint — проверка пары фото"
    assert [tag["name"] for tag in schema["tags"]] == ["Проверка", "Служебное"]
    assert tags == {
        "/verify": "Проверка",
        "/health": "Служебное",
    }
    docs = client.get("/docs").text
    assert '"tryItOutEnabled": true' in docs
    assert '"displayRequestDuration": true' in docs


def test_verify_upload_does_not_keep_files(client: TestClient) -> None:
    before = upload_dirs()

    response = client.post(
        "/verify",
        files={
            "query": ("photo.jpg", photo_bytes(), "image/jpeg"),
            "reference": ("label.png", photo_bytes("PNG"), "image/png"),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["verified"] is False
    assert body["model"] == "fake-matcher"
    assert body["device"] == "cpu"
    assert body["num_matches"] == 2
    assert body["num_inliers"] == 2
    assert body["query_width"] == 32
    assert body["reference_height"] == 16
    assert body["query_path"] == "photo.jpg"
    assert body["reference_path"] == "label.png"
    assert "superpoint-upload" not in body["query_path"]
    assert upload_dirs() == before, "uploaded photos must not be kept"


def test_verify_upload_accepts_a_pair_when_thresholds_pass(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MIN_MATCHES", "2")
    monkeypatch.setenv("MIN_INLIERS", "2")
    monkeypatch.setenv("MIN_INLIER_RATIO", "0.5")

    response = client.post(
        "/verify",
        files={
            "query": ("photo.jpg", photo_bytes(), "image/jpeg"),
            "reference": ("label.jpg", photo_bytes(), "image/jpeg"),
        },
    )

    assert response.status_code == 200
    assert response.json()["verified"] is True
    assert response.json()["inlier_ratio"] == 1.0
    assert response.json()["mean_match_score"] == 0.8


def test_verify_upload_uses_content_type_without_extension(client: TestClient) -> None:
    response = client.post(
        "/verify",
        files={
            "query": ("blob", photo_bytes("WEBP"), "image/webp"),
            "reference": ("other", photo_bytes(), "image/jpeg"),
        },
    )

    assert response.status_code == 200


def test_verify_requires_both_files(client: TestClient) -> None:
    response = client.post(
        "/verify",
        files={"query": ("photo.jpg", photo_bytes(), "image/jpeg")},
    )

    assert response.status_code == 422


def test_verify_upload_uses_file_name_not_a_directory(client: TestClient) -> None:
    response = client.post(
        "/verify",
        files={
            "query": ("../../secret.jpg", photo_bytes(), "image/jpeg"),
            "reference": ("labels/ref.png", photo_bytes("PNG"), "image/png"),
        },
    )

    assert response.status_code == 200
    assert response.json()["query_path"] == "secret.jpg"
    assert response.json()["reference_path"] == "ref.png"


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
        files={
            "query": query_upload,
            "reference": ("label.jpg", photo_bytes(), "image/jpeg"),
        },
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
        files={
            "query": ("photo.jpg", photo_bytes(), "image/jpeg"),
            "reference": ("label.jpg", photo_bytes(), "image/jpeg"),
        },
    )

    assert response.status_code == 413
    assert upload_dirs() == before


def test_verify_rejects_body_larger_than_two_images(
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
