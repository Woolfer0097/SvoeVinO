from __future__ import annotations

import io
import tempfile
from collections.abc import Sequence
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from dinov2_retrieval.contracts import ReferenceImageRecord, ReferenceMatch
from dinov2_retrieval.entrypoints.api import EmbedderProvider, create_app
from dinov2_retrieval.retrieval.reference_repository import RepositoryError


class FakeEmbedder:
    model_name = "fake/model"
    device = "cpu"

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, image: Image.Image) -> list[float]:
        assert image.mode == "RGB"
        self.calls += 1
        return [0.6, 0.8]


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    queries = tmp_path / "queries"
    queries.mkdir()
    Image.new("RGB", (32, 16), color=(20, 40, 60)).save(queries / "test.jpg")
    (queries / "broken.jpg").write_bytes(b"not an image")
    (queries / "notes.txt").write_text("text")
    return TestClient(create_app(embedder=FakeEmbedder()))


def test_health(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_swagger_is_available(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/search", "/search/uri", "/validate", "/embed"} <= set(paths)


def test_validate_returns_image_info(client: TestClient) -> None:
    response = client.post("/validate", json={"image_uri": "queries/test.jpg"})

    assert response.status_code == 200
    body = response.json()
    assert body["width"] == 32
    assert body["height"] == 16
    assert body["mime_type"] == "image/jpeg"


def test_embed_returns_full_embedding(client: TestClient) -> None:
    response = client.post("/embed", json={"image_uri": "queries/test.jpg"})

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "fake/model"
    assert body["dimension"] == 2
    assert body["embedding"] == [0.6, 0.8]


@pytest.mark.parametrize(
    ("image_uri", "status_code"),
    [
        ("queries/missing.jpg", 404),
        ("/etc/passwd", 400),
        ("queries/notes.txt", 415),
        ("queries/broken.jpg", 422),
    ],
)
def test_validation_errors_map_to_http_status(
    client: TestClient, image_uri: str, status_code: int
) -> None:
    response = client.post("/validate", json={"image_uri": image_uri})

    assert response.status_code == status_code
    assert response.json()["detail"]


def test_file_too_large_returns_413(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("MAX_IMAGE_SIZE_BYTES", "10")
    Image.new("RGB", (32, 16)).save(tmp_path / "big.jpg")
    client = TestClient(create_app(embedder=FakeEmbedder()))

    response = client.post("/embed", json={"image_uri": "big.jpg"})

    assert response.status_code == 413


def test_embedder_provider_loads_model_once(monkeypatch: pytest.MonkeyPatch) -> None:
    from dinov2_retrieval.embedding import dinov2_embedder

    created: list[FakeEmbedder] = []

    def fake_factory(**options) -> FakeEmbedder:
        created.append(FakeEmbedder())
        return created[-1]

    monkeypatch.setattr(dinov2_embedder, "DinoV2Embedder", fake_factory)
    provider = EmbedderProvider()

    assert provider.get() is provider.get()
    assert len(created) == 1


def test_embedder_provider_respects_gpu_environment(monkeypatch) -> None:
    from dinov2_retrieval.embedding import dinov2_embedder

    monkeypatch.setenv("DINO_DEVICE", "cuda")
    monkeypatch.setenv("DINO_DTYPE", "float16")
    captured = {}

    def factory(**options):
        captured.update(options)
        return FakeEmbedder()

    monkeypatch.setattr(dinov2_embedder, "DinoV2Embedder", factory)
    EmbedderProvider().get()
    assert captured == {"device": "cuda", "dtype": "float16"}



class FakeRepository:
    def __init__(self, matches: list[ReferenceMatch]) -> None:
        self.matches = matches
        self.limits: list[int] = []
        self.upserts: list[ReferenceImageRecord] = []
        self.opened = 0
        self.closed = 0

    def get_wine_ids(self, model_name: str) -> set[str]:
        return {match.wine_id for match in self.matches}

    def get_reference_count(self, model_name: str | None = None) -> int:
        return len({record.image_uri for record in self.upserts}) or len(self.matches)

    def list_references(self, model_name: str, limit: int, offset: int = 0):
        from dinov2_retrieval.contracts import ReferenceWine

        wines = [
            ReferenceWine(wine_id=m.wine_id, slug=m.slug, image_uris=[m.image_uri])
            for m in sorted(self.matches, key=lambda m: m.wine_id)
        ]
        return wines[offset : offset + limit]

    def delete_references_except(self, model_name: str, keep_image_uris) -> int:
        return 0

    def search_similar(
        self, query_embedding: Sequence[float], limit: int, model_name: str
    ) -> list[ReferenceMatch]:
        assert list(query_embedding) == [0.6, 0.8]
        self.limits.append(limit)
        return self.matches[:limit]

    def upsert_reference_embedding(self, record: ReferenceImageRecord) -> str:
        self.upserts.append(record)
        return "inserted"

    def close(self) -> None:
        self.closed += 1


def wine(index: int, distance: float) -> ReferenceMatch:
    return ReferenceMatch(
        wine_id=f"wine-{index:03d}",
        slug=f"wine-{index:03d}-slug",
        image_uri=f"/data/reference/wine-{index:03d}.webp",
        distance=distance,
    )


@pytest.fixture
def repository() -> FakeRepository:
    return FakeRepository([wine(index, index / 100) for index in range(30)])


@pytest.fixture
def search_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, repository: FakeRepository
) -> TestClient:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    (tmp_path / "queries").mkdir()
    Image.new("RGB", (32, 16), color=(20, 40, 60)).save(tmp_path / "queries/test.jpg")

    def connect() -> FakeRepository:
        repository.opened += 1
        return repository

    return TestClient(create_app(embedder=FakeEmbedder(), repository_factory=connect))


def photo_bytes(image_format: str = "JPEG") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 16), color=(120, 20, 60)).save(buffer, format=image_format)
    return buffer.getvalue()


def upload_dirs() -> set[str]:
    return {path.name for path in Path(tempfile.gettempdir()).glob("dinov2-upload-*")}


def test_search_upload_returns_top_20_wines(
    search_client: TestClient, repository: FakeRepository
) -> None:
    before = upload_dirs()

    response = search_client.post(
        "/search", files={"file": ("bottle.jpg", photo_bytes(), "image/jpeg")}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_name"] == "fake/model"
    assert len(body["request_id"]) == 32
    assert [c["wine_id"] for c in body["candidates"]] == [
        f"wine-{index:03d}" for index in range(20)
    ]
    assert body["candidates"][1]["score"] == pytest.approx(0.99)
    assert "message" not in body
    assert (repository.opened, repository.closed) == (1, 1)
    assert repository.upserts == []
    assert upload_dirs() == before, "the uploaded photo must not be kept"


def test_search_upload_with_top_k_and_request_id(search_client: TestClient) -> None:
    response = search_client.post(
        "/search",
        files={"file": ("bottle.webp", photo_bytes("WEBP"), "image/webp")},
        data={"top_k": "5", "request_id": "request-001"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == "request-001"
    assert len(body["candidates"]) == 5


def test_search_upload_default_top_k_comes_from_config(
    search_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEFAULT_TOP_K", "3")

    response = search_client.post(
        "/search", files={"file": ("bottle.png", photo_bytes("PNG"), "image/png")}
    )

    assert len(response.json()["candidates"]) == 3


def test_search_upload_uses_content_type_without_extension(
    search_client: TestClient,
) -> None:
    response = search_client.post(
        "/search", files={"file": ("blob", photo_bytes("WEBP"), "image/webp")}
    )

    assert response.status_code == 200


@pytest.mark.parametrize(
    ("upload", "status_code"),
    [
        (("notes.txt", b"text", "text/plain"), 415),
        (("broken.jpg", b"not an image", "image/jpeg"), 422),
        (("empty.jpg", b"", "image/jpeg"), 422),
    ],
)
def test_search_upload_errors(
    search_client: TestClient,
    repository: FakeRepository,
    upload: tuple[str, bytes, str],
    status_code: int,
) -> None:
    response = search_client.post("/search", files={"file": upload})

    assert response.status_code == status_code
    assert response.json()["detail"]
    assert repository.limits == []


def test_search_upload_too_large(
    search_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MAX_IMAGE_SIZE_BYTES", "100")

    response = search_client.post(
        "/search", files={"file": ("bottle.jpg", photo_bytes(), "image/jpeg")}
    )

    assert response.status_code == 413


@pytest.mark.parametrize("data", [{"top_k": "0"}, {"top_k": "many"}])
def test_search_upload_rejects_invalid_top_k(
    search_client: TestClient, data: dict[str, str]
) -> None:
    response = search_client.post(
        "/search", files={"file": ("bottle.jpg", photo_bytes(), "image/jpeg")}, data=data
    )

    assert response.status_code == 422


def test_search_requires_a_file(search_client: TestClient) -> None:
    assert search_client.post("/search").status_code == 422


def test_search_without_indexed_photos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    client = TestClient(
        create_app(embedder=FakeEmbedder(), repository_factory=lambda: FakeRepository([]))
    )

    response = client.post(
        "/search", files={"file": ("bottle.jpg", photo_bytes(), "image/jpeg")}
    )

    assert response.status_code == 200
    assert response.json()["status"] == "no_results"
    assert "run the index command" in response.json()["message"]


def test_database_unavailable_returns_503(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))

    def unavailable() -> FakeRepository:
        raise RepositoryError("Cannot connect to PostgreSQL")

    client = TestClient(create_app(embedder=FakeEmbedder(), repository_factory=unavailable))

    response = client.post(
        "/search", files={"file": ("bottle.jpg", photo_bytes(), "image/jpeg")}
    )

    assert response.status_code == 503
    assert "Cannot connect" in response.json()["detail"]


def test_search_by_uri(search_client: TestClient) -> None:
    response = search_client.post(
        "/search/uri",
        json={"image_uri": "queries/test.jpg", "top_k": 2, "request_id": "r-1"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == "r-1"
    assert [c["wine_id"] for c in body["candidates"]] == ["wine-000", "wine-001"]


def test_search_by_uri_errors(search_client: TestClient) -> None:
    assert search_client.post(
        "/search/uri", json={"image_uri": "queries/missing.jpg"}
    ).status_code == 404
    assert search_client.post(
        "/search/uri", json={"image_uri": "queries/test.jpg", "top_k": 0}
    ).status_code == 422


def test_search_ids_returns_only_top_20_wine_ids(
    search_client: TestClient,
    repository: FakeRepository,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEFAULT_TOP_K", "3")

    response = search_client.get(
        "/search/ids", params={"image_uri": str(tmp_path / "queries/test.jpg")}
    )

    assert response.status_code == 200
    assert response.json() == [f"wine-{index:03d}" for index in range(20)]
    assert (repository.opened, repository.closed) == (1, 1)
    assert repository.upserts == []


def test_search_ids_accepts_a_path_relative_to_data_root(
    search_client: TestClient,
) -> None:
    response = search_client.get("/search/ids", params={"image_uri": "queries/test.jpg"})

    assert response.status_code == 200
    assert len(response.json()) == 20


def test_search_ids_without_indexed_photos_is_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    Image.new("RGB", (32, 16)).save(tmp_path / "bottle.jpg")
    client = TestClient(
        create_app(embedder=FakeEmbedder(), repository_factory=lambda: FakeRepository([]))
    )

    response = client.get("/search/ids", params={"image_uri": "bottle.jpg"})

    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.parametrize(
    ("params", "status_code"),
    [
        ({"image_uri": "queries/missing.jpg"}, 404),
        ({"image_uri": "/etc/passwd"}, 400),
        ({"image_uri": ""}, 422),
        ({}, 422),
    ],
)
def test_search_ids_errors(
    search_client: TestClient,
    repository: FakeRepository,
    params: dict[str, str],
    status_code: int,
) -> None:
    response = search_client.get("/search/ids", params=params)

    assert response.status_code == status_code
    assert repository.limits == []


def test_model_is_loaded_once_at_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, repository: FakeRepository
) -> None:
    from dinov2_retrieval.embedding import dinov2_embedder

    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    created: list[FakeEmbedder] = []

    def fake_factory(**options) -> FakeEmbedder:
        created.append(FakeEmbedder())
        return created[-1]

    monkeypatch.setattr(dinov2_embedder, "DinoV2Embedder", fake_factory)
    app = create_app(repository_factory=lambda: repository)

    with TestClient(app) as client:
        assert len(created) == 1, "the model must be loaded before the first request"
        assert created[0].calls == 1, "one warm-up pass at startup"
        for _ in range(3):
            response = client.post(
                "/search", files={"file": ("bottle.jpg", photo_bytes(), "image/jpeg")}
            )
            assert response.status_code == 200

    assert len(created) == 1



# --- Swagger and the endpoints that replace CLI commands -------------------


@pytest.fixture
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    (tmp_path / "queries").mkdir()
    Image.new("RGB", (1200, 600), color=(20, 40, 60)).save(tmp_path / "queries/test.jpg")
    (tmp_path / "reference" / "donum-2023").mkdir(parents=True)
    Image.new("RGB", (32, 16)).save(tmp_path / "reference/87.6_20-08-2026.webp")
    Image.new("RGB", (32, 16)).save(tmp_path / "reference/donum-2023/front.png")
    return tmp_path.resolve()


@pytest.fixture
def app_client(data_root: Path, repository: FakeRepository) -> TestClient:
    return TestClient(
        create_app(embedder=FakeEmbedder(), repository_factory=lambda: repository)
    )


def test_root_opens_swagger(app_client: TestClient) -> None:
    response = app_client.get("/", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


def test_swagger_is_grouped_and_ready_to_try(app_client: TestClient) -> None:
    schema = app_client.get("/openapi.json").json()
    tags = {
        path: next(iter(operation.values()))["tags"][0]
        for path, operation in schema["paths"].items()
    }

    assert [tag["name"] for tag in schema["tags"]] == [
        "Поиск",
        "Эталоны",
        "Оценка качества",
        "Служебное",
    ]
    assert tags == {
        "/search": "Поиск",
        "/search/uri": "Поиск",
        "/search/ids": "Поиск",
        "/index": "Эталоны",
        "/references": "Эталоны",
        "/images": "Эталоны",
        "/evaluate": "Оценка качества",
        "/health": "Служебное",
        "/health/details": "Служебное",
        "/validate": "Служебное",
        "/embed": "Служебное",
    }
    docs = app_client.get("/docs").text
    assert '"tryItOutEnabled": true' in docs
    assert '"displayRequestDuration": true' in docs


def test_index_without_body_fields_uses_data_reference(
    app_client: TestClient, repository: FakeRepository, data_root: Path
) -> None:
    response = app_client.post("/index", json={})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["source"] == f"folder {data_root}/reference"
    assert (body["total"], body["inserted"]) == (2, 2)
    assert sorted(record.wine_id for record in repository.upserts) == [
        "87.6_20-08-2026",
        "donum-2023",
    ]
    assert repository.closed == 1


def test_index_with_prune_and_manifest(
    app_client: TestClient, repository: FakeRepository, data_root: Path
) -> None:
    (data_root / "reference/manifest.csv").write_text(
        "wine_id,slug,image_uri\n"
        f"10231,donum-xxiv,{data_root}/reference/donum-2023/front.png\n"
    )

    response = app_client.post("/index", json={"prune": True})

    assert response.status_code == 200
    body = response.json()
    assert body["source"] == f"manifest {data_root}/reference/manifest.csv"
    assert [record.wine_id for record in repository.upserts] == ["10231"]


@pytest.mark.parametrize(
    ("payload", "status_code"),
    [
        ({"manifest": "reference/missing.csv"}, 400),
        ({"reference_dir": "empty"}, 400),
        ({"manifest": "a.csv", "reference_dir": "reference"}, 422),
    ],
)
def test_index_errors(
    app_client: TestClient, data_root: Path, payload: dict, status_code: int
) -> None:
    response = app_client.post("/index", json=payload)

    assert response.status_code == status_code
    assert response.json()["detail"]


def test_references_lists_wines(app_client: TestClient) -> None:
    response = app_client.get("/references", params={"limit": 2, "offset": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "fake/model"
    assert (body["wine_count"], body["photo_count"], body["offset"]) == (30, 30, 1)
    assert [wine["wine_id"] for wine in body["wines"]] == ["wine-001", "wine-002"]
    assert body["wines"][0]["image_uris"] == ["/data/reference/wine-001.webp"]


def test_images_returns_a_preview_by_default(app_client: TestClient) -> None:
    response = app_client.get("/images", params={"uri": "queries/test.jpg"})

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    preview = Image.open(io.BytesIO(response.content))
    assert preview.size == (800, 400)


def test_images_returns_the_original_file(
    app_client: TestClient, data_root: Path
) -> None:
    response = app_client.get(
        "/images", params={"uri": "reference/87.6_20-08-2026.webp", "max_side": 0}
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/webp"
    assert response.content == (data_root / "reference/87.6_20-08-2026.webp").read_bytes()


@pytest.mark.parametrize(
    ("uri", "status_code"),
    [("queries/missing.jpg", 404), ("/etc/passwd", 400), ("../outside.jpg", 400)],
)
def test_images_errors(app_client: TestClient, uri: str, status_code: int) -> None:
    assert app_client.get("/images", params={"uri": uri}).status_code == status_code


def test_evaluate_reports_recall(app_client: TestClient, data_root: Path) -> None:
    (data_root / "evaluation").mkdir()
    (data_root / "evaluation/queries.csv").write_text(
        f"query_image_uri,wine_id\n{data_root}/queries/test.jpg,wine-002\n"
    )

    response = app_client.post(
        "/evaluate", json={"queries": "evaluation/queries.csv", "top_k": 5}
    )

    assert response.status_code == 200
    body = response.json()
    assert (body["status"], body["queries_processed"], body["top_k"]) == ("ok", 1, 5)
    assert body["recall_at"] == {"1": 0.0, "5": 1.0, "20": 1.0}


def test_evaluate_without_queries_file(app_client: TestClient) -> None:
    response = app_client.post("/evaluate", json={})

    assert response.status_code == 400
    assert "Queries file does not exist" in response.json()["detail"]


def healthy_database(settings):
    from dinov2_retrieval.contracts import DatabaseInspection

    return DatabaseInspection(
        server_version="16",
        pgvector_version="0.8.0",
        table_exists=True,
        embedding_dimension=1536,
        reference_count=1,
        model_reference_count=1,
    )


class DimensionEmbedder(FakeEmbedder):
    def embed(self, image: Image.Image) -> list[float]:
        return [0.0] * 1536


@pytest.mark.parametrize(("database_ok", "status_code"), [(True, 200), (False, 503)])
def test_health_details(
    data_root: Path, monkeypatch: pytest.MonkeyPatch, database_ok: bool, status_code: int
) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://dinov2:secret@postgres:5432/dinov2")

    def inspect(settings):
        if not database_ok:
            raise RepositoryError("Cannot connect to PostgreSQL")
        return healthy_database(settings)

    client = TestClient(
        create_app(embedder=DimensionEmbedder(), database_inspector=inspect)
    )

    response = client.get("/health/details")

    assert response.status_code == status_code
    body = response.json()
    assert body["status"] == ("ok" if database_ok else "error")
    assert body["checks"]["model"]["status"] == "ok"
    assert "secret" not in response.text


def test_relative_path_error_explains_data_root(app_client: TestClient, data_root: Path) -> None:
    response = app_client.get("/images", params={"uri": "data/queries/test.jpg"})

    assert response.status_code == 404
    detail = response.json()["detail"]
    assert f"relative paths are taken from DATA_ROOT {data_root}" in detail
    assert f"{data_root}/data/queries/test.jpg" in detail
    assert app_client.get("/images", params={"uri": "queries/test.jpg"}).status_code == 200
    assert (
        app_client.get("/images", params={"uri": f"{data_root}/queries/test.jpg"}).status_code
        == 200
    )


def test_index_examples_never_combine_both_sources(app_client: TestClient) -> None:
    schema = app_client.get("/openapi.json").json()
    body = schema["paths"]["/index"]["post"]["requestBody"]["content"]["application/json"]
    examples = [example["value"] for example in body["examples"].values()]

    assert examples[0] == {"prune": True}
    assert all(not {"manifest", "reference_dir"} <= set(value) for value in examples)
