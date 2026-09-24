from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from PIL import Image

from dinov2_retrieval.contracts import (
    EmbeddingResult,
    HealthCheck,
    HealthReport,
    ReferenceImageRecord,
    ReferenceMatch,
)
from dinov2_retrieval.embedding.base import EmbeddingError
from dinov2_retrieval.entrypoints import cli
from dinov2_retrieval.retrieval.reference_repository import RepositoryError


class FakeEmbedder:
    model_name = "fake/model"
    device = "cpu"

    def embed(self, image: Image.Image) -> list[float]:
        return [1.0, 0.0]


class FakeRepository:
    def __init__(self, matches: list[ReferenceMatch] | None = None) -> None:
        self.matches = matches or []
        self.records: dict[str, ReferenceImageRecord] = {}
        self.search_limits: list[int] = []
        self.closed = False

    def upsert_reference_embedding(self, record: ReferenceImageRecord) -> str:
        outcome = "updated" if record.image_uri in self.records else "inserted"
        self.records[record.image_uri] = record
        return outcome

    def get_reference_count(self, model_name: str | None = None) -> int:
        return len(self.records)

    def get_wine_ids(self, model_name: str) -> set[str]:
        return {match.wine_id for match in self.matches}

    def delete_references_except(self, model_name: str, keep_image_uris) -> int:
        stale = [uri for uri in self.records if uri not in set(keep_image_uris)]
        for uri in stale:
            del self.records[uri]
        return len(stale)

    def search_similar(
        self, query_embedding: Sequence[float], limit: int, model_name: str
    ) -> list[ReferenceMatch]:
        self.search_limits.append(limit)
        return self.matches[:limit]

    def close(self) -> None:
        self.closed = True


def match(wine_id: str, distance: float) -> ReferenceMatch:
    return ReferenceMatch(
        wine_id=wine_id,
        slug=f"{wine_id}-slug",
        image_uri=f"/data/reference/{wine_id}/photo-1.jpg",
        distance=distance,
    )


@pytest.fixture
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    (tmp_path / "queries").mkdir()
    Image.new("RGB", (16, 8), color=(10, 20, 30)).save(tmp_path / "queries/test.jpeg")
    for wine_id in ("wine-001", "wine-002"):
        (tmp_path / "reference" / wine_id).mkdir(parents=True)
        Image.new("RGB", (16, 8)).save(tmp_path / "reference" / wine_id / "photo-1.jpg")
    (tmp_path / "requests").mkdir()
    return tmp_path.resolve()


@pytest.fixture
def repository(monkeypatch: pytest.MonkeyPatch) -> FakeRepository:
    fake = FakeRepository([match("wine-001", 0.06), match("wine-002", 0.25)])
    monkeypatch.setattr(cli, "_connect_repository", lambda: fake)
    monkeypatch.setattr(cli, "_create_embedder", FakeEmbedder)
    return fake


def run_cli(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict]:
    exit_code = cli.main(list(argv))
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1, "without a TTY the result must be one JSON line"
    return exit_code, json.loads(lines[0])


def test_cli_embed_prints_only_preview(monkeypatch, capsys) -> None:
    expected = EmbeddingResult(
        path="/data/queries/test.jpeg",
        model="facebook/dinov2-small",
        dimension=384,
        device="cpu",
        embedding=[float(index) for index in range(384)],
    )
    monkeypatch.setattr(cli, "create_embedding", lambda image_uri: expected)

    assert cli.main(["embed", "--image-uri", "/data/queries/test.jpeg"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["path"] == expected.path
    assert output["dimension"] == 384
    assert output["device"] == "cpu"
    assert output["embedding_preview"] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert "embedding" not in output


def test_validate_prints_image_info(data_root: Path, capsys) -> None:
    exit_code, output = run_cli(
        capsys, "validate", "--image-uri", str(data_root / "queries/test.jpeg")
    )

    assert exit_code == 0
    assert output == {
        "path": str(data_root / "queries/test.jpeg"),
        "width": 16,
        "height": 8,
        "mime_type": "image/jpeg",
    }


def test_failures_are_json_with_an_exit_code(data_root: Path, capsys) -> None:
    exit_code = cli.main(["validate", "--image-uri", "queries/missing.jpeg"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert json.loads(captured.out) == {
        "status": "error",
        "command": "validate",
        "error": {
            "category": "input",
            "type": "ImageNotFoundError",
            "message": "Image file does not exist: queries/missing.jpeg "
            f"(relative paths are taken from DATA_ROOT {data_root}, "
            f"so this is {data_root}/queries/missing.jpeg)",
        },
    }
    assert "validation failed: Image file does not exist" in captured.err


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["search"],
        ["index", "--prune", "--manifest"],
        ["search", "--image-uri", "queries/test.jpeg", "--top-k", "0"],
        ["search", "--image-uri", "a.jpg", "--request-json", "request.json"],
        ["index", "--manifest", "m.csv", "--reference-dir", "reference"],
    ],
)
def test_usage_errors_exit_with_code_2(argv: list[str], capsys) -> None:
    exit_code, output = run_cli(capsys, *argv)

    assert exit_code == 2
    assert output["status"] == "error"
    assert output["error"]["category"] == "usage"


def test_search_by_image_uri(data_root: Path, repository: FakeRepository, capsys) -> None:
    exit_code, output = run_cli(
        capsys,
        "search",
        "--image-uri",
        "queries/test.jpeg",
        "--top-k",
        "1",
        "--request-id",
        "request-001",
    )

    assert exit_code == 0
    assert output == {
        "request_id": "request-001",
        "status": "ok",
        "model_name": "fake/model",
        "query_embedding_dimension": 2,
        "candidates": [
            {
                "wine_id": "wine-001",
                "slug": "wine-001-slug",
                "score": pytest.approx(0.94),
                "distance": 0.06,
                "best_image_uri": "/data/reference/wine-001/photo-1.jpg",
            }
        ],
    }
    assert repository.search_limits == [100]
    assert repository.closed


def test_search_defaults_to_generated_id_and_default_top_k(
    data_root: Path, repository: FakeRepository, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("DEFAULT_TOP_K", "1")

    exit_code, output = run_cli(capsys, "search", "--image-uri", "queries/test.jpeg")

    assert exit_code == 0
    assert len(output["request_id"]) == 32
    assert len(output["candidates"]) == 1


def test_search_by_request_json(data_root: Path, repository: FakeRepository, capsys) -> None:
    request = {
        "request_id": "request-001",
        "image_uri": str(data_root / "queries/test.jpeg"),
        "top_k": 20,
    }
    (data_root / "requests/search-request.json").write_text(json.dumps(request))

    exit_code, output = run_cli(
        capsys, "search", "--request-json", "requests/search-request.json"
    )

    assert exit_code == 0
    assert output["request_id"] == "request-001"
    assert [c["wine_id"] for c in output["candidates"]] == ["wine-001", "wine-002"]


def test_request_json_without_top_k_uses_default(
    data_root: Path, repository: FakeRepository, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("DEFAULT_TOP_K", "1")
    request_path = data_root / "requests/search-request.json"
    request_path.write_text('{"request_id": "r-1", "image_uri": "queries/test.jpeg"}')

    exit_code, output = run_cli(capsys, "search", "--request-json", str(request_path))

    assert exit_code == 0
    assert len(output["candidates"]) == 1


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not json", "is not valid JSON"),
        ("[]", "must contain a JSON object"),
        ('{"image_uri": "queries/test.jpeg"}', "request_id: Field required"),
        ('{"request_id": "r", "image_uri": "q.jpg", "top_k": 0}', "top_k: Input should be greater than 0"),
        ('{"request_id": "r", "image_uri": "q.jpg", "topk": 5}', "topk: Extra inputs are not permitted"),
    ],
)
def test_invalid_request_json(
    data_root: Path, repository: FakeRepository, content: str, message: str, capsys
) -> None:
    (data_root / "requests/bad.json").write_text(content)

    exit_code, output = run_cli(capsys, "search", "--request-json", "requests/bad.json")

    assert exit_code == 1
    assert output["error"]["category"] == "input"
    assert output["error"]["type"] == "RequestFileError"
    assert message in output["error"]["message"]


def test_missing_request_json(data_root: Path, repository: FakeRepository, capsys) -> None:
    exit_code, output = run_cli(capsys, "search", "--request-json", "requests/nope.json")

    assert exit_code == 1
    assert "Request file does not exist" in output["error"]["message"]


def test_request_json_cannot_be_combined_with_flags(data_root: Path, capsys) -> None:
    exit_code, output = run_cli(
        capsys, "search", "--request-json", "requests/r.json", "--top-k", "5"
    )

    assert exit_code == 2
    assert output["error"]["category"] == "usage"


def test_search_without_indexed_photos(data_root: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "_connect_repository", FakeRepository)
    monkeypatch.setattr(cli, "_create_embedder", FakeEmbedder)

    exit_code, output = run_cli(capsys, "search", "--image-uri", "queries/test.jpeg")

    assert exit_code == 0
    assert output["status"] == "no_results"
    assert output["candidates"] == []
    assert "run the index command" in output["message"]


def test_database_errors_exit_with_code_4(data_root: Path, monkeypatch, capsys) -> None:
    def unavailable():
        raise RepositoryError("Cannot connect to PostgreSQL at postgresql://u:***@db/x")

    monkeypatch.setattr(cli, "_connect_repository", unavailable)

    exit_code, output = run_cli(capsys, "search", "--image-uri", "queries/test.jpeg")

    assert exit_code == 4
    assert output["error"]["category"] == "database"


def test_model_errors_exit_with_code_3(
    data_root: Path, repository: FakeRepository, monkeypatch, capsys
) -> None:
    def broken_model():
        raise EmbeddingError("Cannot load model facebook/dinov2-small")

    monkeypatch.setattr(cli, "_create_embedder", broken_model)

    exit_code, output = run_cli(capsys, "search", "--image-uri", "queries/test.jpeg")

    assert exit_code == 3
    assert output["error"]["category"] == "model"
    assert repository.closed


def write_manifest(data_root: Path, *rows: str) -> Path:
    path = data_root / "reference/manifest.csv"
    path.write_text("\n".join(["wine_id,slug,image_uri", *rows]) + "\n")
    return path


def test_index_with_manifest(data_root: Path, repository: FakeRepository, capsys) -> None:
    write_manifest(
        data_root,
        f"wine-001,cabernet-2020,{data_root}/reference/wine-001/photo-1.jpg",
        f"wine-002,merlot-2021,{data_root}/reference/wine-002/photo-1.jpg",
    )

    exit_code, output = run_cli(capsys, "index", "--manifest", "reference/manifest.csv")

    assert exit_code == 0
    assert output == {
        "status": "ok",
        "model_name": "fake/model",
        "source": f"manifest {data_root}/reference/manifest.csv",
        "total": 2,
        "processed": 2,
        "inserted": 2,
        "updated": 0,
        "skipped": 0,
        "deleted": 0,
        "errors": 0,
        "error_details": [],
        "reference_count": 2,
    }
    assert sorted(record.slug for record in repository.records.values()) == [
        "cabernet-2020",
        "merlot-2021",
    ]
    assert repository.closed


def test_index_with_some_broken_photos_exits_with_code_5(
    data_root: Path, repository: FakeRepository, capsys
) -> None:
    write_manifest(
        data_root,
        f"wine-001,cabernet-2020,{data_root}/reference/wine-001/photo-1.jpg",
        f"wine-003,syrah-2019,{data_root}/reference/wine-003/missing.jpg",
    )

    exit_code, output = run_cli(capsys, "index", "--manifest", "reference/manifest.csv")

    assert exit_code == 5
    assert (output["status"], output["processed"], output["errors"]) == ("partial", 1, 1)
    assert output["error_details"][0]["error_type"] == "ImageNotFoundError"


def test_index_without_a_single_valid_photo_exits_with_code_1(
    data_root: Path, repository: FakeRepository, capsys
) -> None:
    write_manifest(data_root, f"wine-003,syrah-2019,{data_root}/reference/wine-003/missing.jpg")

    exit_code, output = run_cli(capsys, "index", "--manifest", "reference/manifest.csv")

    assert exit_code == 1
    assert output["status"] == "failed"


def test_invalid_manifest_stops_before_connecting(data_root: Path, monkeypatch, capsys) -> None:
    (data_root / "reference/manifest.csv").write_text("wine,slug,path\n")

    def must_not_connect():
        raise AssertionError("the manifest is checked first")

    monkeypatch.setattr(cli, "_connect_repository", must_not_connect)

    exit_code, output = run_cli(capsys, "index", "--manifest", "reference/manifest.csv")

    assert exit_code == 1
    assert output["error"]["type"] == "ManifestError"


def test_index_reference_dir(data_root: Path, repository: FakeRepository, capsys) -> None:
    exit_code, output = run_cli(capsys, "index", "--reference-dir", "reference")

    assert exit_code == 0
    assert output["inserted"] == 2
    assert sorted(record.wine_id for record in repository.records.values()) == [
        "wine-001",
        "wine-002",
    ]


@pytest.mark.parametrize(
    ("failed_check", "expected_exit_code"),
    [(None, 0), ("config", 1), ("reference_table", 4), ("model", 3)],
)
def test_health_exit_codes(
    failed_check: str | None, expected_exit_code: int, monkeypatch, capsys
) -> None:
    names = ["config", "database", "pgvector", "reference_table", "model"]
    report = HealthReport(
        status="ok" if failed_check is None else "error",
        checks={
            name: HealthCheck(status="error" if name == failed_check else "ok")
            for name in names
        },
    )
    monkeypatch.setattr(cli, "check_health", lambda: report)

    exit_code, output = run_cli(capsys, "health")

    assert exit_code == expected_exit_code
    assert output["status"] == report.status
    assert list(output["checks"]) == names


def write_queries(data_root: Path, *rows: str) -> Path:
    path = data_root / "evaluation/queries.csv"
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(["query_image_uri,wine_id", *rows]) + "\n")
    return path


def test_evaluate_prints_recall_report(
    data_root: Path, repository: FakeRepository, monkeypatch, capsys
) -> None:
    created = []

    def create_embedder():
        created.append(FakeEmbedder())
        return created[-1]

    monkeypatch.setattr(cli, "_create_embedder", create_embedder)
    query = f"{data_root}/queries/test.jpeg"
    write_queries(data_root, f"{query},wine-001", f"{data_root}/queries/copy.jpeg,wine-002")
    (data_root / "queries/copy.jpeg").write_bytes((data_root / "queries/test.jpeg").read_bytes())

    exit_code, output = run_cli(
        capsys, "evaluate", "--queries", "evaluation/queries.csv", "--top-k", "1"
    )

    assert exit_code == 0
    assert len(created) == 1
    assert output["status"] == "ok"
    assert (output["queries_total"], output["queries_processed"]) == (2, 2)
    assert (output["top_k"], output["recall_at_k"], output["correct_queries"]) == (1, 0.5, 1)
    assert output["recall_at"] == {"1": 0.5, "5": 1.0, "20": 1.0}
    assert output["incorrect_queries"] == [
        {
            "query_image_uri": f"{data_root}/queries/copy.jpeg",
            "expected_wine_id": "wine-002",
            "predicted_wine_ids": ["wine-001"],
            "expected_rank": 2,
            "reason": "expected wine is not in top-k",
        }
    ]
    assert output["errors"] == []
    assert repository.records == {}
    assert repository.closed


def test_evaluate_with_broken_photo_exits_with_code_5(
    data_root: Path, repository: FakeRepository, capsys
) -> None:
    write_queries(
        data_root,
        f"{data_root}/queries/test.jpeg,wine-001",
        f"{data_root}/queries/missing.jpeg,wine-002",
    )

    exit_code, output = run_cli(capsys, "evaluate", "--queries", "evaluation/queries.csv")

    assert exit_code == 5
    assert output["status"] == "partial"
    assert output["top_k"] == 20
    assert output["errors"][0]["error_type"] == "ImageNotFoundError"


def test_evaluate_rejects_invalid_queries_before_connecting(
    data_root: Path, monkeypatch, capsys
) -> None:
    write_queries(data_root, "queries/test.jpeg,wine-001")

    def must_not_connect():
        raise AssertionError("the queries file is checked first")

    monkeypatch.setattr(cli, "_connect_repository", must_not_connect)

    exit_code, output = run_cli(capsys, "evaluate", "--queries", "evaluation/queries.csv")

    assert exit_code == 1
    assert output["error"]["type"] == "ManifestError"
    assert "must be absolute" in output["error"]["message"]


def test_evaluate_requires_queries(capsys) -> None:
    exit_code, output = run_cli(capsys, "evaluate", "--top-k", "20")

    assert exit_code == 2
    assert output["error"]["category"] == "usage"


def test_index_without_source_scans_data_root_reference(
    data_root: Path, repository: FakeRepository, capsys
) -> None:
    Image.new("RGB", (16, 8)).save(data_root / "reference/87.6_20-08-2026.webp")

    exit_code, output = run_cli(capsys, "index")

    assert exit_code == 0
    assert output["source"] == f"folder {data_root}/reference"
    assert output["inserted"] == 3
    assert sorted(record.wine_id for record in repository.records.values()) == [
        "87.6_20-08-2026",
        "wine-001",
        "wine-002",
    ]


def test_index_without_source_prefers_reference_manifest(
    data_root: Path, repository: FakeRepository, capsys
) -> None:
    write_manifest(data_root, f"wine-001,cabernet-2020,{data_root}/reference/wine-001/photo-1.jpg")

    exit_code, output = run_cli(capsys, "index")

    assert exit_code == 0
    assert output["source"] == f"manifest {data_root}/reference/manifest.csv"
    assert output["total"] == 1


def test_index_prune_removes_photos_missing_from_the_source(
    data_root: Path, repository: FakeRepository, capsys
) -> None:
    run_cli(capsys, "index")
    (data_root / "reference/wine-002/photo-1.jpg").unlink()
    (data_root / "reference/wine-002").rmdir()

    exit_code, output = run_cli(capsys, "index", "--prune")

    assert exit_code == 0
    assert (output["updated"], output["deleted"], output["reference_count"]) == (1, 1, 1)
    assert [record.wine_id for record in repository.records.values()] == ["wine-001"]


def test_index_without_prune_keeps_old_photos(
    data_root: Path, repository: FakeRepository, capsys
) -> None:
    run_cli(capsys, "index")
    (data_root / "reference/wine-002/photo-1.jpg").unlink()

    exit_code, output = run_cli(capsys, "index")

    assert exit_code == 0
    assert (output["deleted"], output["reference_count"]) == (0, 2)
