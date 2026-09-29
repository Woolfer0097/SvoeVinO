import asyncio
import base64
import io
import json
import math
import time

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from wine_pipeline.api import create_app
from wine_pipeline.fusion import fuse, scores_by_id
from wine_pipeline.services import Pipeline


def candidates():
    return [
        {"wine_id": "1", "slug": "alpha", "score": .9, "distance": .1, "best_image_uri": "/a"},
        {"wine_id": "2", "slug": "beta", "score": .8, "distance": .2, "best_image_uri": "/b"},
    ]


def image_bytes():
    stream = io.BytesIO()
    Image.new("RGB", (12, 12), "white").save(stream, format="PNG")
    return stream.getvalue()


def prepared_payload():
    encoded = base64.b64encode(image_bytes()).decode("ascii")
    return {"embedding_png": encoded, "ocr_png": encoded, "metadata": {"schema_version": "1.0"}}


def test_intersection_weights_and_no_fake_distance():
    result, mode = fuse(candidates(), {"alpha": .8, "beta": .9}, {"alpha": .95, "beta": .5})
    assert mode == "weighted_visual"
    assert result[0]["slug"] == "alpha"
    assert result[0]["score"] == pytest.approx(.5 * .8 + .5 * .95)
    assert "distance" not in result[0]


def test_sparse_text_does_not_drop_a_visual_candidate_missing_from_ocr():
    result, mode = fuse(candidates(), {"alpha": .8, "beta": .9}, {"alpha": .8})
    assert [item["slug"] for item in result] == ["alpha", "beta"]
    assert mode == "weighted_visual"


@pytest.mark.parametrize("visual,ocr,expected,mode", [
    ({"beta": .8}, {"outside": 1}, "beta", "visual_fallback"),
    ({}, {"beta": 1}, "alpha", "dino_fallback"),
    ({"alpha": 0, "beta": 0}, {"beta": 1}, "alpha", "dino_fallback"),
])
def test_explicit_fallbacks(visual, ocr, expected, mode):
    result, actual = fuse(candidates(), visual, ocr)
    assert actual == mode
    assert result[0]["slug"] == expected


def test_ties_use_dino_order():
    for _ in range(4):
        result, _ = fuse(candidates(), {"beta": 1, "alpha": 1}, {"beta": 1, "alpha": 1})
        assert result[0]["slug"] == "alpha"


@pytest.mark.parametrize("score", [math.nan, math.inf, -.01, 1.01])
def test_reject_invalid_scores(score):
    with pytest.raises(ValueError):
        scores_by_id([{"id": "1", "score": score}])


def test_unknown_verifier_id_rejected():
    with pytest.raises(ValueError):
        scores_by_id([{"id": "9", "score": 1}], {"1"})


def test_verifier_top_ten_dedup():
    items = [{"id": str(i), "score": i / 20} for i in range(20)]
    items += [{"id": "19", "score": 1}]
    result = scores_by_id(items)
    assert len(result) == 10
    assert result["19"] == 1


class FakePipeline:
    def __init__(self, fail=False, delay=.01):
        self.fail, self.delay = fail, delay

    async def run(self, data, filename, progress):
        progress("rank", .6)
        await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("secret internal connection string")
        return {"slug": "alpha", "candidates": candidates()}


def test_sync_eval_returns_only_top_one_and_status():
    with TestClient(create_app(FakePipeline())) as client:
        response = client.post("/v1/eval/predict", files={"image": ("label.png", image_bytes())})
        assert response.status_code == 200
        assert response.json() == {"slug": "alpha"}
        identity = response.headers["X-Job-ID"]
        status = client.get("/image/status", params={"job_id": identity}).json()
        assert status["state"] == "done"
        assert status["result"]["request_id"] == identity
        assert status["progress"] == 1
        assert [event["stage"] for event in status["history"]] == ["prepare", "rank", "done"]
        assert all(event["elapsed_ms"] >= 0 for event in status["history"])


def test_async_mode_polling():
    with TestClient(create_app(FakePipeline(delay=.03))) as client:
        response = client.post("/v1/eval/predict?wait=false", files={"image": ("label.png", image_bytes())})
        assert response.status_code == 202
        identity = response.json()["job_id"]
        for _ in range(30):
            status = client.get("/image/status", params={"job_id": identity}).json()
            if status["state"] == "done":
                break
            time.sleep(.01)
        assert status["state"] == "done"
        assert status["result"]["slug"] == "alpha"


def test_failed_job_and_no_exception_leak():
    with TestClient(create_app(FakePipeline(fail=True))) as client:
        response = client.post("/v1/eval/predict", files={"image": ("label.png", image_bytes())})
        assert response.status_code == 503
        assert "secret" not in response.text
        status = client.get("/image/status", params={"job_id": response.headers["X-Job-ID"]}).json()
        assert status["state"] == "failed"
        assert status["error"]["code"] == "recognition_failed"
        assert status["history"][-1]["stage"] == "failed"


@pytest.mark.parametrize("data", [b"", b"not an image"])
def test_invalid_image(data):
    with TestClient(create_app(FakePipeline())) as client:
        assert client.post("/v1/eval/predict", files={"image": ("a.png", data)}).status_code == 422


def test_wrong_field_and_unknown_status():
    with TestClient(create_app(FakePipeline())) as client:
        assert client.post("/v1/eval/predict", files={"file": ("a.png", image_bytes())}).status_code == 422
        assert client.get("/image/status", params={"job_id": "unknown"}).status_code == 404


def test_large_body_rejected_before_multipart_parse():
    with TestClient(create_app(FakePipeline())) as client:
        response = client.post("/v1/eval/predict", content=b"x",
                               headers={"Content-Length": str(20 * 1024 * 1024)})
        assert response.status_code == 413


def test_chunked_body_limit_without_content_length():
    async def run():
        app = create_app(FakePipeline())
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test",
            ) as client:
                async def chunks():
                    for _ in range(12):
                        yield b"x" * 1024 * 1024
                response = await client.post("/v1/eval/predict", content=chunks())
                assert response.status_code == 413
    asyncio.run(run())


def test_queue_is_bounded():
    with TestClient(create_app(FakePipeline(delay=.2), capacity=1)) as client:
        assert client.post("/v1/eval/predict?wait=false", files={"image": ("a.png", image_bytes())}).status_code == 202
        response = client.post("/v1/eval/predict?wait=false", files={"image": ("a.png", image_bytes())})
        assert response.status_code == 503


def test_expired_job():
    with TestClient(create_app(FakePipeline(), retention_seconds=.01)) as client:
        response = client.post("/v1/eval/predict", files={"image": ("a.png", image_bytes())})
        time.sleep(.03)
        assert client.get("/image/status", params={"job_id": response.headers["X-Job-ID"]}).status_code == 404


class FakeCatalog:
    async def references(self, slugs, model_name):
        return []

    async def cards(self, slugs):
        return {}

    async def slugs(self, ids):
        return {"11": "alpha", "12": "beta", "21": "alpha"}


@pytest.mark.parametrize("broken", [None, "ocr", "superpoint", "dino"])
def test_real_http_adapters_parallelism_and_failure_policy(broken, monkeypatch):
    monkeypatch.setenv("COLOR_WEIGHT", "0")
    async def run():
        called = []
        ocr_started = asyncio.Event()

        async def handler(request):
            path = request.url.path
            called.append(path)
            assert b"filename=" in request.content
            if path == "/preprocess":
                assert b'name="image"' in request.content
                return httpx.Response(200, json=prepared_payload())
            if path == "/match":
                ocr_started.set()
                if broken == "ocr":
                    return httpx.Response(503)
                return httpx.Response(200, json={"top_10": {"11": .9, "21": .8, "12": .4}})
            if path == "/search":
                # This would deadlock if OCR did not start in parallel.
                await asyncio.wait_for(ocr_started.wait(), .3)
                if broken == "dino":
                    return httpx.Response(503)
                assert b'name="file"' in request.content
                return httpx.Response(200, json={
                    "candidates": candidates(), "model_name": "giant",
                    "query_embedding_dimension": 1536,
                })
            assert path == "/verify"
            assert b'name="query"' in request.content
            assert b'["1", "2"]' in request.content
            if broken == "superpoint":
                return httpx.Response(503)
            return httpx.Response(200, json=[{"id": "1", "score": .8}, {"id": "2", "score": .7}])

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            pipeline = Pipeline(client, FakeCatalog())
            if broken == "dino":
                with pytest.raises(RuntimeError, match="retrieval"):
                    await pipeline.run(image_bytes(), "a.png", lambda *args: None)
            else:
                result = await pipeline.run(image_bytes(), "a.png", lambda *args: None)
                assert result["slug"] == "alpha"
                assert result["branch_counts"]["dino"] == 2
                assert result["fusion_mode"] == (
                    "dino_fallback" if broken == "superpoint" else
                    "visual_fallback" if broken == "ocr" else "weighted_visual"
                )
                assert bool(result["warnings"]) == bool(broken)
            assert "/match" in called and "/search" in called
    asyncio.run(run())


def test_ocr_candidate_outside_dino_requires_geometric_verification(monkeypatch):
    monkeypatch.setenv("COLOR_WEIGHT", "0")

    class RescueCatalog(FakeCatalog):
        async def slugs(self, ids):
            return {"30": "gamma"}

        async def references(self, slugs, model_name):
            assert slugs == ["gamma"]
            return [{"wine_id": "3", "slug": "gamma", "score": None,
                     "best_image_uri": "/c", "retrieval_sources": ["ocr"]}]

    async def run():
        calls = []

        async def handler(request):
            if request.url.path == "/preprocess":
                return httpx.Response(200, json=prepared_payload())
            if request.url.path == "/match":
                return httpx.Response(200, json={"top_10": {"30": .9}})
            if request.url.path == "/search":
                return httpx.Response(200, json={"candidates": candidates(),
                    "model_name": "giant", "query_embedding_dimension": 1536})
            calls.append(request.content)
            if b'["3"]' in request.content:
                return httpx.Response(200, json=[{"id": "3", "score": .95}])
            return httpx.Response(200, json=[{"id": "1", "score": .8}])

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await Pipeline(client, RescueCatalog()).run(image_bytes(), "a.png", lambda *args: None)
        assert len(calls) == 2
        assert result["slug"] == "gamma"
        assert result["branch_counts"]["dino"] == 2
        assert result["branch_counts"]["ocr_rescue"] == 1
        assert result["candidates"][0]["dino_score"] is None
    asyncio.run(run())


def test_ocr_only_candidate_cannot_become_dino_fallback():
    expanded = candidates() + [{"slug": "gamma", "score": None}]
    result, mode = fuse(expanded, {}, {"gamma": .99})
    assert mode == "dino_fallback"
    assert result[0]["slug"] == "alpha"
    assert all(candidate["slug"] != "gamma" for candidate in result)


def test_preprocessing_failure_does_not_silently_use_full_resolution(monkeypatch):
    monkeypatch.setenv("COLOR_WEIGHT", "0")
    async def run():
        def handler(request):
            assert request.url.path == "/preprocess"
            return httpx.Response(503)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(httpx.HTTPStatusError):
                await Pipeline(client, FakeCatalog()).run(image_bytes(), "a.png", lambda *args: None)
    asyncio.run(run())
