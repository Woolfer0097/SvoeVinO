import hashlib
import asyncio
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from wine_pipeline.api import create_app
from wine_pipeline.feedback import normalize_slug
from wine_pipeline.fusion import fuse
from wine_pipeline.text_quality import assess_text
from wine_pipeline.services import Pipeline
from test_pipeline import FakePipeline, FakeCatalog, candidates, image_bytes, prepared_payload


@pytest.mark.parametrize("text,confidence,reason", [
    ("1298 2 r фанагория λ∑ ки", .99, "too_little_distinctive_text"),
    ("Фанагория сухое вино белое", .99, "too_little_distinctive_text"),
    ("Каберне Совиньон Аратти 2021", .92, "sufficient_text"),
    ("Каберне Совиньон Аратти", .6, "low_recognition_confidence"),
    ("Каберне Совиньон Аратти", None, "recognition_confidence_missing"),
])
def test_adaptive_weight_uses_distinctive_text_and_recognition_confidence(text, confidence, reason):
    result = assess_text({"text": text, "text_confidence": confidence},
                         {"a": {"winery": "Фанагория"}, "b": {"winery": "Аратти"}}, available=True)
    assert result["reason"] == reason
    assert result["sufficient"] == (reason == "sufficient_text")


def test_text_primary_can_rescue_a_hit_without_geometry_and_changes_winner():
    visual, ocr = {"alpha": 1}, {"beta": 1}
    high, mode = fuse(candidates(), visual, ocr, ocr_weight=.7, text_primary=True)
    low, _ = fuse(candidates(), visual, ocr, ocr_weight=.3)
    assert mode == "text_primary"
    assert high[0]["slug"] == "beta"
    assert high[0]["score"] == .7
    assert low[0]["slug"] == "alpha"


def test_strong_text_survives_verifier_failure_using_dino_as_visual_signal():
    result, mode = fuse(candidates(), {}, {"beta": 1}, ocr_weight=.7, text_primary=True)
    assert mode == "text_primary_dino"
    assert result[0]["slug"] == "beta"
    assert result[0]["visual_score"] is None


@pytest.mark.parametrize("text,confidence,weight", [
    ("Каберне Совиньон Аратти", .94, .7), ("Аратти 1298", .99, .3),
])
def test_pipeline_reports_the_weights_actually_used(text, confidence, weight, monkeypatch):
    monkeypatch.setenv("COLOR_WEIGHT", "0")
    class Cards(FakeCatalog):
        async def cards(self, slugs):
            return {"alpha": {"winery": "Аратти"}}
    async def run():
        def handler(request):
            if request.url.path == "/preprocess":
                payload = prepared_payload()
            elif request.url.path == "/search":
                payload = {"candidates": candidates(), "model_name": "giant", "query_embedding_dimension": 1536}
            elif request.url.path == "/match":
                payload = {"top_10": {"11": .9}, "evidence": {"text": text, "text_confidence": confidence}}
            else:
                payload = [{"id": "1", "score": .8}]
            return httpx.Response(200, json=payload)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await Pipeline(client, Cards()).run(image_bytes(), "label.png", lambda *args: None)
        assert result["fusion_weights"]["ocr"] == weight
        assert result["text_quality"]["sufficient"] == (weight == .7)
    asyncio.run(run())


class MemoryFeedback:
    def __init__(self):
        self.records = {}

    async def initialize(self):
        pass

    async def save(self, job_id, image, result, correct, slug):
        if slug and slug not in {"alpha", "beta"}:
            raise ValueError("Wine slug not found in catalog")
        self.records[job_id] = (image, result, correct, slug)
        return {"saved": True, "job_id": job_id, "image_sha256": hashlib.sha256(image).hexdigest(),
                "is_correct": correct, "correct_slug": slug}

    async def stats(self):
        return {"total": len(self.records)}


def recognize(client):
    response = client.post("/v1/eval/predict", files={"image": ("label.png", image_bytes())})
    assert response.status_code == 200
    return response.headers["X-Job-ID"]


def test_feedback_keeps_authoritative_prediction_image_and_retry_is_one_vote():
    store = MemoryFeedback()
    with TestClient(create_app(FakePipeline(), feedback_store=store)) as client:
        identity = recognize(client)
        body = {"job_id": identity, "is_correct": False,
                "correct_slug": "https://vino-svoe.ru/wines/beta"}
        for _ in range(2):
            response = client.post("/feedback", json=body)
            assert response.status_code == 200
            assert response.json()["correct_slug"] == "beta"
        assert client.get("/feedback/stats").json() == {"total": 1}
        image, prediction, correct, slug = store.records[identity]
        assert image == image_bytes() and prediction["slug"] == "alpha"
        assert not correct and slug == "beta"
        body.update(is_correct=True, correct_slug=None)
        assert client.post("/feedback", json=body).json()["correct_slug"] == "alpha"


@pytest.mark.parametrize("correct,slug", [(False, "alpha"), (True, "beta"), (False, "unknown"),
                                         (False, "https://evil.test/wines/beta")])
def test_feedback_rejects_invalid_or_contradictory_labels(correct, slug):
    store = MemoryFeedback()
    with TestClient(create_app(FakePipeline(), feedback_store=store)) as client:
        identity = recognize(client)
        assert client.post("/feedback", json={"job_id": identity, "is_correct": correct,
                                               "correct_slug": slug}).status_code == 422
        assert not store.records


def test_feedback_rejects_forged_result_and_allows_unknown_correction():
    store = MemoryFeedback()
    with TestClient(create_app(FakePipeline(), feedback_store=store)) as client:
        identity = recognize(client)
        body = {"job_id": identity, "is_correct": False}
        assert client.post("/feedback", json={**body, "predicted_slug": "beta"}).status_code == 422
        assert client.post("/feedback", json=body).status_code == 200
        assert store.records[identity][-1] is None
        assert client.post("/feedback", json={**body, "is_correct": "false"}).status_code == 422


def test_feedback_for_running_or_expired_job_is_not_saved():
    store = MemoryFeedback()
    with TestClient(create_app(FakePipeline(delay=.2), retention_seconds=.01, feedback_store=store)) as client:
        response = client.post("/v1/eval/predict?wait=false", files={"image": ("label.png", image_bytes())})
        body = {"job_id": response.json()["job_id"], "is_correct": True}
        assert client.post("/feedback", json=body).status_code == 409
        time.sleep(.3)
        assert client.post("/feedback", json=body).status_code == 404
        assert not store.records


def test_storage_failure_is_visible_and_does_not_leak_connection_details():
    class Broken(MemoryFeedback):
        async def save(self, *args):
            raise RuntimeError("postgres password secret")
    with TestClient(create_app(FakePipeline(), feedback_store=Broken())) as client:
        identity = recognize(client)
        response = client.post("/feedback", json={"job_id": identity, "is_correct": True})
        assert response.status_code == 503 and "secret" not in response.text


@pytest.mark.parametrize("value", ["../beta", "beta?x=1", "https://vino-svoe.ru.evil/wines/beta",
                                   "https://vino-svoe.ru/wines/beta/extra"])
def test_feedback_url_never_fetches_arbitrary_remote_resources(value):
    with pytest.raises(ValueError):
        normalize_slug(value)
