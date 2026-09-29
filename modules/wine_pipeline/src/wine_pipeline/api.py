"""Synchronous organizer API and asynchronous polling share one job pipeline."""
from __future__ import annotations

import asyncio
import copy
import io
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import httpx
from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from .services import Catalog, Pipeline
from .feedback import FeedbackInput, FeedbackStore, normalize_slug
from .feedback_indexer import FeedbackIndexer
from .text_label import TextLabelMatcher
from .upload_limit import UploadLimit

logger = logging.getLogger(__name__)


@dataclass
class Job:
    job_id: str
    image: bytes = field(default=b"", repr=False)
    state: str = "queued"
    stage: str | None = None
    progress: float = 0
    result: dict | None = None
    error: dict | None = None
    completed: float | None = None
    task: asyncio.Task | None = field(default=None, repr=False)
    created: float = field(default_factory=time.monotonic, repr=False)
    history: list[dict] = field(default_factory=list)

    def record(self, stage: str, value: float) -> None:
        self.stage = stage
        self.progress = max(self.progress, value)
        self.history.append({
            "stage": stage, "progress": self.progress,
            "elapsed_ms": round((time.monotonic() - self.created) * 1000),
        })

    def snapshot(self):
        return copy.deepcopy({
            "job_id": self.job_id, "state": self.state, "stage": self.stage,
            "progress": self.progress, "poll_after_ms": 1000,
            "result": self.result, "error": self.error,
            "history": self.history,
        })


def validate_image(data: bytes) -> None:
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in {"JPEG", "PNG", "WEBP"}:
                raise HTTPException(415, "Supported images: JPEG, PNG, WebP")
            if image.width * image.height > 25_000_000:
                raise HTTPException(413, "Image exceeds 25 million pixels")
            image.verify()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(422, "Corrupted or unsupported image") from exc


def create_app(pipeline=None, *, capacity: int = 16, retention_seconds: float = 3600,
               feedback_store=None):
    if capacity <= 0 or retention_seconds <= 0:
        raise ValueError("Job capacity and retention must be positive")
    jobs: dict[str, Job] = {}
    semaphore = asyncio.Semaphore(1)
    tasks = set()
    max_bytes = 10 * 1024 * 1024
    indexer = None
    text_label_matcher = None

    @asynccontextmanager
    async def lifespan(app):
        nonlocal pipeline, feedback_store, indexer, text_label_matcher
        if pipeline is None:
            client = httpx.AsyncClient(timeout=httpx.Timeout(
                float(os.getenv("BRANCH_TIMEOUT_SECONDS", "180")), connect=10,
            ))
            pipeline = Pipeline(client, Catalog(os.environ["DATABASE_URL"]))
            if feedback_store is None:
                feedback_store = FeedbackStore(os.environ["DATABASE_URL"])
        else:
            client = None
        if feedback_store is not None:
            await feedback_store.initialize()
        index_task = None
        if client is not None and feedback_store is not None:
            indexer = FeedbackIndexer(feedback_store, client)
            text_label_matcher = TextLabelMatcher(client, pipeline.catalog)
            index_task = asyncio.create_task(indexer.run())
        yield
        if index_task is not None:
            index_task.cancel()
            await asyncio.gather(index_task, return_exceptions=True)
        for task in list(tasks):
            task.cancel()
        await asyncio.gather(*list(tasks), return_exceptions=True)
        if client is not None:
            await client.aclose()

    app = FastAPI(title="Wine recognition pipeline", lifespan=lifespan)
    app.add_middleware(UploadLimit)

    def prune():
        now = time.monotonic()
        expired = [identity for identity, job in jobs.items()
                   if job.completed is not None and now - job.completed > retention_seconds]
        for identity in expired:
            del jobs[identity]
        # Keep total memory bounded even during high turnover.
        finished = sorted(
            (job for job in jobs.values() if job.completed is not None),
            key=lambda job: job.completed,
        )
        while len(jobs) >= capacity and finished:
            del jobs[finished.pop(0).job_id]

    async def process(job, data, filename):
        try:
            async with semaphore:
                job.state = "processing"
                job.record("prepare", .1)

                def progress(stage, value):
                    job.record(stage, value)

                result = await pipeline.run(data, filename, progress)
                result["request_id"] = job.job_id
                job.result = result
                job.state = "done"
                job.record("done", 1)
        except asyncio.CancelledError:
            job.state = "failed"
            job.error = {"code": "cancelled", "message": "Service is stopping"}
            job.record("failed", job.progress)
            raise
        except Exception:
            logger.exception("Recognition job %s failed", job.job_id)
            job.state = "failed"
            job.error = {"code": "recognition_failed", "message": "Recognition unavailable"}
            job.record("failed", job.progress)
        finally:
            job.completed = time.monotonic()

    @app.post("/v1/eval/predict")
    async def predict(
        response: Response, image: UploadFile = File(...),
        wait: bool = Query(True, description="false: HTTP 202 and polling"),
    ):
        prune()
        if len(jobs) >= capacity:
            raise HTTPException(503, "Recognition queue is full", headers={"Retry-After": "5"})
        data = await image.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise HTTPException(413, "Image exceeds 10 MiB")
        if not data:
            raise HTTPException(422, "Image is empty")
        await asyncio.to_thread(validate_image, data)
        # Recheck after awaited validation, before inserting into the queue.
        prune()
        if len(jobs) >= capacity:
            raise HTTPException(503, "Recognition queue is full")
        identity = uuid.uuid4().hex
        job = Job(identity, image=data)
        jobs[identity] = job
        task = asyncio.create_task(process(job, data, image.filename or "image.jpg"))
        job.task = task
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        if not wait:
            return JSONResponse(
                {"job_id": identity, "state": "queued", "poll_after_ms": 1000},
                status_code=202, headers={"X-Job-ID": identity},
            )
        # Disconnecting an eval client does not cancel a queued job.
        await asyncio.shield(task)
        response.headers["X-Job-ID"] = identity
        if job.state != "done":
            raise HTTPException(503, job.error, headers={"X-Job-ID": identity})
        return {"slug": job.result["slug"]}

    @app.get("/image/status")
    async def status(job_id: str = Query(...)):
        # GET must not evict completed jobs merely because capacity is reached.
        now = time.monotonic()
        for identity in list(jobs):
            completed = jobs[identity].completed
            if completed is not None and now - completed > retention_seconds:
                del jobs[identity]
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Job not found or expired")
        return job.snapshot()

    @app.post("/feedback")
    async def feedback(payload: FeedbackInput):
        if feedback_store is None:
            raise HTTPException(503, "Feedback storage unavailable")
        job = jobs.get(payload.job_id)
        if (job is None or (job.completed is not None
                and time.monotonic() - job.completed > retention_seconds)):
            try:
                context = await feedback_store.context(payload.job_id)
            except Exception:
                logger.exception("Saved feedback context unavailable")
                raise HTTPException(503, "Feedback storage unavailable")
            if context is None:
                raise HTTPException(404, "Job not found or expired; recognize the photo again")
            image, result = context
        elif job.state != "done" or not job.result:
            raise HTTPException(409, "Recognition is not complete")
        else:
            image, result = job.image, job.result
        try:
            slug = normalize_slug(payload.correct_slug)
            predicted = result["slug"]
            label_match = None
            information = payload.wine_information
            has_information = information and any(
                value is not None and str(value).strip() for value in information.model_dump().values())
            if not slug and has_information:
                if text_label_matcher is None:
                    raise RuntimeError("Text label matcher unavailable")
                label_match = await text_label_matcher.resolve(information)
                if not label_match["accepted"]:
                    return {"saved": False, "label_match": label_match, "message":
                        "Не удалось однозначно определить вино по сведениям. Фото не сохранено. Уточните название или винодельню."}
                slug = label_match["slug"]
                if slug == predicted and not payload.is_correct:
                    return {"saved": False, "message":
                        "Сведения указывают на найденное вино. Если это другое вино, добавьте больше деталей; фото не сохранено."}
            if payload.is_correct:
                if slug and slug != predicted:
                    raise ValueError("Correct vote cannot name another wine")
                slug = predicted
            elif slug == predicted:
                raise ValueError("Correction must differ from prediction")
            elif not slug:
                existing = await feedback_store.context(payload.job_id)
                if existing is None:
                    return {"saved": False, "message":
                        "Расскажите, что знаете о вине: название, винодельню или текст этикетки. Фото пока не сохранено."}
            saved = await feedback_store.save(
                payload.job_id, image, result, payload.is_correct, slug,
                wine_information=information.model_dump() if has_information else None,
                label_match=label_match,
            )
            if indexer is not None:
                indexer.wake.set()
            return saved
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except Exception:
            logger.exception("Feedback persistence failed")
            raise HTTPException(503, "Feedback was not saved; retry later")

    @app.get("/feedback/status")
    async def feedback_status(job_id: str = Query(..., pattern=r"^[a-f0-9]{32}$")):
        if feedback_store is None:
            raise HTTPException(503, "Feedback storage unavailable")
        try:
            result = await feedback_store.status(job_id)
        except Exception:
            logger.exception("Feedback status unavailable")
            raise HTTPException(503, "Feedback status unavailable")
        if result is None:
            raise HTTPException(404, "Feedback not found")
        return result

    @app.get("/feedback/stats")
    async def feedback_stats():
        if feedback_store is None:
            raise HTTPException(503, "Feedback storage unavailable")
        try:
            return await feedback_store.stats()
        except Exception:
            logger.exception("Feedback statistics failed")
            raise HTTPException(503, "Feedback statistics unavailable")

    return app
