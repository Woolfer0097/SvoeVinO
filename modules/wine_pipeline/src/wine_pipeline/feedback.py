"""Persist user labels and a durable queue for reference-image enrichment."""
from __future__ import annotations

import hashlib
import re
from urllib.parse import urlsplit

import psycopg
from psycopg.types.json import Jsonb
from pydantic import BaseModel, ConfigDict, Field, StrictBool


class WineInformation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(default="", max_length=200)
    year: int | None = Field(default=None, ge=1900, le=2099)
    locality: str = Field(default="", max_length=200)
    winery: str = Field(default="", max_length=200)
    notes: str = Field(default="", max_length=1500)


class FeedbackInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    is_correct: StrictBool
    correct_slug: str | None = Field(default=None, max_length=500)
    wine_information: WineInformation | None = None


def normalize_slug(value: str | None) -> str | None:
    value = (value or "").strip()
    if not value:
        return None
    if "://" in value:
        parsed = urlsplit(value)
        if (parsed.scheme != "https" or parsed.netloc != "vino-svoe.ru"
                or parsed.query or parsed.fragment or not parsed.path.startswith("/wines/")):
            raise ValueError("Use a slug or https://vino-svoe.ru/wines/<slug>")
        value = parsed.path.removeprefix("/wines/").rstrip("/")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,299}", value):
        raise ValueError("Invalid wine slug")
    return value


class FeedbackStore:
    def __init__(self, database_url: str):
        self.database_url = database_url

    async def connect(self):
        return await psycopg.AsyncConnection.connect(
            self.database_url, connect_timeout=5, options="-c statement_timeout=10000",
        )

    async def initialize(self):
        async with await self.connect() as connection:
            await connection.execute("""
                CREATE TABLE IF NOT EXISTS recognition_samples (
                    image_sha256 text PRIMARY KEY,
                    image_bytes bytea NOT NULL,
                    created_at timestamptz NOT NULL DEFAULT now()
                )
            """)
            await connection.execute("""
                CREATE TABLE IF NOT EXISTS recognition_feedback (
                    job_id text PRIMARY KEY,
                    image_sha256 text NOT NULL REFERENCES recognition_samples(image_sha256),
                    predicted_slug text NOT NULL,
                    is_correct boolean NOT NULL,
                    correct_slug text,
                    prediction jsonb NOT NULL,
                    review_status text NOT NULL DEFAULT 'pending'
                        CHECK (review_status IN ('pending', 'approved', 'rejected')),
                    created_at timestamptz NOT NULL DEFAULT now(),
                    updated_at timestamptz NOT NULL DEFAULT now()
                )
            """)
            await connection.execute("""
                ALTER TABLE recognition_feedback
                    ADD COLUMN IF NOT EXISTS embedding_state text NOT NULL DEFAULT 'unlabeled',
                    ADD COLUMN IF NOT EXISTS embedding_error text,
                    ADD COLUMN IF NOT EXISTS reference_uri text,
                    ADD COLUMN IF NOT EXISTS indexed_slug text,
                    ADD COLUMN IF NOT EXISTS wine_information jsonb,
                    ADD COLUMN IF NOT EXISTS label_match jsonb,
                    ADD COLUMN IF NOT EXISTS label_revision integer NOT NULL DEFAULT 0
            """)
            # This application runs one enrichment worker. Recover unfinished
            # work on restart, including labeled feedback collected previously.
            await connection.execute("""
                UPDATE recognition_feedback SET embedding_state='queued'
                WHERE embedding_state='indexing'
                   OR (embedding_state='unlabeled' AND correct_slug IS NOT NULL)
            """)

    async def save(self, job_id: str, image: bytes, result: dict,
                   is_correct: bool, correct_slug: str | None, *, wine_information=None, label_match=None):
        digest = hashlib.sha256(image).hexdigest()
        async with await self.connect() as connection:
            cursor = await connection.execute("""
                SELECT correct_slug,reference_uri FROM recognition_feedback
                WHERE job_id=%s FOR UPDATE
            """, (job_id,))
            previous = await cursor.fetchone()
            if correct_slug:
                cursor = await connection.execute(
                    "SELECT 1 FROM wines WHERE slug = %s LIMIT 1", (correct_slug,),
                )
                if await cursor.fetchone() is None:
                    raise ValueError("Wine slug not found in catalog")
            await connection.execute("""
                INSERT INTO recognition_samples (image_sha256, image_bytes)
                VALUES (%s, %s) ON CONFLICT DO NOTHING
            """, (digest, image))
            # Retrying the same vote does not inflate statistics. Changing a
            # label invalidates approval, but an identical retry preserves it.
            await connection.execute("""
                INSERT INTO recognition_feedback
                    (job_id,image_sha256,predicted_slug,is_correct,correct_slug,prediction,embedding_state,
                     wine_information,label_match)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (job_id) DO UPDATE SET
                    is_correct=EXCLUDED.is_correct, correct_slug=EXCLUDED.correct_slug,
                    review_status=CASE
                        WHEN recognition_feedback.is_correct=EXCLUDED.is_correct
                         AND recognition_feedback.correct_slug IS NOT DISTINCT FROM EXCLUDED.correct_slug
                        THEN recognition_feedback.review_status ELSE 'pending' END,
                    label_revision=recognition_feedback.label_revision + CASE
                        WHEN recognition_feedback.correct_slug IS DISTINCT FROM EXCLUDED.correct_slug
                        THEN 1 ELSE 0 END,
                    embedding_state=CASE
                        WHEN recognition_feedback.correct_slug IS DISTINCT FROM EXCLUDED.correct_slug
                          OR recognition_feedback.embedding_state='failed'
                        THEN 'queued' ELSE recognition_feedback.embedding_state END,
                    embedding_error=NULL,
                    wine_information=EXCLUDED.wine_information,
                    label_match=EXCLUDED.label_match,
                    updated_at=now()
                RETURNING embedding_state
            """, (job_id, digest, result["slug"], is_correct, correct_slug, Jsonb(result),
                  'queued' if correct_slug else 'unlabeled', Jsonb(wine_information), Jsonb(label_match)))
            cursor = await connection.execute(
                "SELECT embedding_state FROM recognition_feedback WHERE job_id=%s", (job_id,),
            )
            state = (await cursor.fetchone())[0]
            if previous and previous[0] != correct_slug and previous[1]:
                old_uri = previous[1]
                await connection.execute("""
                    UPDATE recognition_feedback SET reference_uri=NULL,indexed_slug=NULL WHERE job_id=%s
                """, (job_id,))
                if old_uri.startswith("/data/reference/feedback/"):
                    await connection.execute("""
                        DELETE FROM reference_images r WHERE r.image_uri=%s
                        AND NOT EXISTS (SELECT 1 FROM recognition_feedback f WHERE f.reference_uri=r.image_uri)
                    """, (old_uri,))
        return {"saved": True, "job_id": job_id, "image_sha256": digest,
                "is_correct": is_correct, "correct_slug": correct_slug, "embedding_state": state,
                "label_match": label_match}

    async def status(self, job_id: str):
        async with await self.connect() as connection:
            cursor = await connection.execute("""
                SELECT embedding_state, correct_slug, indexed_slug, reference_uri
                FROM recognition_feedback WHERE job_id=%s
            """, (job_id,))
            row = await cursor.fetchone()
        return None if row is None else dict(zip(
            ("embedding_state", "correct_slug", "indexed_slug", "reference_uri"), row,
        ))

    async def context(self, job_id: str):
        """Existing feedback can be corrected even after in-memory jobs expire."""
        async with await self.connect() as connection:
            cursor = await connection.execute("""
                SELECT s.image_bytes,f.prediction FROM recognition_feedback f
                JOIN recognition_samples s USING (image_sha256) WHERE f.job_id=%s
            """, (job_id,))
            row = await cursor.fetchone()
        return None if row is None else (bytes(row[0]), row[1])

    async def stats(self):
        async with await self.connect() as connection:
            cursor = await connection.execute("""
                SELECT count(*), count(*) FILTER (WHERE is_correct),
                    count(*) FILTER (WHERE NOT is_correct),
                    count(*) FILTER (WHERE correct_slug IS NOT NULL),
                    count(*) FILTER (WHERE review_status='pending'),
                    count(*) FILTER (WHERE embedding_state='ready'),
                    count(*) FILTER (WHERE embedding_state IN ('queued','indexing')),
                    count(*) FILTER (WHERE embedding_state='failed')
                FROM recognition_feedback
            """)
            row = await cursor.fetchone()
        return dict(zip(("total", "correct", "incorrect", "labeled", "pending_review",
                         "indexed", "indexing", "index_failed"), row))
