"""Add confirmed user photos to the same giant/1536 search index on GPU."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import math
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)
MODEL = "facebook/dinov2-with-registers-giant"
URI_ROOT = "/data/reference/feedback/"


class FeedbackIndexer:
    def __init__(self, store, client):
        self.store, self.client = store, client
        self.root = Path(os.getenv("FEEDBACK_REFERENCE_DIR", "/feedback-references"))
        self.dino = os.getenv("DINO_URL", "http://dinov2:8000").rstrip("/")
        self.preprocessing = os.getenv("PREPROCESS_URL", "http://preprocessing:8000").rstrip("/")
        self.wake = asyncio.Event()

    async def run(self):
        while True:
            self.wake.clear()
            try:
                while await self.process_next():
                    pass
            except Exception:
                logger.exception("Cannot process feedback queue")
            try:
                await asyncio.wait_for(self.wake.wait(), 30)
            except TimeoutError:
                pass

    async def process_next(self):
        async with await self.store.connect() as connection:
            cursor = await connection.execute("""
                SELECT f.job_id,f.correct_slug,f.label_revision,s.image_bytes
                FROM recognition_feedback f JOIN recognition_samples s USING (image_sha256)
                WHERE f.embedding_state='queued' ORDER BY f.updated_at
                LIMIT 1 FOR UPDATE OF f SKIP LOCKED
            """)
            row = await cursor.fetchone()
            if row is None:
                return False
            job_id, slug, revision, raw = row
            await connection.execute("""
                UPDATE recognition_feedback SET embedding_state='indexing',embedding_error=NULL
                WHERE job_id=%s
            """, (job_id,))
        try:
            uri, vector = None, None
            canonical = None
            if slug:
                async with await self.store.connect() as connection:
                    cursor = await connection.execute("SELECT min(id) FROM wines WHERE slug=%s", (slug,))
                    canonical = (await cursor.fetchone())[0]
                if canonical is None:
                    raise ValueError("Label no longer exists in catalog")
                prepared = await self.client.post(self.preprocessing + "/preprocess",
                    files={"image": ("feedback.image", bytes(raw), "application/octet-stream")})
                prepared.raise_for_status()
                image = base64.b64decode(prepared.json()["embedding_png"], validate=True)
                if not image or len(image) > 15 * 1024 * 1024:
                    raise ValueError("Invalid prepared feedback image")
                digest = hashlib.sha256(image).hexdigest()
                relative = Path(str(canonical)) / (digest + ".png")
                await asyncio.to_thread(self._write_image, relative, image, digest)
                uri = URI_ROOT + relative.as_posix()
                embedded = await self.client.post(self.dino + "/embed", json={"image_uri": uri})
                embedded.raise_for_status()
                result = embedded.json()
                vector = result["embedding"]
                if (result["model"] != MODEL or result["dimension"] != 1536
                        or len(vector) != 1536 or any(not math.isfinite(float(v)) for v in vector)
                        or not any(float(v) != 0 for v in vector)):
                    raise ValueError("Feedback embedding is incompatible with giant/1536")
            await self._publish(job_id, slug, revision, canonical, uri, vector)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Cannot index feedback %s", job_id)
            async with await self.store.connect() as connection:
                await connection.execute("""
                    UPDATE recognition_feedback SET embedding_state='failed',
                        embedding_error='Reference indexing failed; submit feedback again to retry'
                    WHERE job_id=%s AND label_revision=%s AND embedding_state='indexing'
                """, (job_id, revision))
        return True

    def _write_image(self, relative, image, digest):
        root = self.root.resolve(strict=True)
        target = (root / relative).resolve()
        if not target.is_relative_to(root):
            raise ValueError("Feedback path escapes reference directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise ValueError("Existing feedback reference differs")
        else:
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".feedback-", delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(image)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, target)
            finally:
                if temporary is not None and temporary.exists():
                    temporary.unlink()

    async def _publish(self, job_id, slug, revision, canonical, uri, vector):
        async with await self.store.connect() as connection:
            cursor = await connection.execute("""
                SELECT label_revision,reference_uri FROM recognition_feedback
                WHERE job_id=%s FOR UPDATE
            """, (job_id,))
            current, old_uri = await cursor.fetchone()
            # The user may have corrected a label while inference was running.
            if current != revision:
                return
            if uri:
                cursor = await connection.execute("""
                    SELECT 1 FROM recognition_feedback other JOIN recognition_feedback current
                    ON other.image_sha256=current.image_sha256
                    WHERE current.job_id=%s AND other.job_id<>current.job_id
                      AND other.embedding_state='ready' AND other.correct_slug<>%s LIMIT 1
                """, (job_id, slug))
                if await cursor.fetchone():
                    raise ValueError("Same photo has conflicting confirmed wine labels")
                await connection.execute("""
                    INSERT INTO reference_images (wine_id,slug,image_uri,model_name,embedding)
                    VALUES (%s,%s,%s,%s,%s::vector)
                    ON CONFLICT (image_uri) DO UPDATE SET embedding=EXCLUDED.embedding,
                        model_name=EXCLUDED.model_name,updated_at=now()
                """, (str(canonical), slug, uri, MODEL, json.dumps(vector)))
            await connection.execute("""
                UPDATE recognition_feedback SET embedding_state=%s,indexed_slug=%s,
                    reference_uri=%s,embedding_error=NULL,review_status=%s WHERE job_id=%s
            """, ('ready' if uri else 'unlabeled', slug, uri,
                  'approved' if uri else 'pending', job_id))
            # Retract only feedback-generated references, preserving original
            # catalog photos and references still used by another feedback row.
            if old_uri and old_uri != uri and old_uri.startswith(URI_ROOT):
                await connection.execute("""
                    DELETE FROM reference_images r WHERE r.image_uri=%s
                    AND NOT EXISTS (SELECT 1 FROM recognition_feedback f WHERE f.reference_uri=r.image_uri)
                """, (old_uri,))
        logger.info("Feedback %s reference state: %s", job_id, 'ready' if uri else 'unlabeled')
