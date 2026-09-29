"""Export labeled feedback for review or subsequent reference-image ingestion."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path

import psycopg
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--approved-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "feedback.jsonl"
    # Refuse to accidentally overwrite an existing review/export manifest.
    with psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=5,
                          options="-c default_transaction_read_only=on") as connection:
        with manifest.open("x", encoding="utf-8") as stream:
            count = 0
            with connection.cursor(name="feedback_export") as cursor:
                cursor.execute("""
                    SELECT f.job_id, f.image_sha256, f.predicted_slug, f.correct_slug,
                        f.review_status, s.image_bytes
                    FROM recognition_feedback f JOIN recognition_samples s USING (image_sha256)
                    WHERE f.correct_slug IS NOT NULL AND f.review_status <> 'rejected'
                      AND (%s = false OR f.review_status = 'approved')
                    ORDER BY f.created_at, f.job_id
                """, (args.approved_only,))
                for job_id, digest, predicted, correct, status, raw in cursor:
                    data = bytes(raw)
                    if hashlib.sha256(data).hexdigest() != digest:
                        raise ValueError("Stored sample hash mismatch")
                    with Image.open(io.BytesIO(data)) as source:
                        extension = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}[source.format]
                    name = digest + extension
                    image = args.output / name
                    if image.exists():
                        if hashlib.sha256(image.read_bytes()).hexdigest() != digest:
                            raise ValueError("Export path contains another image")
                    else:
                        with image.open("xb") as target:
                            target.write(data)
                    stream.write(json.dumps({"job_id": job_id, "image_path": name,
                        "image_sha256": digest, "predicted_slug": predicted,
                        "correct_slug": correct, "review_status": status}, ensure_ascii=False) + "\n")
                    count += 1
    print(json.dumps({"exported": count, "manifest": str(manifest)}))


if __name__ == "__main__":
    main()
