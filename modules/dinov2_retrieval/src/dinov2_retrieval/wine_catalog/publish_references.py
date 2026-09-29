"""Publish existing catalog vectors to the visual API, without re-embedding.

Canonical wine_id is MIN(wines.id) per slug. Per-wine hard links preserve
references shared by different wines despite reference_images.image_uri UNIQUE.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from .photo_manifest import read_photo_manifest

MODEL = "facebook/dinov2-with-registers-giant"


def publish(connection, data_root: Path, model_name: str = MODEL) -> dict[str, int]:
    root = data_root.resolve(strict=True)
    manifest = read_photo_manifest(root / "exports/dataset_photo_manifest.csv")
    rows = connection.execute("""
        SELECT id, slug, MIN(id) OVER (PARTITION BY slug),
               dataset_photo, dataset_photo_embedding, dataset_photo_embedding_model,
               web_photo, web_photo_embedding, web_photo_embedding_model
        FROM wines WHERE slug IS NOT NULL AND slug <> '' ORDER BY id
    """).fetchall()
    prepared = {}
    for row in rows:
        _, slug, canonical, *photos = row
        for kind, filename, vector, model in (
            ("dataset", *photos[:3]), ("web", *photos[3:]),
        ):
            if vector is None or model != model_name:
                continue
            if hasattr(vector, "to_numpy"):
                vector = vector.to_numpy()
            if len(vector) != 1536:
                raise ValueError(f"Wrong vector dimension for {slug}: {len(vector)}")
            relative = (
                Path("exports/dataset_photos") / manifest[filename]
                if kind == "dataset" else Path("web_photos") / Path(filename).name
            )
            source = (root / relative).resolve(strict=True)
            if not source.is_relative_to(root) or not source.is_file():
                raise ValueError(f"Reference escapes DATA_ROOT: {relative}")
            digest = hashlib.sha256(relative.as_posix().encode()).hexdigest()[:16]
            target = root / "reference/catalog" / str(canonical) / (
                f"{kind}-{digest}{source.suffix.lower()}"
            )
            prepared[str(target)] = (str(canonical), slug, source, target, vector)
    if not prepared:
        raise ValueError("No compatible catalog image vectors; refusing empty publication")
    for wine_id, slug, source, target, vector in prepared.values():
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
        # A repeat run must never pair a new vector with an old/different image.
        if target.stat().st_size != source.stat().st_size or (
            not os.path.samefile(source, target)
            and hashlib.sha256(target.read_bytes()).digest()
            != hashlib.sha256(source.read_bytes()).digest()
        ):
            raise ValueError(f"Published reference differs from source: {target}")
        connection.execute("""
            INSERT INTO reference_images (wine_id, slug, image_uri, model_name, embedding)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (image_uri) DO UPDATE SET
                wine_id=EXCLUDED.wine_id, slug=EXCLUDED.slug,
                model_name=EXCLUDED.model_name, embedding=EXCLUDED.embedding,
                updated_at=NOW()
        """, (wine_id, slug, str(target), model_name, vector))
    connection.commit()
    return {
        "reference_images": len(prepared),
        "distinct_slugs": len({item[1] for item in prepared.values()}),
    }


def main() -> None:
    import psycopg
    from pgvector.psycopg import register_vector

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/data"))
    args = parser.parse_args()
    with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
        register_vector(connection)
        print(json.dumps(publish(connection, args.data_root)))


if __name__ == "__main__":
    main()
