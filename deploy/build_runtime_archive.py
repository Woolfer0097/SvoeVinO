"""Export a sanitized reviewer runtime; never modify the active database."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

EXCLUDED_DATA = (
    "recognition_samples", "recognition_feedback", "wine_embedding_jobs",
    "wine_scrape_jobs",
)

RUNTIME_README = """# SvoeVinO: runtime для проверяющих

Архив содержит только каталог PostgreSQL и публичные эталонные фотографии.
Распакуйте его в modules/dinov2_retrieval/data/ клона SvoeVinO.
Исходный код, Docker-образы и веса моделей в архив НЕ включены.
Первый запуск требует интернета для скачивания образов и весов.

Из корня актуального репозитория:

```bash
mkdir -p modules/dinov2_retrieval/model-cache
mkdir -p modules/dinov2_retrieval/data/reference/feedback
cp modules/dinov2_retrieval/.env.example modules/dinov2_retrieval/.env
docker compose -f modules/dinov2_retrieval/docker-compose.yml up -d postgres
bash modules/dinov2_retrieval/data/reviewer/restore_runtime.sh "$PWD"
# CPU, без NVIDIA и без домена:
docker compose -f compose.pipeline.yml \\
  -f modules/dinov2_retrieval/data/reviewer/compose.cpu.yml up -d --build
# Или NVIDIA GPU:
# docker compose -f compose.pipeline.yml up -d --build
```

Интерфейс: http://localhost:3000/; Swagger: http://localhost:8080/docs.
Восстановление разрешено только в пустую БД: скрипт не перезаписывает каталог.
Рекомендуется 16 GiB RAM и 35 GiB свободного диска для данных, моделей и образов.
Параметры моделей: DINOv2 giant / 1536 и multilingual-e5-base / 768.
Отзывы, пользовательские фото/эмбеддинги и организаторский eval не включены.
Состав, число записей и SHA-256 каждого файла находятся в runtime_manifest.json.
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", default="dinov2_retrieval-postgres-1")
    parser.add_argument("--database", default="dinov2")
    parser.add_argument("--user", default="dinov2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data = (root / "modules/dinov2_retrieval/data").resolve()
    output = args.output.resolve()
    checksum_path = output.with_suffix(output.suffix + ".sha256")
    if output.exists() or checksum_path.exists():
        raise SystemExit("Refusing to overwrite an existing runtime archive/checksum")
    output.parent.mkdir(parents=True, exist_ok=True)
    scratch = "runtime_export_" + uuid.uuid4().hex
    if args.database == scratch:
        raise SystemExit("Scratch database must differ from the source database")

    def docker(*command: str, stdin=None, stdout=subprocess.PIPE, interactive=False):
        invocation = ["docker", "exec"] + (["-i"] if interactive else [])
        invocation += [args.container, *command]
        return subprocess.run(invocation, stdin=stdin, stdout=stdout,
                              stderr=subprocess.PIPE, check=True).stdout

    def sql(database: str, statement: str) -> str:
        return docker("psql", "-X", "-v", "ON_ERROR_STOP=1", "-U", args.user,
                      "-d", database, "-Atc", statement).decode().strip()

    created = False
    try:
        with tempfile.TemporaryDirectory(prefix="svoevino-runtime-") as directory:
            stage = Path(directory)
            snapshot = stage / "snapshot.dump"
            print("Creating a consistent DB snapshot without feedback data", flush=True)
            excludes = ["--exclude-table-data=public." + name for name in EXCLUDED_DATA]
            with snapshot.open("wb") as stream:
                docker("pg_dump", "-U", args.user, "-d", args.database,
                       "-Fc", "--no-owner", "--no-acl", *excludes, stdout=stream)
            docker("createdb", "-U", args.user, scratch)
            created = True
            with snapshot.open("rb") as stream:
                docker("pg_restore", "--exit-on-error", "-U", args.user,
                       "-d", scratch, "--no-owner", "--no-acl",
                       stdin=stream, interactive=True)
            # Only this uniquely named temporary export DB is changed.
            sql(scratch, "DELETE FROM reference_images WHERE image_uri IS NULL "
                "OR image_uri NOT LIKE '/data/reference/catalog/%'")
            counts = json.loads(sql(scratch, "SELECT json_build_object("
                "'wines', (SELECT count(*) FROM wines),"
                "'slugs', (SELECT count(DISTINCT slug) FROM wines),"
                "'reference_images', (SELECT count(*) FROM reference_images),"
                "'text_embeddings', (SELECT count(*) FROM wines WHERE description_text_embedding IS NOT NULL))"))
            for table in EXCLUDED_DATA:
                if sql(scratch, "SELECT count(*) FROM " + table) != "0":
                    raise RuntimeError("Private/transient data remains in " + table)
            image_models = sql(scratch, "SELECT DISTINCT model_name || ':' || vector_dims(embedding) "
                               "FROM reference_images ORDER BY 1").splitlines()
            text_models = sql(scratch, "SELECT DISTINCT description_text_embedding_model || ':' || "
                              "vector_dims(description_text_embedding) FROM wines "
                              "WHERE description_text_embedding IS NOT NULL ORDER BY 1").splitlines()
            if image_models != ["facebook/dinov2-with-registers-giant:1536"]:
                raise RuntimeError("Unexpected image embedding model/dimension: " + repr(image_models))
            if text_models != ["intfloat/multilingual-e5-base:768"]:
                raise RuntimeError("Unexpected text embedding model/dimension: " + repr(text_models))

            names = {uri.removeprefix("/data/") for uri in sql(
                scratch, "SELECT image_uri FROM reference_images ORDER BY image_uri").splitlines()}
            names.update("web_photos/" + name for name in sql(
                scratch, "SELECT DISTINCT web_photo FROM wines WHERE web_photo IS NOT NULL "
                         "AND web_photo <> '' ORDER BY web_photo").splitlines())
            files: dict[str, Path] = {}
            for name in sorted(names):
                relative = PurePosixPath(name)
                if relative.is_absolute() or ".." in relative.parts or "\\" in name:
                    raise RuntimeError("Unsafe reference path: " + name)
                if not (name.startswith("reference/catalog/") or name.startswith("web_photos/")):
                    raise RuntimeError("Unexpected reference directory: " + name)
                path = (data / name).resolve()
                if not path.is_relative_to(data) or not path.is_file():
                    raise RuntimeError("Missing or outside-data reference photo: " + name)
                files[name] = path

            dump = stage / "catalog-giant-e5.dump"
            with dump.open("wb") as stream:
                docker("pg_dump", "-U", args.user, "-d", scratch, "-Fc",
                       "--no-owner", "--no-acl", stdout=stream)
            files["exports/catalog-giant-e5.dump"] = dump
            # Bundle these small helpers so the runtime does not depend on
            # pushing the new deployment files to the public Git repository.
            files["reviewer/restore_runtime.sh"] = root / "deploy/restore_runtime.sh"
            files["reviewer/compose.cpu.yml"] = root / "compose.pipeline.cpu.yml"
            manifest = {"format": "svoevino-runtime-v1",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "counts": counts, "image_models": image_models, "text_models": text_models,
                        "excluded_table_data": list(EXCLUDED_DATA),
                        "reference_scope": "/data/reference/catalog/",
                        "files": {name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                                  for name, path in sorted(files.items())}}
            print(json.dumps({"counts": counts, "files": len(files),
                              "uncompressed_bytes": sum(p.stat().st_size for p in files.values())}), flush=True)
            with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
                for name, path in sorted(files.items()):
                    archive.write(path, name)
                archive.writestr("runtime_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
                archive.writestr("README_RUNTIME.md", RUNTIME_README)
            with zipfile.ZipFile(output) as archive:
                bad = archive.testzip()
                if bad is not None:
                    raise RuntimeError("ZIP CRC verification failed: " + bad)
            digest = sha256(output)
            checksum_path.write_text(digest + "  " + output.name + "\n", encoding="utf-8")
            print(json.dumps({"archive": str(output), "bytes": output.stat().st_size,
                              "sha256": digest}, ensure_ascii=False), flush=True)
    finally:
        if created:
            # Never drop an existing DB or a user-supplied name.
            if scratch.startswith("runtime_export_") and len(scratch) == 47:
                docker("dropdb", "-U", args.user, scratch)


if __name__ == "__main__":
    main()
