# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

SvoeVinO is a wine-label recognition project, split into independent Python packages under `modules/`. Each module has its own `pyproject.toml` (setuptools, `src/` layout, Python >=3.11), its own tests, and its own README (in Russian; project docs and user-facing text are written in Russian). **Modules do not import each other.** Design specs and implementation plans live in `docs/superpowers/{specs,plans}/`.

- `modules/dinov2_retrieval` — visual wine search: embeds a photo with `facebook/dinov2-with-registers-giant` (1536-dim CLS token; the largest DINOv2), searches reference-photo embeddings in PostgreSQL/pgvector and returns Top-K distinct wines; a separate `index` command stores reference embeddings, `evaluate` measures Recall@K on labelled queries. CLI is the Airflow contract. OCR, SuperPoint/LightGlue, frontend and recommendations are out of scope.
- `modules/wine_label_preprocessing` — CPU-first preprocessing of bottle photos: crop, safe photometry, cylindrical label unwrapping, optional DewarpNet adapter, A–E comparison and benchmarking. Contains no detector, OCR, or embedding model.

## Commands

Run from inside the module directory. Each module expects its own `.venv` (the repo-root `.venv` is not tied to either module).

```bash
# dinov2_retrieval
cd modules/dinov2_retrieval
python -m venv .venv && .venv/bin/pip install -e '.[dev,db,api]'  # add ,ml for real torch/transformers
.venv/bin/pytest                                               # unit tests; integration tests are deselected by default
.venv/bin/pytest tests/test_cli.py::test_name
docker compose build && docker compose up -d postgres          # needs .env (cp .env.example .env), ./data, ./model-cache
docker compose run --rm dinov2-retrieval pytest -q
docker compose run --rm dinov2-retrieval pytest -m integration -q   # real PostgreSQL + DINOv2, run manually
docker compose run --rm dinov2-retrieval health
docker compose run --rm dinov2-retrieval index [--prune]    # DATA_ROOT/reference: its manifest.csv if present, else a folder scan; or --manifest X / --reference-dir X
docker compose run --rm dinov2-retrieval search --image-uri /data/queries/test.jpeg --top-k 20   # or --request-json /data/requests/x.json
docker compose run --rm dinov2-retrieval evaluate --queries /data/evaluation/queries.csv --top-k 20
docker run --rm -v "$PWD/data:/data" -v "$PWD/examples:/app/examples:ro" --entrypoint python dinov2_retrieval-dinov2-retrieval examples/generate_demo_data.py /data   # synthetic demo data; never overwrites without --force
docker compose up -d                                           # long-running service; Swagger at http://localhost:8000 runs search/index/references/images/evaluate/health

# wine_label_preprocessing
cd modules/wine_label_preprocessing
python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev]'  # add ,dewarpnet for torch
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest tests/test_geometry.py -k name
.venv/bin/python -m wine_label_preprocessing process <file-or-dir> --output-dir out [--label-bbox ... --cx ... --radius ...] [--annotations x.jsonl]
.venv/bin/python -m wine_label_preprocessing benchmark benchmark.jsonl --output-dir out
.venv/bin/python examples/generate_synthetic_case.py /tmp/demo   # synthetic benchmark input
```

No linter/formatter is configured.

## dinov2_retrieval architecture

Layered: `entrypoints/cli.py` (argparse) and `entrypoints/api.py` (FastAPI `create_app()` factory, `api` extra) → `application/` use cases (`validate_image`, `create_embedding`, `index_reference_images`, `search_similar_wines`, `evaluate_retrieval`, `check_health`) → `preprocessing/` (Pillow decode → EXIF transpose → alpha flattened onto white → RGB; changing it requires re-indexing), `embedding/` (`Embedder` Protocol + `DinoV2Embedder`), `infrastructure/storage/local_storage.py` (`LocalImageStorage`: path must resolve under `DATA_ROOT`, extension/size/MIME checks) → `retrieval/` Protocols (`Retriever.search_similar`, `ReferenceRepository`, `RepositoryError`) → `infrastructure/database/` (psycopg + pgvector: `connection.py`, `PostgresReferenceRepository`, `schema.sql`). `infrastructure/csv_manifest.py` is the shared CSV reader; `reference_manifest.py` (manifest.csv, or a folder scan where `<dir>/<file>` is one wine named by the file stem and `<dir>/<wine>/<file>` one wine named by the subfolder) and `evaluation_manifest.py` (queries.csv) build on it. Paths inside CSVs must be absolute container paths under `DATA_ROOT` (`/data/...`); CLI path arguments may be relative to `DATA_ROOT`.

- Application code never imports psycopg: the CLI composes `PostgresReferenceRepository` (lazy import, `db` extra) and passes it in. Both index and search go through `create_embedding`, so reference and query photos are processed identically.
- Search fetches `max(RAW_RETRIEVAL_LIMIT, top_k)` nearest photos of the same `model_name` (cosine `<=>`), groups by `wine_id` keeping the closest photo, `score = 1 - distance`; if that covers fewer than `top_k` wines while more rows exist, the limit doubles and the query repeats. Query photos are never stored.
- Indexing upserts by canonical `image_uri` (rerun = update); `--prune` deletes rows of the same model not in the source, but keeps rows of photos that failed this run and never runs if nothing was processed. Per-photo errors go to `error_details`, `RepositoryError` aborts the run. `IndexingStats.source` is filled by the CLI.
- `evaluate` runs every query through `search_similar_wines` at depth `max(20, top_k)`, reports Recall@1/5/20 and @top_k over processed queries only; broken photos and wines without indexed references go to `errors`, misses to `incorrect_queries`. Statuses/exit codes mirror `index`.
- `data/` is gitignored and may hold the synthetic demo set (`DEMO.txt` markers); don't treat demo Recall as model quality.
- CLI contract: exactly one JSON document on stdout for every outcome (indented on a TTY, one line otherwise for Airflow XCom); exit codes 0 ok, 1 input, 2 usage, 3 model, 4 database, 5 partial index/evaluate.
- `database/init.sql` must stay identical to `infrastructure/database/schema.sql` (a test checks it). Schema is fixed at `VECTOR(1536)`; tables are created `IF NOT EXISTS`, so a DB created for another dimension needs `DROP TABLE reference_images` and a re-index.
- `tests/conftest.py` clears module env vars for unit tests; `tests/integration/` is marked `integration` and deselected by default.
- Use cases take collaborators as optional arguments (dependency injection); tests pass mocks. `DinoV2Embedder` is imported lazily so the package works without the `ml` extra, and tests never download weights.
- `cli.py` and `storage.py` at the package root are thin compatibility shims — put real code in `entrypoints/` and `infrastructure/storage/`.
- Config is env-only via `config.py`: `DATA_ROOT` (required, must exist), `MAX_IMAGE_SIZE_BYTES` (default 10 MiB), `SUPPORTED_IMAGE_EXTENSIONS` (subset of jpg/jpeg/png/webp), `HF_HOME`, `DATABASE_URL`, `DINO_MODEL_NAME`, `DINO_EMBEDDING_DIMENSION`, `DEFAULT_TOP_K`, `RAW_RETRIEVAL_LIMIT`.
- Pydantic contracts in `contracts.py`. The `embed` CLI prints only the first 5 values; the full vector is in `EmbeddingResult`.
- API (`entrypoints/api.py`): the model is loaded in the FastAPI lifespan (once, before the first request); `POST /search` takes a multipart upload that `infrastructure/storage/upload_storage.temporary_upload` writes to a temp dir only for the request (so uploads get the same validation/preprocessing and are never kept), `POST /search/uri` takes a DATA_ROOT path; `GET /search/ids?image_uri=...` returns only a JSON array of the top-20 `wine_id`s (fixed `ID_SEARCH_TOP_K`, ignores `DEFAULT_TOP_K`); `POST /index` (one run at a time, else 409), `GET /references`, `GET /images` (JPEG preview or original, via `LocalImageStorage.locate`), `POST /evaluate` and `GET /health/details` mirror the CLI; endpoints are grouped by Russian tags and Swagger opens with Try-it-out enabled, `/` redirects to `/docs`. A DB connection is opened per request via an injectable `repository_factory`. Reference source selection (`load_references`) and `resolve_data_path` are shared with the CLI. Image errors map to HTTP 404/400/415/413/422, DB/model errors to 503. Needs the `api` extra incl. `python-multipart`; API tests `importorskip` fastapi.
- Docker: `docker-entrypoint.sh` routes `pytest` to pytest, `serve` (default) to uvicorn on :8000, everything else to the CLI. The `serve` service has `restart: unless-stopped` and a 300 s healthcheck start period (loading the ~4.5 GB giant model). Deps are installed from a stub package in an earlier layer (pip cache mount), so code changes rebuild fast; torch/torchvision come first from `TORCH_INDEX_URL` (build arg, default the CPU-only index — Docker here has no GPU runtime). `./data` is mounted read-only at `/data`; `./model-cache` is `/home/app/.cache/huggingface`. The app service depends on `postgres` (pgvector/pgvector:pg16, `127.0.0.1:5433`, volume `postgres_data`); published ports are bound to localhost on purpose.

## wine_label_preprocessing architecture

`pipeline.preprocess()` is the core entry point; `cli.py` handles `process`/`benchmark` subcommands and JSONL manifests; `benchmark.py` builds A–E variants (A original crop, B mild photometry, C cylindrical unwrap, D unwrap+photometry, E DewarpNet) and accepts pluggable `OCRBackend`/`EmbeddingBackend` Protocols. All dataclasses live in `models.py`, errors in `errors.py`.

Invariants to preserve (tests enforce them):
- All returned images are contiguous `uint8` `(H, W, 3)` **RGB**. NumPy input requires explicit `input_color_order="RGB"|"BGR"`; file/Pillow input gets `ImageOps.exif_transpose` first.
- BBox is `(x_min, y_min, x_max, y_max)`, half-open, in post-EXIF full-image coordinates, clamped to bounds. `cx`/`radius` describe the bottle body, not the label. Angles are radians.
- `original_crop` is never resized or photometrically altered; small images are never upscaled. Final model resize/normalization is left to the downstream model processor.
- Photometry touches only the LAB L-channel and is off by default for embedding/OCR branches.
- Unwrap uses an orthographic model (`x = cx + r·sinθ`, `u = r·θ`) with a single `cv2.remap`, bounded by `max_stretch` (1/cosθ), and returns a valid mask. On invalid geometry/weak mask it **skips with a reason** rather than failing; base branches stay available. DewarpNet is lazy-loaded and reports `unavailable`/`failed` — never substitute a plain resize as its result.
- Benchmark: the same `series_id` may not appear in both `tuning` and `test`; OCR/embedding sections are `not_run` from the CLI. Don't claim OCR/retrieval quality improvements — there is no real data, OCR backend, or checkpoints in the repo.
