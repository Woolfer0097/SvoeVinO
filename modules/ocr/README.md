# Wine OCR

The module includes lightweight tests and manual-review tools. Photos,
annotations, and generated model responses are local inputs and are not needed
in Git for installation or unit tests.

OCR service for Russian wine bottle or label photos. The core recognizer extracts
text signals; an optional matching path searches catalog text embeddings.

The core recognizer remains catalog-agnostic. It returns raw OCR text,
normalized text, text blocks with confidence and coordinates, and candidate
fields such as a provisional wine name, years, percentages, and volumes. The
separate `/match` path uses the existing PostgreSQL catalog to return candidate
IDs without changing the recognizer's output. It currently ranks text matches
alone; combining them with visual retrieval is future integration work.

## Scope

Owned area:

```text
modules/ocr/**
```

The standalone `/ocr` and `/ocr/txt` endpoints run without other project
modules. `/match` additionally needs the catalog in PostgreSQL, populated with
text embeddings. No other Python package in this repository is imported by
the OCR module.

## Runtime

- Python 3.12
- Docker-first deployment
- Local inference only
- PaddleOCR baseline backend
- REST entrypoint for image uploads

The Dockerfile pins PaddleOCR 3.7.0 and PaddlePaddle 3.2.2 and installs a
CPU-only PyTorch 2.8.0 wheel for text embeddings. Setting
`TEXT_EMBEDDING_DEVICE=cuda` alone does not make this Docker image GPU-capable.

Importing `wine_ocr` or `wine_ocr.engine.paddle` does not create a PaddleOCR
client, download weights, contact the network, or run inference. PaddleOCR is
initialized lazily only when `recognize()` is called. The REST service keeps a
single cached OCR engine for the active backend settings, so model initialization
happens on the first OCR request and later requests reuse the loaded backend.
First use may download PaddleOCR or E5 weights; inference itself is local.

## REST API

Start the standalone OCR service from `modules/ocr` (the default published port
is `8000`, unless `.env` sets `OCR_PORT`). The tracked `outputs/.gitkeep`
keeps the bind-mounted output directory in a fresh checkout:

```bash
docker compose up --build
```

To run `/match` against the catalog, start `postgres` from
`modules/dinov2_retrieval` first and ensure its `wines` table contains text
embeddings (see [the retrieval setup](../dinov2_retrieval/README.md); starting
an empty PostgreSQL instance is insufficient). In this module, copy
`.env.example` to an untracked `.env`, set a
real `DATABASE_URL` for the reachable PostgreSQL service (for example,
`postgresql://USER:PASSWORD@postgres:5432/wine_catalog`) and set
`OCR_PORT=127.0.0.1:8001`. Then start OCR with the catalog network override:

```bash
docker compose -f docker-compose.yml -f docker-compose.catalog.yml up -d --build
```

`docker-compose.catalog.yml` joins the retrieval Compose network
`dinov2_retrieval_default`. If the retrieval project uses a different Compose
project name, update the external network name in that file. With this setup,
OCR is available at `http://localhost:8001`; the ordinary standalone command
above still works without the catalog network. The catalog override also keeps
downloaded Hugging Face/E5 model files in a Docker volume across OCR container
recreation. The examples below use port `8001` for the integrated setup;
replace it with `8000` for the standalone default. A successful `/health`
response only checks the API process, not model readiness or database access.

JSON response with structured OCR result and a TXT report:

```bash
curl -F "file=@label.jpg" http://localhost:8001/ocr
```

By default, `/ocr` returns the TXT report inline in the JSON response and does
not persist it on disk. Set `OCR_REPORT_RETENTION=true` to also save reports
under `OCR_OUTPUT_DIR` and return the saved path in `txt_file`.

Download a plain TXT report:

```bash
curl -F "file=@label.jpg" http://localhost:8001/ocr/txt -o ocr_report.txt
```

Find the ten closest catalog rows from the same uploaded photo:

```bash
curl -F "file=@label.jpg" http://localhost:8001/match
```

`/match` runs the existing OCR pipeline, converts its complete
`normalized_text` into a multilingual E5 query vector, and compares that vector
with `wines.description_text_embedding` in PostgreSQL. For the embedding query
only, it repeats the likely central `candidate_name` once and repairs words made
entirely of Cyrillic letters and visually similar Latin OCR characters (for
example `3akat` → `закат`). The original OCR JSON and TXT stay unchanged.
It returns only the catalog row IDs and scores:

```json
{"top_10": {"123": 0.94, "456": 0.88}}
```

The example shows two entries for brevity; the service returns up to ten. JSON
object keys are strings even though `wines.id` is a numeric PostgreSQL ID. The
score is `1 - cosine_distance / 2`, constrained to `[0, 1]`. It orders results;
it is not a calibrated probability. No recognized text or TXT report is included
in this endpoint. An image with no recognized text returns `{"top_10": {}}`.
The existing `/ocr` and `/ocr/txt` endpoints remain available for manual review.

The new stages are separate packages under `wine_ocr`: `text_processing` handles
only text-to-vector conversion, while `embedding_comparison` performs a read-only
PostgreSQL search and creates the final response. `text_processing` uses
`intfloat/multilingual-e5-base` by default, with the same mean pooling and L2
normalization as the catalog generator. Catalog entries use `passage: ` and OCR
queries use `query: `. Text longer than the model's 512-token input is split at
word boundaries; all chunks contribute to one normalized query vector. The
matching code does not import the DINOv2 package or compare text vectors with
image vectors.

`/match` requires the `matching` dependency extra, which the OCR Dockerfile
installs. Configure `DATABASE_URL` for a PostgreSQL database with pgvector and
populated `wines.description_text_embedding` rows. The query selects only rows
whose `description_text_embedding_model` equals the query model and whose vector
dimension matches. It never creates tables or changes catalog records. Set
`TEXT_EMBEDDING_MODEL_NAME` only when the catalog has vectors from that exact
model; `TEXT_EMBEDDING_DEVICE` accepts `auto`, `cpu`, or `cuda`. Loading the E5
model and connecting to PostgreSQL happen on the first matching request that
produces non-empty OCR text, not during package import or `/health`. If the model
or database is unavailable, `/match` returns HTTP 503. An unreadable image
returns HTTP 400.

The OCR and retrieval Compose projects are separate. Before an integrated run,
give the OCR container a reachable `DATABASE_URL` and, if both HTTP services
run on one host, select different published ports (for example `OCR_PORT=8001`).
The existing retrieval service's `reference_images.wine_id` is a string and is
not automatically linked to the numeric `wines.id` returned here. Combining
visual and text rankings therefore needs an explicit ID mapping later.

The TXT report contains sections:

- raw lines
- normalized text
- candidate name
- candidate years
- candidate percentages
- candidate volumes

`candidate_name` is selected heuristically from name-like text near the center
of the image, preferring the central-crop OCR pass. The same phrase appears as
`field_type="name"` in `candidate_fields` with its OCR confidence and source
variant. It may be `null` when no plausible phrase is found; it is not a
catalog match or a calibrated probability of the wine name. The heuristic can
join two similarly sized centered lines, including `ESTATE` when it is part of
a name, and can retain a visually dominant title despite moderately low OCR
confidence. It does not rewrite the raw recognized text.

`result.variant_times_ms` records the elapsed time of each OCR pass (`full` and
`central_crop`). The first pass also includes lazy model initialization on a
cold request. The service continues to run both passes by default because the
central pass recovers useful label text on the current review photos.

## PaddleOCR experiments

The service exposes the installed PaddleOCR 3.x pipeline options as environment
variables. The evaluated default disables whole-image document orientation,
keeps document unwarping and text-line orientation enabled, uses PaddleOCR's
default detector and recognizer, and recognizes both image variants. This
profile was selected after local comparison on annotated photos. The
previously used all-enabled profile remains available by setting
`OCR_USE_DOC_ORIENTATION_CLASSIFY=true`.
PaddleOCR 3.7.0 and PaddlePaddle 3.2.2 are pinned for reproducible comparisons;
the OCR model and weights can still be changed in a later experiment.

| Variable | Meaning |
| --- | --- |
| `OCR_USE_DOC_ORIENTATION_CLASSIFY` | Enable whole-image orientation classifier |
| `OCR_USE_DOC_UNWARPING` | Enable PaddleOCR document unwarping |
| `OCR_USE_TEXTLINE_ORIENTATION` | Enable text-line orientation classifier |
| `OCR_TEXT_DET_LIMIT_SIDE_LEN` | Optional positive detection-side limit |
| `OCR_TEXT_DET_THRESH` | Optional pixel threshold for detecting weak text |
| `OCR_TEXT_DET_BOX_THRESH` | Optional box confidence threshold for detecting weak text |
| `OCR_TEXT_DETECTION_MODEL_NAME` | Optional PaddleOCR detection model name |
| `OCR_TEXT_RECOGNITION_MODEL_NAME` | Optional PaddleOCR recognition model name |

These are OCR pipeline settings, not image enhancement. Change one setting at a
time and compare against the same labeled photos before adopting it as a
default. A faster profile is not promoted if it loses important text.

Docker Compose uses the official BOS model source for first-time PaddleOCR
downloads (`PADDLE_PDX_MODEL_SOURCE=BOS`). Inference remains local. PaddleOCR
weights are cached inside the container under `/home/app/.paddlex`, so a newly
created container may need to download them again. `docker stop` preserves the
container and that cache; `docker compose down` removes the container. The E5
cache is persisted separately by `docker-compose.catalog.yml`.

## Manual review with your own photos

The tracked `tests/fixtures/manual_review/photos/README.txt` keeps an empty
input directory in a fresh checkout. Put your JPEG, PNG, WebP, BMP, or TIFF
photos there, or pass another directory as the first argument to a run tool.
The run tools sort filenames case-insensitively, copy images into a new
`outputs/testN` directory as `image1`, `image2`, and so on, and record the
original names in `manifest.json`. The marker TXT is ignored by the tools;
photos placed in this input directory are ignored by Git.

Start the OCR service and, from `modules/ocr`, run:

```bash
python tests/run_manual_review.py --url http://127.0.0.1:8001/ocr
```

Use port `8000` if running the standalone default. The command creates
`imageN_output.json` and `imageN_output.txt` for each photo. `outputs/testN`
is local and Git-ignored. The user can inspect these files without an
annotation file.

To evaluate against known text, copy
`tests/fixtures/manual_review/correct_text.example.txt` to
`tests/fixtures/manual_review/correct_text.txt` and fill in `name`, optional
`years`, and `other` for **every** `imageN` in the manifest. The real
annotation file is Git-ignored. Then run:

```bash
python tests/evaluate_manual_review.py tests/fixtures/manual_review/correct_text.txt outputs/testN --label chosen-profile
python tests/diagnose_manual_review.py outputs/testN
```

The evaluator writes `outputs/testN/evaluation.json`, and the diagnostic tool
writes `diagnostics.json`. They compare selected phrases strictly after Unicode
normalization; their counts are not a complete OCR accuracy measure. These
commands require an OCR run with `imageN_output.json` files; they cannot
evaluate a `/match`-only run.

With a populated PostgreSQL catalog, test the matching endpoint separately:

```bash
python tests/run_match_review.py --url http://127.0.0.1:8001/match --expected-count 10
```

That tool saves `imageN_match_output.json` in a new `outputs/testN` folder and
checks response structure, numeric IDs, score range, and rank order. It does
not generate TXT. Omit `--expected-count 10` for a smaller catalog. Both run
tools accept `--resume outputs/testN` after an interrupted run. Historical
`outputs/testN` results and local `tests/fixtures/baselines/` are deliberately
excluded from Git.

No historical output file is required for the API, unit tests, or a new manual
run. `outputs/.gitkeep` is the only tracked file needed there to preserve the
bind-mount directory. The tests use small synthetic data, so a clean checkout
passes unit tests before anyone supplies photos. In Windows PowerShell, use
`curl.exe` for the direct HTTP examples above if `curl` is an alias.

## Local Checks

The deterministic unit tests do not require PaddleOCR, FastAPI, Pillow, or model
weights:

```bash
python -m unittest discover -s tests
python -m compileall src tests
```

These checks are functional tests for code paths that do not perform OCR
inference. They are not an OCR quality benchmark.
# Common pipeline / GPU

Root `compose.pipeline.yml` runs DINO/SuperPoint on GPU and OCR/E5 on CPU
for a 4 GiB workstation. `compose.pipeline.gpu.yml` additionally moves
PaddleOCR/E5 to GPU using `Dockerfile.gpu`; build its CUDA base first
as described in `modules/wine_pipeline/README.md`.
Text search returns ten distinct slugs even when the source export contains
multiple rows for one wine. The orchestrator maps returned row IDs to slugs.
