# Wine OCR

OCR service for Russian wine bottle or label photos. The core recognizer extracts
text signals; an optional matching path searches catalog text embeddings.

The core recognizer remains catalog-agnostic. It returns raw OCR text, normalized
text, text blocks with
confidence and coordinates, and candidate fields such as a provisional wine
name, years, percentages, and volumes. The downstream retrieval/matching
pipeline can use these signals to rerank visually similar candidates. The
separate `/match` path uses the existing PostgreSQL catalog to return candidate
IDs without changing the recognizer's output.

## Scope

Owned area:

```text
modules/ocr/**
```

No files outside this directory are required for the module to run.

## Runtime

- Python 3.12
- Docker-first deployment
- Local inference only
- PaddleOCR baseline backend
- REST entrypoint for image uploads

Importing `wine_ocr` or `wine_ocr.engine.paddle` does not create a PaddleOCR
client, download weights, contact the network, or run inference. PaddleOCR is
initialized lazily only when `recognize()` is called. The REST service keeps a
single cached OCR engine for the active backend settings, so model initialization
happens on the first OCR request and later requests reuse the loaded backend.

## REST API

Start the service:

```bash
docker compose up --build
```

To run `/match` against the restored wine catalog, start `postgres` from
`modules/dinov2_retrieval` first. In this module, copy `.env.example` to `.env`,
set `DATABASE_URL=postgresql://dinov2:dinov2@postgres:5432/wine_catalog` and
`OCR_PORT=127.0.0.1:8001`, then start OCR with the catalog network override:

```bash
docker compose -f docker-compose.yml -f docker-compose.catalog.yml up -d --build
```

`docker-compose.catalog.yml` joins the retrieval Compose network
`dinov2_retrieval_default`. If the retrieval project uses a different Compose
project name, update the external network name in that file. With this setup,
OCR is available at `http://localhost:8001`; the ordinary standalone command
above still works without the catalog network. The catalog override also keeps
downloaded Hugging Face model files in a Docker volume across OCR container
recreation.

JSON response with structured OCR result and a TXT report:

```bash
curl -F "file=@label.jpg" http://localhost:8000/ocr
```

By default, `/ocr` returns the TXT report inline in the JSON response and does
not persist it on disk. Set `OCR_REPORT_RETENTION=true` to also save reports
under `OCR_OUTPUT_DIR` and return the saved path in `txt_file`.

Download a plain TXT report:

```bash
curl -F "file=@label.jpg" http://localhost:8000/ocr/txt -o ocr_report.txt
```

Find the ten closest catalog rows from the same uploaded photo:

```bash
curl -F "file=@label.jpg" http://localhost:8000/match
```

`/match` runs the existing OCR pipeline, converts its complete
`normalized_text` into a multilingual E5 query vector, and compares that vector
with `wines.description_text_embedding` in PostgreSQL. It returns only the
catalog row IDs and scores:

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
model and connecting to PostgreSQL happen on the first matching request, not
during package import or `/health`. If the model or database is unavailable,
`/match` returns HTTP 503.

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
profile improved the annotated text on the current 15 photos without reducing
strict matches for any annotated phrase. The previously used all-enabled
profile remains available by setting `OCR_USE_DOC_ORIENTATION_CLASSIFY=true`.
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

Docker Compose uses the official BOS model source for first-time model
downloads (`PADDLE_PDX_MODEL_SOURCE=BOS`). Inference remains local. Model files
are not persisted outside the container, so a newly created container may need
to download them again.

## Manual review evaluation

`scripts/run_manual_review.py` writes a new `outputs/testN` folder with copied
`imageN` files, JSON/TXT responses, and a source-name manifest. The expected
text file remains outside the OCR module and is read-only during evaluation.
With the Docker service running, use the current local photo set like this:

```bash
python scripts/run_manual_review.py ../../images/test_images
python scripts/evaluate_manual_review.py \
  "../../images/test_images/correct text.txt" outputs/testN \
  --label chosen-profile
```

Replace `testN` with the newly printed folder name. The second command saves
`outputs/testN/evaluation.json`. It compares `candidate_name`,
the complete normalized-text phrase, each annotated year, and each `other`
phrase. Matching uses Unicode NFKC, case folding, whitespace normalization,
and word boundaries. It deliberately does not correct Latin/Cyrillic lookalikes
or approximate spellings, so its counts are strict regression indicators rather
than a complete measure of OCR quality. Review the individual JSON/TXT outputs
alongside the evaluation report before accepting a new profile.

On the current 15 manually annotated photos, the quality experiments produced:

| Run | Change from `test8` | Exact name candidate | Complete name in text | Expected years | `other` phrases |
| --- | --- | ---: | ---: | ---: | ---: |
| `test8` | Current default | 7/15 | 6/15 | 7/8 | 16/32 |
| `test9` | Cyrillic PP-OCRv5 recognizer | 4/15 | 6/15 | 8/8 | 17/32 |
| `test10` | PP-OCRv6 medium detector | 1/15 | 3/15 | 8/8 | 11/32 |
| `test11` | PP-OCRv5 box threshold 0.4 | 5/15 | 6/15 | 6/8 | 15/32 |

These counts are strict matches against selected phrases, not CER or general
accuracy estimates. The faster alternative recognizer and detector both lost
previously correct names. The lower box threshold recovered `пет-нат` in the
text of `image10`, but lost two correct name candidates and one expected year.
The default profile therefore remains unchanged. A 0.5 threshold probe on four
key photos also lost a correct name and failed to recover `пет-нат`, so it was
not run on the complete set.

## Local Checks

The deterministic unit tests do not require PaddleOCR, FastAPI, Pillow, or model
weights:

```bash
python -m unittest discover -s tests
python -m compileall src tests
```

These checks are functional tests for code paths that do not perform OCR
inference. They are not an OCR quality benchmark.
