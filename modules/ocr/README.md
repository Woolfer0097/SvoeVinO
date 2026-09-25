# Wine OCR

Isolated OCR service for extracting text signals from Russian wine bottle or
label photos.

The module is intentionally catalog-agnostic. It does not decide which wine was
photographed. It returns raw OCR text, normalized text, text blocks with
confidence and coordinates, and candidate fields such as a provisional wine
name, years, percentages, and volumes. The downstream retrieval/matching
pipeline can use these signals to rerank visually similar candidates.

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
| `OCR_TEXT_DETECTION_MODEL_NAME` | Optional PaddleOCR detection model name |
| `OCR_TEXT_RECOGNITION_MODEL_NAME` | Optional PaddleOCR recognition model name |

These are OCR pipeline settings, not image enhancement. Change one setting at a
time and compare against the same labeled photos before adopting it as a
default. A faster profile is not promoted if it loses important text.

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

## Local Checks

The deterministic unit tests do not require PaddleOCR, FastAPI, Pillow, or model
weights:

```bash
python -m unittest discover -s tests
python -m compileall src tests
```

These checks are functional tests for code paths that do not perform OCR
inference. They are not an OCR quality benchmark.
