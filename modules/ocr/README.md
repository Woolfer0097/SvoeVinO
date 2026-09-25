# Wine OCR

Isolated OCR service for extracting text signals from Russian wine bottle or
label photos.

The module is intentionally catalog-agnostic. It does not decide which wine was
photographed. It returns raw OCR text, normalized text, text blocks with
confidence and coordinates, and simple candidate fields such as years,
percentages, and volumes. The downstream retrieval/matching pipeline can use
these signals to rerank visually similar candidates.

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
- candidate years
- candidate percentages
- candidate volumes

## Local Checks

The deterministic unit tests do not require PaddleOCR, FastAPI, Pillow, or model
weights:

```bash
python -m unittest discover -s tests
python -m compileall src tests
```

These checks are functional tests for code paths that do not perform OCR
inference. They are not an OCR quality benchmark.
