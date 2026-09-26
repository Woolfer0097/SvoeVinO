"""FastAPI REST entrypoint for OCR image uploads."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.concurrency import run_in_threadpool

from ..application.matching import match_image_from_bytes
from ..application.runtime import CachedOCRRuntime
from ..config import OCRConfig
from ..embedding_comparison import EmbeddingComparisonError, PostgresEmbeddingRepository
from ..exceptions import (
    ConfigurationError,
    ImageDecodeError,
    OCREngineError,
    OCRError,
)
from ..reporting import render_txt_report, safe_report_filename
from ..text_processing import E5TextEmbedder, TextEmbeddingError

app = FastAPI(title="Wine OCR", version="0.1.0")
ocr_runtime = CachedOCRRuntime()
text_embedder = E5TextEmbedder()
embedding_repository = PostgresEmbeddingRepository()


@app.get("/health")
def health() -> dict[str, str]:
    """Return a lightweight health response without initializing OCR models."""

    return {"status": "ok"}


@app.post("/ocr")
async def recognize_image(file: UploadFile = File(...)) -> JSONResponse:
    """Recognize uploaded image and return JSON plus a TXT report."""

    result, report, report_path = await _process_upload(file, save_report=True)
    return JSONResponse(
        {
            "result": result.to_dict(),
            "txt_report": report,
            "txt_file": str(report_path) if report_path is not None else None,
        }
    )


@app.post("/ocr/txt")
async def recognize_image_txt(file: UploadFile = File(...)) -> PlainTextResponse:
    """Recognize uploaded image and return a downloadable TXT report."""

    _, report, _ = await _process_upload(file, save_report=False)
    filename = safe_report_filename(file.filename)
    return PlainTextResponse(
        report,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/match")
async def match_image(file: UploadFile = File(...)) -> JSONResponse:
    """Recognize a photo, embed all OCR text, and return ten catalog IDs."""

    try:
        config = OCRConfig.from_env()
        data = await file.read()
        matches = await run_in_threadpool(
            match_image_from_bytes,
            data,
            config=config,
            ocr_runtime=ocr_runtime,
            text_embedder=text_embedder,
            repository=embedding_repository,
        )
    except ConfigurationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ImageDecodeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OCREngineError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except TextEmbeddingError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except EmbeddingComparisonError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except OCRError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return JSONResponse(matches)


async def _process_upload(
    file: UploadFile,
    *,
    save_report: bool,
):
    try:
        config = OCRConfig.from_env()
        data = await file.read()
        result = ocr_runtime.run_ocr_from_bytes(data, config=config)
        report = render_txt_report(result)
        report_path = (
            _save_report(report, file.filename, config)
            if save_report and config.report_retention
            else None
        )
    except ConfigurationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ImageDecodeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OCREngineError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except OCRError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return result, report, report_path


def _save_report(
    report: str,
    original_filename: str | None,
    config: OCRConfig,
) -> Path:
    output_dir = config.output_dir
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / safe_report_filename(original_filename)
        report_path.write_text(report, encoding="utf-8")
    except OSError as exc:
        raise OCRError("Cannot save OCR TXT report") from exc
    return report_path
