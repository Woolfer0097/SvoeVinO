"""Internal HTTP adapter for the independent image preprocessing package."""
from __future__ import annotations

import asyncio
import base64
import io
import os
from dataclasses import replace

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from .models import PreprocessingConfig
from .photometry import mild_ocr_profile
from .pipeline import preprocess

MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 25_000_000


def _flag(name: str, default: bool) -> bool:
    value = os.getenv(name, str(default)).strip().lower()
    if value not in {"true", "false", "1", "0"}:
        raise ValueError(f"{name} must be true or false")
    return value in {"true", "1"}


def service_config() -> PreprocessingConfig:
    visual_side = int(os.getenv("PREPROCESS_VISUAL_MAX_SIDE", "1600"))
    ocr_side = int(os.getenv("PREPROCESS_OCR_MAX_SIDE", "1920"))
    if not 128 <= visual_side <= 2048 or not 128 <= ocr_side <= 2048:
        raise ValueError("Preprocessing sides must be between 128 and 2048")
    config = PreprocessingConfig(embedding_max_long_side=visual_side, ocr_max_long_side=ocr_side)
    if _flag("PREPROCESS_OCR_MILD", True):
        config = replace(config, ocr_photometric=mild_ocr_profile(
            unsharp=_flag("PREPROCESS_OCR_UNSHARP", False),
        ))
    return config


def _encode(array) -> str:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def process_bytes(data: bytes, config: PreprocessingConfig) -> dict:
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in {"JPEG", "PNG", "WEBP"}:
                raise HTTPException(415, "Supported images: JPEG, PNG, WebP")
            if image.width * image.height > MAX_PIXELS:
                raise HTTPException(413, "Image exceeds 25 million pixels")
            result = preprocess(image, config=config)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(422, "Corrupted or unsupported image") from exc
    return {
        "embedding_png": _encode(result.embedding_image),
        "ocr_png": _encode(result.ocr_image),
        "metadata": result.metadata,
    }


def create_app(config: PreprocessingConfig | None = None) -> FastAPI:
    config = config or service_config()
    cv2.setNumThreads(2)
    app = FastAPI(title="Wine label preprocessing")
    semaphore = asyncio.Semaphore(1)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/preprocess")
    async def process(image: UploadFile = File(...)):
        data = await image.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise HTTPException(413, "Image exceeds 10 MiB")
        if not data:
            raise HTTPException(422, "Image is empty")
        async with semaphore:
            return await asyncio.to_thread(process_bytes, data, config)

    return app
