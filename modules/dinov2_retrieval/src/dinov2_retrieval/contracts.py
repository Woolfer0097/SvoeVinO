"""Input and output contracts for the retrieval scaffold."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class RetrievalRequest(BaseModel):
    """Request contract used by the future retrieval pipeline."""

    job_id: str
    query_id: str
    image_uri: str
    top_k: int = 20


class ValidatedImage(BaseModel):
    """Information about an image that passed local validation."""

    path: Path
    width: int
    height: int
    mime_type: str


class EmbeddingResult(BaseModel):
    """Full embedding and metadata returned by the embedding use case."""

    path: str
    model: str
    dimension: int
    device: str
    embedding: list[float]
