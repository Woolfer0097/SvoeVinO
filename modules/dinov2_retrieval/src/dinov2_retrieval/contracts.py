"""Input and output contracts of the retrieval module."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .config import DEFAULT_TOP_K


class RetrievalRequest(BaseModel):
    """Request contract of the validate command."""

    job_id: str
    query_id: str
    image_uri: str
    top_k: int = 20


class ImageRequest(BaseModel):
    """HTTP request body that points to an image inside DATA_ROOT."""

    image_uri: str = Field(examples=["/data/queries/test.jpeg"])


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


class ReferenceImage(BaseModel):
    """One reference photo of a wine, as described by the manifest."""

    wine_id: str
    slug: str
    image_uri: str


class ReferenceImageRecord(BaseModel):
    """Reference photo embedding with the metadata stored in the database."""

    wine_id: str
    slug: str
    image_uri: str
    model_name: str
    embedding: list[float]


class ReferenceMatch(BaseModel):
    """One reference photo found by the vector search, closest first."""

    wine_id: str
    slug: str
    image_uri: str
    distance: float


class ReferenceWine(BaseModel):
    """One indexed wine with all of its reference photos."""

    wine_id: str
    slug: str
    image_uris: list[str]


class ReferenceList(BaseModel):
    """A page of indexed wines of one model."""

    model_name: str
    wine_count: int
    photo_count: int
    offset: int
    wines: list[ReferenceWine]


class IndexingError(BaseModel):
    """A reference photo that could not be indexed."""

    wine_id: str
    image_uri: str
    error_type: str
    message: str


class IndexingStats(BaseModel):
    """Summary of one indexing run."""

    status: Literal["ok", "partial", "failed"]
    model_name: str
    source: str | None = None
    total: int
    processed: int
    inserted: int
    updated: int
    skipped: int
    deleted: int = 0
    errors: int
    error_details: list[IndexingError] = Field(default_factory=list)
    reference_count: int


class SearchRequest(BaseModel):
    """Search request, also accepted as a JSON file from Apache Airflow."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1)
    image_uri: str = Field(min_length=1)
    top_k: int = Field(default=DEFAULT_TOP_K, gt=0)


class WineCandidate(BaseModel):
    """A wine similar to the query photo, represented by its best photo."""

    wine_id: str
    slug: str
    score: float
    distance: float
    best_image_uri: str


class SearchResponse(BaseModel):
    """Top-K distinct wines for one query photo."""

    request_id: str
    status: Literal["ok", "no_results"]
    model_name: str
    query_embedding_dimension: int
    candidates: list[WineCandidate]
    message: str | None = None


class EvaluationQuery(BaseModel):
    """A user photo together with the wine that is known to be on it."""

    query_image_uri: str
    wine_id: str


class IncorrectQuery(BaseModel):
    """A processed query whose correct wine is not among the top-K wines."""

    query_image_uri: str
    expected_wine_id: str
    predicted_wine_ids: list[str]
    expected_rank: int | None
    reason: str


class EvaluationError(BaseModel):
    """A query that could not be evaluated, e.g. an unreadable photo."""

    query_image_uri: str
    expected_wine_id: str
    error_type: str
    reason: str


class EvaluationReport(BaseModel):
    """Retrieval quality on a labelled set of user photos.

    Recall@K is the share of processed queries whose correct wine_id is among
    the K best wines; queries with errors are not part of the denominator.
    """

    status: Literal["ok", "partial", "failed"]
    model_name: str
    queries_total: int
    queries_processed: int
    queries_with_errors: int
    top_k: int
    recall_at_k: float | None
    correct_queries: int
    recall_at: dict[int, float | None]
    correct_at: dict[int, int]
    indexed_wines: int
    incorrect_queries: list[IncorrectQuery] = Field(default_factory=list)
    errors: list[EvaluationError] = Field(default_factory=list)
    message: str | None = None


class DatabaseInspection(BaseModel):
    """State of the database schema, as seen by the health check."""

    server_version: str
    pgvector_version: str | None
    table_exists: bool
    embedding_dimension: int | None
    reference_count: int | None
    model_reference_count: int | None


class HealthCheck(BaseModel):
    """Result of one health check."""

    status: Literal["ok", "error", "skipped"]
    message: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class HealthReport(BaseModel):
    """Result of all health checks."""

    status: Literal["ok", "error"]
    checks: dict[str, HealthCheck]
