"""Data contracts shared by the OCR application, API, and reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Point:
    """A point in image pixel coordinates."""

    x: float
    y: float


@dataclass(frozen=True)
class BoundingBox:
    """Polygonal text bounding box in original image coordinates."""

    points: tuple[Point, ...]

    @property
    def min_x(self) -> float:
        return min(point.x for point in self.points)

    @property
    def min_y(self) -> float:
        return min(point.y for point in self.points)

    @property
    def max_x(self) -> float:
        return max(point.x for point in self.points)

    @property
    def max_y(self) -> float:
        return max(point.y for point in self.points)

    def translated(self, dx: float, dy: float) -> "BoundingBox":
        """Return the same box translated by ``dx`` and ``dy`` pixels."""

        return BoundingBox(
            tuple(Point(point.x + dx, point.y + dy) for point in self.points)
        )

    def intersection_over_union(self, other: "BoundingBox") -> float:
        """Axis-aligned IoU approximation for duplicate suppression."""

        left = max(self.min_x, other.min_x)
        top = max(self.min_y, other.min_y)
        right = min(self.max_x, other.max_x)
        bottom = min(self.max_y, other.max_y)
        if right <= left or bottom <= top:
            return 0.0

        intersection = (right - left) * (bottom - top)
        self_area = (self.max_x - self.min_x) * (self.max_y - self.min_y)
        other_area = (other.max_x - other.min_x) * (other.max_y - other.min_y)
        union = self_area + other_area - intersection
        return intersection / union if union > 0 else 0.0


@dataclass(frozen=True)
class RecognizedTextBlock:
    """Raw text block returned by an OCR backend."""

    text: str
    confidence: float | None
    bbox: BoundingBox | None
    source_variant: str = "full"


@dataclass(frozen=True)
class TextBlock:
    """Postprocessed text block exposed to downstream matching."""

    text: str
    normalized_text: str
    confidence: float | None
    bbox: BoundingBox | None
    source_variant: str = "full"


@dataclass(frozen=True)
class CandidateField:
    """Potentially discriminative field extracted from OCR text."""

    field_type: str
    value: str
    normalized_value: str
    source_text: str
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OCRResult:
    """Structured OCR result used by REST and downstream components."""

    raw_text: str
    normalized_text: str
    text_blocks: list[TextBlock]
    tokens: list[str]
    candidate_fields: list[CandidateField]
    engine: str
    processing_time_ms: float
    warnings: list[str] = field(default_factory=list)
    candidate_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return asdict(self)
