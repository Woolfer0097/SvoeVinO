"""Input and output contracts for photo verification."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class VerificationRequest(BaseModel):
    """Pair of local images to compare."""

    query_uri: str
    reference_uri: str


class ValidatedImage(BaseModel):
    """Information about an image that passed local validation."""

    path: Path
    width: int
    height: int
    mime_type: str


class MatchPrediction(BaseModel):
    """Local correspondences produced by SuperPoint and LightGlue."""

    query_points: list[tuple[float, float]]
    reference_points: list[tuple[float, float]]
    scores: list[float]
    num_keypoints_query: int
    num_keypoints_reference: int

    @property
    def num_matches(self) -> int:
        return len(self.query_points)


class VerificationThresholdsModel(BaseModel):
    """Serializable copy of the decision thresholds."""

    min_matches: int
    min_inliers: int
    min_inlier_ratio: float


class VerificationResult(BaseModel):
    """Verdict and match statistics for one photo pair."""

    verified: bool
    query_path: str
    reference_path: str
    query_width: int
    query_height: int
    reference_width: int
    reference_height: int
    model: str
    device: str
    num_keypoints_query: int
    num_keypoints_reference: int
    num_matches: int
    num_inliers: int
    inlier_ratio: float
    mean_match_score: float
    thresholds: VerificationThresholdsModel = Field(
        description="Thresholds used to accept the pair."
    )
