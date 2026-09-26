"""Use case for verifying a query photo against a reference photo."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from ..config import VerificationThresholds, get_verification_thresholds
from ..contracts import (
    MatchPrediction,
    VerificationRequest,
    VerificationResult,
    VerificationThresholdsModel,
)
from ..infrastructure.storage.local_storage import LocalImageStorage
from ..matching.base import PhotoMatcher
from ..preprocessing.image_preprocessor import ImagePreprocessor
from ..verification.decision import decide_verification
from .validate_image import ImageStorage


class GeometryVerifier(Protocol):
    """Count correspondences that agree with a geometric model."""

    def count_inliers(
        self,
        query_points: Sequence[tuple[float, float]],
        reference_points: Sequence[tuple[float, float]],
    ) -> int:
        """Return the number of geometrically consistent correspondences."""

        ...


def verify_photos(
    request: VerificationRequest,
    *,
    storage: ImageStorage | None = None,
    reference_storage: ImageStorage | None = None,
    preprocessor: ImagePreprocessor | None = None,
    matcher: PhotoMatcher | None = None,
    geometry: GeometryVerifier | None = None,
    thresholds: VerificationThresholds | None = None,
) -> VerificationResult:
    """Validate both photos, match them and decide whether they show the same object.

    ``reference_storage`` is the catalog root when the query lives in a
    temporary upload directory and the reference path comes from the database.
    """

    image_storage = storage if storage is not None else LocalImageStorage()
    catalog_storage = (
        reference_storage if reference_storage is not None else image_storage
    )
    image_preprocessor = (
        preprocessor if preprocessor is not None else ImagePreprocessor()
    )
    decision_thresholds = (
        thresholds if thresholds is not None else get_verification_thresholds()
    )
    query_image = image_storage.validate(request.query_uri)
    reference_image = catalog_storage.validate(request.reference_uri)
    photo_matcher = matcher if matcher is not None else _default_matcher()
    geometry_verifier = geometry if geometry is not None else _default_geometry()

    with image_preprocessor.open_rgb(Path(query_image.path)) as query_rgb, (
        image_preprocessor.open_rgb(Path(reference_image.path))
    ) as reference_rgb:
        prediction = photo_matcher.match(query_rgb, reference_rgb)

    _require_consistent_prediction(prediction)
    num_inliers = geometry_verifier.count_inliers(
        prediction.query_points,
        prediction.reference_points,
    )
    verified, inlier_ratio = decide_verification(
        num_matches=prediction.num_matches,
        num_inliers=num_inliers,
        thresholds=decision_thresholds,
    )
    mean_score = (
        sum(prediction.scores) / len(prediction.scores) if prediction.scores else 0.0
    )
    return VerificationResult(
        verified=verified,
        query_path=str(query_image.path),
        reference_path=str(reference_image.path),
        query_width=query_image.width,
        query_height=query_image.height,
        reference_width=reference_image.width,
        reference_height=reference_image.height,
        model=photo_matcher.model_name,
        device=photo_matcher.device,
        num_keypoints_query=prediction.num_keypoints_query,
        num_keypoints_reference=prediction.num_keypoints_reference,
        num_matches=prediction.num_matches,
        num_inliers=num_inliers,
        inlier_ratio=round(inlier_ratio, 6),
        mean_match_score=round(mean_score, 6),
        thresholds=VerificationThresholdsModel(
            min_matches=decision_thresholds.min_matches,
            min_inliers=decision_thresholds.min_inliers,
            min_inlier_ratio=decision_thresholds.min_inlier_ratio,
        ),
    )


def _default_matcher() -> PhotoMatcher:
    from ..matching.superpoint_lightglue import SuperPointLightGlueMatcher

    return SuperPointLightGlueMatcher()


def _default_geometry() -> GeometryVerifier:
    from ..verification.homography import HomographyVerifier

    return HomographyVerifier()


def _require_consistent_prediction(prediction: MatchPrediction) -> None:
    if not (
        len(prediction.query_points)
        == len(prediction.reference_points)
        == len(prediction.scores)
    ):
        raise RuntimeError("matcher returned inconsistent correspondence lengths")
    if prediction.num_keypoints_query < len(prediction.query_points):
        raise RuntimeError("matcher reported fewer query keypoints than matches")
    if prediction.num_keypoints_reference < len(prediction.reference_points):
        raise RuntimeError("matcher reported fewer reference keypoints than matches")
