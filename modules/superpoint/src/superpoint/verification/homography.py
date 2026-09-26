"""Geometric verification of LightGlue correspondences."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from importlib import import_module
from typing import Any

from ..config import get_ransac_confidence, get_ransac_reproj_threshold

FindHomography = Callable[
    [Sequence[tuple[float, float]], Sequence[tuple[float, float]], float, float],
    tuple[Any, Any],
]


class HomographyVerifier:
    """Count inliers of a homography estimated with OpenCV RANSAC."""

    def __init__(
        self,
        *,
        reproj_threshold: float | None = None,
        confidence: float | None = None,
        find_homography: FindHomography | None = None,
    ) -> None:
        self.reproj_threshold = (
            reproj_threshold
            if reproj_threshold is not None
            else get_ransac_reproj_threshold()
        )
        self.confidence = (
            confidence if confidence is not None else get_ransac_confidence()
        )
        self._find_homography = find_homography

    def count_inliers(
        self,
        query_points: Sequence[tuple[float, float]],
        reference_points: Sequence[tuple[float, float]],
    ) -> int:
        """Return the number of correspondences consistent with one homography."""

        if len(query_points) != len(reference_points):
            raise ValueError("query and reference correspondences must have the same length")
        if len(query_points) < 4:
            return 0

        find_homography = self._find_homography or _cv2_find_homography
        _matrix, mask = find_homography(
            query_points,
            reference_points,
            self.reproj_threshold,
            self.confidence,
        )
        return count_inlier_mask(mask)


def count_inlier_mask(mask: Any) -> int:
    """Count positive entries in an OpenCV inlier mask."""

    if mask is None:
        return 0
    values = mask.ravel() if hasattr(mask, "ravel") else mask
    if hasattr(values, "tolist"):
        values = values.tolist()
    return sum(1 for value in values if int(value) > 0)


def _cv2_find_homography(
    query_points: Sequence[tuple[float, float]],
    reference_points: Sequence[tuple[float, float]],
    reproj_threshold: float,
    confidence: float,
) -> tuple[Any, Any]:
    try:
        cv2 = import_module("cv2")
        numpy = import_module("numpy")
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required for geometric verification. "
            "Install the ML extra with: pip install -e '.[ml]'"
        ) from exc

    query = numpy.asarray(query_points, dtype=numpy.float32)
    reference = numpy.asarray(reference_points, dtype=numpy.float32)
    try:
        # RANSAC is otherwise nondeterministic around the inlier threshold.
        cv2.setRNGSeed(0)
        return cv2.findHomography(
            query,
            reference,
            method=cv2.RANSAC,
            ransacReprojThreshold=reproj_threshold,
            confidence=confidence,
        )
    except cv2.error as exc:
        raise RuntimeError("Homography estimation failed") from exc
