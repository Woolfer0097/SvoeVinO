"""Turn match statistics into a verification verdict."""

from __future__ import annotations

from ..config import VerificationThresholds


def decide_verification(
    *,
    num_matches: int,
    num_inliers: int,
    thresholds: VerificationThresholds,
) -> tuple[bool, float]:
    """Return whether the pair is verified and its inlier ratio."""

    if num_matches < 0 or num_inliers < 0:
        raise ValueError("match statistics must be zero or positive")
    if num_inliers > num_matches:
        raise ValueError("num_inliers cannot exceed num_matches")

    ratio = (num_inliers / num_matches) if num_matches else 0.0
    verified = (
        num_matches >= thresholds.min_matches
        and num_inliers >= thresholds.min_inliers
        and ratio >= thresholds.min_inlier_ratio
    )
    return verified, ratio
