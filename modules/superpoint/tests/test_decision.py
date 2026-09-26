from __future__ import annotations

import pytest

from superpoint.config import VerificationThresholds
from superpoint.verification.decision import decide_verification
from superpoint.verification.homography import count_inlier_mask


THRESHOLDS = VerificationThresholds(
    min_matches=15,
    min_inliers=12,
    min_inlier_ratio=0.3,
)


def test_pair_is_verified_when_all_thresholds_pass() -> None:
    verified, ratio = decide_verification(
        num_matches=20,
        num_inliers=12,
        thresholds=THRESHOLDS,
    )

    assert verified is True
    assert ratio == 0.6


def test_pair_is_rejected_when_inlier_ratio_is_too_low() -> None:
    verified, ratio = decide_verification(
        num_matches=100,
        num_inliers=12,
        thresholds=THRESHOLDS,
    )

    assert verified is False
    assert ratio == 0.12


def test_empty_match_set_is_not_verified() -> None:
    verified, ratio = decide_verification(
        num_matches=0,
        num_inliers=0,
        thresholds=THRESHOLDS,
    )

    assert verified is False
    assert ratio == 0.0


def test_inliers_cannot_exceed_matches() -> None:
    with pytest.raises(ValueError, match="num_inliers"):
        decide_verification(num_matches=2, num_inliers=3, thresholds=THRESHOLDS)


def test_inlier_mask_counts_only_positive_values() -> None:
    class Mask:
        def ravel(self):
            return [1, 0, 1, 0]

    assert count_inlier_mask(Mask()) == 2
    assert count_inlier_mask(None) == 0
