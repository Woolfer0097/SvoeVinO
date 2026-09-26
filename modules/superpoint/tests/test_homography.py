from __future__ import annotations

from superpoint.verification.homography import HomographyVerifier


def test_fewer_than_four_correspondences_have_no_homography() -> None:
    verifier = HomographyVerifier(
        reproj_threshold=3.0,
        confidence=0.99,
        find_homography=lambda *args: (_ for _ in ()).throw(AssertionError("RANSAC")),
    )

    assert verifier.count_inliers([(0.0, 0.0)], [(1.0, 1.0)]) == 0


def test_injected_homography_counts_the_returned_mask() -> None:
    def find_homography(query_points, reference_points, reproj_threshold, confidence):
        assert len(query_points) == 4
        assert reproj_threshold == 4.0
        assert confidence == 0.9
        return {"matrix": True}, [1, 1, 0, 1]

    verifier = HomographyVerifier(
        reproj_threshold=4.0,
        confidence=0.9,
        find_homography=find_homography,
    )
    points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]

    assert verifier.count_inliers(points, points) == 3
