from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from superpoint.application.verify_photos import verify_photos
from superpoint.cli import main
from superpoint.config import VerificationThresholds
from superpoint.contracts import MatchPrediction, VerificationRequest


class FakeMatcher:
    model_name = "fake-matcher"
    device = "cpu"

    def __init__(self, prediction: MatchPrediction) -> None:
        self.prediction = prediction
        self.modes: list[str] = []

    def match(self, query_image: Image.Image, reference_image: Image.Image) -> MatchPrediction:
        self.modes.extend([query_image.mode, reference_image.mode])
        return self.prediction


class FakeGeometry:
    def __init__(self, inliers: int) -> None:
        self.inliers = inliers

    def count_inliers(self, query_points, reference_points) -> int:
        assert len(query_points) == len(reference_points)
        return self.inliers


def write_image(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(30, 60, 90)).save(path, format="JPEG")


def test_verify_photos_accepts_a_geometrically_consistent_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    write_image(tmp_path / "queries" / "photo.jpg", (16, 10))
    write_image(tmp_path / "references" / "label.jpg", (20, 12))
    prediction = MatchPrediction(
        query_points=[(1.0, 2.0), (3.0, 4.0)],
        reference_points=[(5.0, 6.0), (7.0, 8.0)],
        scores=[0.9, 0.7],
        num_keypoints_query=4,
        num_keypoints_reference=5,
    )
    matcher = FakeMatcher(prediction)

    result = verify_photos(
        VerificationRequest(
            query_uri="queries/photo.jpg",
            reference_uri="references/label.jpg",
        ),
        matcher=matcher,
        geometry=FakeGeometry(inliers=2),
        thresholds=VerificationThresholds(
            min_matches=2,
            min_inliers=2,
            min_inlier_ratio=0.5,
        ),
    )

    assert result.verified is True
    assert result.model == "fake-matcher"
    assert result.num_matches == 2
    assert result.num_inliers == 2
    assert result.inlier_ratio == 1.0
    assert result.mean_match_score == 0.8
    assert result.query_width == 16
    assert result.reference_height == 12
    assert matcher.modes == ["RGB", "RGB"]


def test_verify_photos_rejects_a_pair_without_loading_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    write_image(tmp_path / "a.jpg", (8, 8))
    write_image(tmp_path / "b.jpg", (8, 8))

    result = verify_photos(
        VerificationRequest(query_uri="a.jpg", reference_uri="b.jpg"),
        matcher=FakeMatcher(
            MatchPrediction(
                query_points=[],
                reference_points=[],
                scores=[],
                num_keypoints_query=3,
                num_keypoints_reference=1,
            )
        ),
        geometry=FakeGeometry(inliers=0),
    )

    assert result.verified is False
    assert result.num_matches == 0
    assert result.inlier_ratio == 0.0


def test_cli_validate_prints_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    write_image(tmp_path / "photo.jpg", (9, 4))

    exit_code = main(["validate", "--image-uri", str(tmp_path / "photo.jpg")])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["width"] == 9
    assert payload["mime_type"] == "image/jpeg"
