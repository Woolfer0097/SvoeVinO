from __future__ import annotations

from pathlib import Path

import pytest

from superpoint.config import (
    ConfigurationError,
    get_data_root,
    get_depth_confidence,
    get_max_num_keypoints,
    get_requested_device,
    get_resize,
    get_verification_thresholds,
    get_width_confidence,
)


def test_data_root_must_be_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATA_ROOT", raising=False)

    with pytest.raises(ConfigurationError, match="DATA_ROOT must be set"):
        get_data_root()


def test_data_root_must_exist(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="does not exist"):
        get_data_root(tmp_path / "missing")


def test_max_num_keypoints_none_disables_the_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAX_NUM_KEYPOINTS", "none")

    assert get_max_num_keypoints() is None


def test_resize_zero_disables_resizing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUPERPOINT_RESIZE", "0")

    assert get_resize() is None


def test_invalid_device_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="DEVICE"):
        get_requested_device("mps")


def test_inlier_ratio_must_be_a_fraction() -> None:
    with pytest.raises(ConfigurationError, match="MIN_INLIER_RATIO"):
        get_verification_thresholds(min_inlier_ratio=1.5)


def test_lightweight_profile_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_NUM_KEYPOINTS", "512")
    monkeypatch.setenv("SUPERPOINT_RESIZE", "768")
    monkeypatch.setenv("LIGHTGLUE_DEPTH_CONFIDENCE", "0.90")
    monkeypatch.setenv("LIGHTGLUE_WIDTH_CONFIDENCE", "0.95")

    assert get_max_num_keypoints() == 512
    assert get_resize() == 768
    assert get_depth_confidence() == 0.90
    assert get_width_confidence() == 0.95
