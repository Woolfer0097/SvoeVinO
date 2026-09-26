from __future__ import annotations

from pathlib import Path

import pytest

from superpoint.config import (
    ConfigurationError,
    get_data_root,
    get_max_num_keypoints,
    get_requested_device,
    get_resize,
    get_verification_thresholds,
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
