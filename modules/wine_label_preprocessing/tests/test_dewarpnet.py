from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wine_label_preprocessing.dewarpnet import (
    DewarpNetAdapter,
    normalized_grid_to_pixel_maps,
)
from wine_label_preprocessing.models import DewarpNetConfig


def config_with_files(tmp_path: Path, **overrides) -> DewarpNetConfig:
    repo = tmp_path / "DewarpNet"
    models = repo / "models"
    models.mkdir(parents=True)
    (models / "unetnc.py").write_text("# test module\n")
    (models / "densenetccnl.py").write_text("# test module\n")
    wc = tmp_path / "wc.pkl"
    bm = tmp_path / "bm.pkl"
    wc.write_bytes(b"test")
    bm.write_bytes(b"test")
    values = {
        "official_repo": repo,
        "wc_checkpoint": wc,
        "bm_checkpoint": bm,
        "blur_backward_map": False,
    }
    values.update(overrides)
    return DewarpNetConfig(**values)


def test_constructor_is_lazy(tmp_path: Path) -> None:
    adapter = DewarpNetAdapter(
        DewarpNetConfig(
            tmp_path / "repo", tmp_path / "wc.pkl", tmp_path / "bm.pkl"
        )
    )

    assert not adapter.loaded


def test_missing_checkpoints_are_unavailable_not_resize(tmp_path: Path) -> None:
    config = DewarpNetConfig(
        tmp_path / "repo", tmp_path / "wc.pkl", tmp_path / "bm.pkl"
    )

    result = DewarpNetAdapter(config).unwrap(
        np.zeros((40, 60, 3), dtype=np.uint8)
    )

    assert result.image is None
    assert result.valid_mask is None
    assert result.status == "unavailable"
    assert "checkpoint" in result.reason.lower() or "checkout" in result.reason.lower()


@pytest.mark.parametrize("align_corners", [False, True])
def test_normalized_grid_maps_original_resolution(align_corners: bool) -> None:
    grid = np.zeros((8, 8, 2), dtype=np.float32)

    map_x, map_y, valid = normalized_grid_to_pixel_maps(
        grid,
        width=321,
        height=197,
        align_corners=align_corners,
    )

    assert map_x.shape == (197, 321)
    assert map_y.shape == (197, 321)
    assert valid.shape == (197, 321)
    assert np.all(valid)
    assert float(map_x[90, 160]) == pytest.approx(160.0)
    assert float(map_y[90, 160]) == pytest.approx(98.0)


def test_models_load_only_once_and_sampling_uses_original_pixels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = config_with_files(tmp_path, align_corners=True)
    adapter = DewarpNetAdapter(config)
    load_calls = 0

    def fake_load_models() -> None:
        nonlocal load_calls
        load_calls += 1

    axis = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(axis, axis)
    identity_grid = np.stack((grid_x, grid_y), axis=-1)
    monkeypatch.setattr(adapter, "_load_models", fake_load_models)
    monkeypatch.setattr(
        adapter,
        "_predict_normalized_map",
        lambda image: identity_grid,
    )
    yy, xx = np.mgrid[:31, :47]
    image = np.stack((xx, yy, np.zeros_like(xx)), axis=-1).astype(np.uint8)

    first = adapter.unwrap(image)
    second = adapter.unwrap(image)

    assert load_calls == 1
    assert adapter.loaded
    assert first.status == "ok"
    assert first.image is not None
    assert first.image.shape == image.shape
    assert second.image is not None
    assert second.image.shape == image.shape
    assert first.metadata["model_map_shape"] == [8, 8]
    assert first.metadata["output_shape"] == [31, 47, 3]
    assert first.metadata["sampling_passes"] == 1


def test_invalid_normalized_map_is_failed_not_returned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = DewarpNetAdapter(config_with_files(tmp_path))
    monkeypatch.setattr(adapter, "_load_models", lambda: None)
    monkeypatch.setattr(
        adapter,
        "_predict_normalized_map",
        lambda image: np.zeros((8, 8, 3), dtype=np.float32),
    )

    result = adapter.unwrap(np.zeros((20, 30, 3), dtype=np.uint8))

    assert result.status == "failed"
    assert result.image is None
    assert "shape" in result.reason.lower()
