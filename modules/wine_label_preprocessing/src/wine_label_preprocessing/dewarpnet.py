"""Lazy adapter for the official document-trained DewarpNet implementation."""

from __future__ import annotations

from collections import OrderedDict
from importlib import import_module, util
from pathlib import Path
from time import perf_counter_ns
from types import ModuleType
from typing import Any

import cv2
import numpy as np

from .errors import DewarpNetError, ImageInputError
from .models import DewarpNetConfig, DewarpNetResult, RGBArray


def _milliseconds(start_ns: int) -> float:
    return (perf_counter_ns() - start_ns) / 1_000_000.0


def normalized_grid_to_pixel_maps(
    normalized_grid: np.ndarray,
    *,
    width: int,
    height: int,
    align_corners: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Upsample a normalized backward map and convert it to source pixels."""

    if (
        not isinstance(normalized_grid, np.ndarray)
        or normalized_grid.ndim != 3
        or normalized_grid.shape[2] != 2
    ):
        shape = getattr(normalized_grid, "shape", None)
        raise ValueError(f"DewarpNet backward map must have shape HxWx2, got {shape}")
    if width < 1 or height < 1:
        raise ValueError("DewarpNet output dimensions must be positive")
    grid = np.asarray(normalized_grid, dtype=np.float32)
    grid_x = cv2.resize(grid[..., 0], (width, height), interpolation=cv2.INTER_LINEAR)
    grid_y = cv2.resize(grid[..., 1], (width, height), interpolation=cv2.INTER_LINEAR)
    if align_corners:
        map_x = (grid_x + 1.0) * (width - 1) / 2.0
        map_y = (grid_y + 1.0) * (height - 1) / 2.0
    else:
        map_x = ((grid_x + 1.0) * width - 1.0) / 2.0
        map_y = ((grid_y + 1.0) * height - 1.0) / 2.0
    map_x = np.asarray(map_x, dtype=np.float32)
    map_y = np.asarray(map_y, dtype=np.float32)
    valid = (
        np.isfinite(map_x)
        & np.isfinite(map_y)
        & (map_x >= 0.0)
        & (map_x <= width - 1)
        & (map_y >= 0.0)
        & (map_y <= height - 1)
    )
    return map_x, map_y, np.ascontiguousarray(valid)


class DewarpNetAdapter:
    """Load official networks once and sample their map at source resolution."""

    def __init__(self, config: DewarpNetConfig) -> None:
        self.config = config
        self._loaded = False
        self._torch: Any | None = None
        self._wc_model: Any | None = None
        self._bm_model: Any | None = None
        self._device: Any | None = None

    @property
    def loaded(self) -> bool:
        return self._loaded

    def _validate_paths(self) -> None:
        repo = Path(self.config.official_repo)
        if not repo.is_dir():
            raise FileNotFoundError(f"Official DewarpNet checkout not found: {repo}")
        for filename in ("unetnc.py", "densenetccnl.py"):
            path = repo / "models" / filename
            if not path.is_file():
                raise FileNotFoundError(f"Official DewarpNet model source not found: {path}")
        for role, path_value in (
            ("world-coordinate checkpoint", self.config.wc_checkpoint),
            ("backward-map checkpoint", self.config.bm_checkpoint),
        ):
            path = Path(path_value)
            if not path.is_file():
                raise FileNotFoundError(f"DewarpNet {role} not found: {path}")

    @staticmethod
    def _load_source_module(path: Path, name: str) -> ModuleType:
        spec = util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import official DewarpNet source: {path}")
        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _checkpoint_state(torch_module: Any, path: Path, device: Any) -> Any:
        checkpoint = torch_module.load(path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            state = checkpoint["model_state"]
        else:
            state = checkpoint
        if not hasattr(state, "items"):
            raise ValueError(f"Checkpoint does not contain a state dictionary: {path}")
        return OrderedDict(
            (
                key[7:] if isinstance(key, str) and key.startswith("module.") else key,
                value,
            )
            for key, value in state.items()
        )

    def _load_models(self) -> None:
        torch_module = import_module("torch")
        repo = Path(self.config.official_repo)
        unique = f"wine_label_dewarpnet_{id(self)}"
        unet_module = self._load_source_module(
            repo / "models" / "unetnc.py", f"{unique}_unetnc"
        )
        dense_module = self._load_source_module(
            repo / "models" / "densenetccnl.py", f"{unique}_densenetccnl"
        )

        requested_device = self.config.device.lower()
        if requested_device == "auto":
            requested_device = "cuda" if torch_module.cuda.is_available() else "cpu"
        if requested_device == "cuda" and not torch_module.cuda.is_available():
            raise RuntimeError("DewarpNet CUDA was requested but is not available")
        if requested_device not in {"cpu", "cuda"}:
            raise ValueError("DewarpNet device must be 'auto', 'cpu', or 'cuda'")
        device = torch_module.device(requested_device)

        wc_model = unet_module.UnetGenerator(
            input_nc=3, output_nc=3, num_downs=7
        )
        bm_model = dense_module.dnetccnl(
            img_size=128,
            in_channels=3,
            out_channels=2,
            filters=32,
        )
        wc_state = self._checkpoint_state(
            torch_module, Path(self.config.wc_checkpoint), device
        )
        bm_state = self._checkpoint_state(
            torch_module, Path(self.config.bm_checkpoint), device
        )
        wc_model.load_state_dict(wc_state)
        bm_model.load_state_dict(bm_state)
        wc_model.to(device).eval()
        bm_model.to(device).eval()

        self._torch = torch_module
        self._wc_model = wc_model
        self._bm_model = bm_model
        self._device = device

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._validate_paths()
        self._load_models()
        self._loaded = True

    def _predict_normalized_map(self, image: RGBArray) -> np.ndarray:
        torch_module = self._torch
        if torch_module is None or self._wc_model is None or self._bm_model is None:
            raise RuntimeError("DewarpNet models are not loaded")
        model_input = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
        bgr_chw = np.ascontiguousarray(model_input[..., ::-1].transpose(2, 0, 1))
        tensor = (
            torch_module.from_numpy(bgr_chw)
            .float()
            .unsqueeze(0)
            .div_(255.0)
            .to(self._device)
        )
        with torch_module.inference_mode():
            world = torch_module.nn.functional.hardtanh(
                self._wc_model(tensor), 0.0, 1.0
            )
            bm_input = torch_module.nn.functional.interpolate(
                world, size=(128, 128), mode="nearest"
            )
            backward_map = self._bm_model(bm_input)
        array = backward_map[0].detach().cpu().numpy().transpose(1, 2, 0)
        return np.asarray(array, dtype=np.float32)

    def unwrap(self, image: RGBArray, *, strict: bool = False) -> DewarpNetResult:
        """Return a true learned remap or an explicit unavailable/failed result."""

        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ImageInputError("DewarpNet input must be RGB uint8 with shape HxWx3")
        timings: dict[str, float] = {}
        load_start = perf_counter_ns()
        try:
            self._ensure_loaded()
        except Exception as exc:
            if strict:
                raise DewarpNetError(f"DewarpNet is unavailable: {exc}") from exc
            timings["model_load"] = _milliseconds(load_start)
            return DewarpNetResult(
                None,
                None,
                "unavailable",
                f"{type(exc).__name__}: {exc}",
                {"timings_ms": timings},
            )
        timings["model_load"] = _milliseconds(load_start)

        try:
            inference_start = perf_counter_ns()
            normalized_map = self._predict_normalized_map(image)
            timings["inference"] = _milliseconds(inference_start)
            if self.config.blur_backward_map:
                normalized_map = np.stack(
                    [
                        cv2.blur(normalized_map[..., channel], (3, 3))
                        for channel in range(2)
                    ],
                    axis=-1,
                )

            remap_start = perf_counter_ns()
            height, width = image.shape[:2]
            map_x, map_y, valid = normalized_grid_to_pixel_maps(
                normalized_map,
                width=width,
                height=height,
                align_corners=self.config.align_corners,
            )
            output = cv2.remap(
                image,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            output[~valid] = 0
            timings["map_and_remap"] = _milliseconds(remap_start)
        except Exception as exc:
            if strict:
                raise DewarpNetError(f"DewarpNet inference failed: {exc}") from exc
            return DewarpNetResult(
                None,
                None,
                "failed",
                f"{type(exc).__name__}: {exc}",
                {"timings_ms": timings},
            )

        return DewarpNetResult(
            image=np.ascontiguousarray(output),
            valid_mask=valid,
            status="ok",
            reason=None,
            metadata={
                "device": str(self._device) if self._device is not None else "test",
                "model_map_shape": list(normalized_map.shape[:2]),
                "output_shape": list(output.shape),
                "align_corners": self.config.align_corners,
                "blur_backward_map": self.config.blur_backward_map,
                "valid_fraction": float(np.mean(valid)),
                "sampling_passes": 1,
                "timings_ms": timings,
            },
        )

