"""SuperPoint + LightGlue matcher backed by cvg/LightGlue."""

from __future__ import annotations

from importlib import import_module
import hashlib
import os
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from ..config import (
    get_depth_confidence,
    get_filter_threshold,
    get_max_num_keypoints,
    get_requested_device,
    get_resize,
    get_width_confidence,
)
from ..contracts import MatchPrediction
from .feature_cache import FeatureCache

MODEL_NAME = "superpoint+lightglue"
ML_INSTALL_HINT = "Install them with: pip install -e '.[ml]'"


class SuperPointLightGlueMatcher:
    """Extract SuperPoint features and match them with LightGlue."""

    def __init__(
        self,
        *,
        torch_module: Any | None = None,
        extractor: Any | None = None,
        matcher: Any | None = None,
        image_to_tensor: Callable[[Any], Any] | None = None,
        match_fn: Callable[..., tuple[Any, Any, Any]] | None = None,
        device: str | None = None,
        max_num_keypoints: int | None = None,
        resize: int | None = None,
        depth_confidence: float | None = None,
        width_confidence: float | None = None,
        filter_threshold: float | None = None,
        feature_cache_mb: int | None = None,
        feature_cache_dir: str | Path | None = None,
        feature_match_fn: Callable | None = None,
    ) -> None:
        if (extractor is None) != (matcher is None):
            raise ValueError("extractor and matcher must be provided together")

        self.max_num_keypoints = (
            max_num_keypoints
            if max_num_keypoints is not None
            else get_max_num_keypoints()
        )
        self.resize = resize if resize is not None else get_resize()
        self.depth_confidence = (
            depth_confidence
            if depth_confidence is not None
            else get_depth_confidence()
        )
        self.width_confidence = (
            width_confidence
            if width_confidence is not None
            else get_width_confidence()
        )
        self.filter_threshold = (
            filter_threshold
            if filter_threshold is not None
            else get_filter_threshold()
        )
        cache_mb = feature_cache_mb if feature_cache_mb is not None else int(
            os.getenv("SUPERPOINT_FEATURE_CACHE_MB", "256"))
        cache_dir = feature_cache_dir if feature_cache_dir is not None else os.getenv(
            "SUPERPOINT_FEATURE_CACHE_DIR")
        self.feature_cache = FeatureCache(cache_mb * 1024 * 1024,
                                          Path(cache_dir) if cache_dir else None)
        # Preserve injected legacy pair matchers. Production separates extraction
        # from matching, without changing SuperPoint/LightGlue parameters.
        self._use_feature_cache = match_fn is None
        self._feature_match_fn = feature_match_fn or _match_cached_features
        self._feature_signature: bytes | None = None

        if extractor is None:
            torch_module, extractor_cls, matcher_cls, image_to_tensor, match_fn = (
                _load_lightglue()
            )
            self._torch = torch_module
            self._device_name = _resolve_device(self._torch, device)
            runtime_device = self._torch.device(self._device_name)
            self.extractor = (
                extractor_cls(max_num_keypoints=self.max_num_keypoints)
                .eval()
                .to(runtime_device)
            )
            self.matcher = (
                matcher_cls(
                    features="superpoint",
                    depth_confidence=self.depth_confidence,
                    width_confidence=self.width_confidence,
                    filter_threshold=self.filter_threshold,
                )
                .eval()
                .to(runtime_device)
            )
            self._image_to_tensor = image_to_tensor
            self._match_fn = match_fn
            return

        self._torch = torch_module if torch_module is not None else _load_torch()
        self._device_name = _resolve_device(self._torch, device)
        self.extractor = extractor
        self.matcher = matcher
        self._image_to_tensor = (
            image_to_tensor if image_to_tensor is not None else _load_image_to_tensor()
        )
        self._match_fn = (match_fn if match_fn is not None else
                          None if feature_match_fn is not None else _load_match_fn())

    @property
    def model_name(self) -> str:
        return MODEL_NAME

    @property
    def device(self) -> str:
        return self._device_name

    def match(self, query_image: Image.Image, reference_image: Image.Image) -> MatchPrediction:
        """Match two RGB images and return original-image correspondences."""

        if query_image.mode != "RGB" or reference_image.mode != "RGB":
            raise ValueError("SuperPoint+LightGlue matcher expects RGB images")

        with self._torch.inference_mode():
            # match_pair's device only moves the result. Inference follows the
            # tensor and module device set above; CPU results are what we serialize.
            if self._use_feature_cache:
                feats0 = self._features(query_image, persistent=False)
                feats1 = self._features(reference_image, persistent=True)
                feats0, feats1, matches01 = self._feature_match_fn(self.matcher, feats0, feats1)
            else:
                query_tensor = self._image_to_tensor(query_image).to(self._device_name)
                reference_tensor = self._image_to_tensor(reference_image).to(self._device_name)
                feats0, feats1, matches01 = self._match_fn(
                    self.extractor, self.matcher, query_tensor, reference_tensor,
                    device="cpu", resize=self.resize,
                )
            if self._device_name.startswith("cuda"):
                # Pinned LightGlue match_pair uses non_blocking=True when
                # copying to CPU. Reading tolist() before those copies finish
                # can produce stale/garbage keypoints and match indices.
                self._torch.cuda.synchronize(self._device_name)
        return prediction_from_features(feats0, feats1, matches01)

    def _features(self, image: Image.Image, *, persistent: bool) -> dict:
        if self._feature_signature is None:
            signature = hashlib.sha256()
            signature.update(repr(("superpoint-cache-v1", str(type(self.extractor)),
                                   str(getattr(self.extractor, "conf", "")),
                                   str(getattr(self._torch, "__version__", "")),
                                   self._device_name)).encode())
            # Invalidate disk features when the actual detector weights change.
            if hasattr(self.extractor, "state_dict"):
                for name, value in sorted(self.extractor.state_dict().items()):
                    signature.update(name.encode())
                    signature.update(value.detach().cpu().numpy().tobytes())
            self._feature_signature = signature.digest()
        digest = hashlib.sha256(self._feature_signature)
        digest.update(repr((image.size, image.mode, self.resize, self.max_num_keypoints)).encode())
        digest.update(image.tobytes())
        key = digest.hexdigest()
        cpu_features = self.feature_cache.get(key, self._torch, persistent=persistent)
        if cpu_features is None:
            tensor = self._image_to_tensor(image).to(self._device_name)
            features = self.extractor.extract(tensor, resize=self.resize)
            cpu_features = self.feature_cache.put(key, features, persistent=persistent)
        # Protect cached tensors even if a matcher version mutates its inputs.
        return {name: value.to(self._device_name).clone() for name, value in cpu_features.items()}


def _match_cached_features(matcher, feats0: dict, feats1: dict):
    utils = import_module("lightglue.utils")
    matched = matcher({"image0": feats0, "image1": feats1})
    return tuple(utils.batch_to_device(utils.rbd(item), "cpu")
                 for item in (feats0, feats1, matched))


def prediction_from_features(
    feats0: dict[str, Any],
    feats1: dict[str, Any],
    matches01: dict[str, Any],
) -> MatchPrediction:
    """Convert a LightGlue result into point correspondences."""

    keypoints_query = _to_points(feats0.get("keypoints"), "query keypoints")
    keypoints_reference = _to_points(feats1.get("keypoints"), "reference keypoints")
    matches = _to_pairs(matches01.get("matches"))
    scores = _to_floats(matches01.get("scores"))
    if len(matches) != len(scores):
        raise RuntimeError("LightGlue returned a different number of matches and scores")

    query_points: list[tuple[float, float]] = []
    reference_points: list[tuple[float, float]] = []
    for left, right in matches:
        if left < 0 or right < 0:
            raise RuntimeError("LightGlue returned an out-of-range match index")
        try:
            query_points.append(keypoints_query[left])
            reference_points.append(keypoints_reference[right])
        except IndexError as exc:
            raise RuntimeError("LightGlue returned an out-of-range match index") from exc

    return MatchPrediction(
        query_points=query_points,
        reference_points=reference_points,
        scores=scores,
        num_keypoints_query=len(keypoints_query),
        num_keypoints_reference=len(keypoints_reference),
    )


def _load_lightglue() -> tuple[Any, Any, Any, Callable[[Any], Any], Callable[..., tuple[Any, Any, Any]]]:
    try:
        torch_module = import_module("torch")
        lightglue = import_module("lightglue")
        utils = import_module("lightglue.utils")
    except ImportError as exc:
        raise ImportError(
            "SuperPoint+LightGlue dependencies are missing. " + ML_INSTALL_HINT
        ) from exc
    return (
        torch_module,
        lightglue.SuperPoint,
        lightglue.LightGlue,
        _wrap_image_to_tensor(utils.numpy_image_to_torch),
        utils.match_pair,
    )


def _load_torch() -> Any:
    try:
        return import_module("torch")
    except ImportError as exc:
        raise ImportError(
            "SuperPoint+LightGlue dependencies are missing. " + ML_INSTALL_HINT
        ) from exc


def _load_image_to_tensor() -> Callable[[Any], Any]:
    try:
        utils = import_module("lightglue.utils")
    except ImportError as exc:
        raise ImportError(
            "SuperPoint+LightGlue dependencies are missing. " + ML_INSTALL_HINT
        ) from exc
    return _wrap_image_to_tensor(utils.numpy_image_to_torch)


def _load_match_fn() -> Callable[..., tuple[Any, Any, Any]]:
    try:
        utils = import_module("lightglue.utils")
    except ImportError as exc:
        raise ImportError(
            "SuperPoint+LightGlue dependencies are missing. " + ML_INSTALL_HINT
        ) from exc
    return utils.match_pair


def _resolve_device(torch_module: Any, requested: str | None) -> str:
    device = get_requested_device(requested)
    cuda = getattr(torch_module, "cuda", None)
    cuda_available = bool(getattr(cuda, "is_available", lambda: False)())
    if device == "cuda" and not cuda_available:
        raise RuntimeError("DEVICE=cuda but CUDA is not available")
    if device == "auto":
        return "cuda" if cuda_available else "cpu"
    return device


def _wrap_image_to_tensor(
    numpy_image_to_torch: Callable[[Any], Any],
) -> Callable[[Image.Image], Any]:
    def convert(image: Image.Image) -> Any:
        try:
            numpy = import_module("numpy")
        except ImportError as exc:
            raise ImportError(
                "SuperPoint+LightGlue dependencies are missing. " + ML_INSTALL_HINT
            ) from exc
        return numpy_image_to_torch(numpy.asarray(image))

    return convert


def _to_nested_list(value: Any, label: str) -> list[Any]:
    if value is None:
        raise RuntimeError(f"LightGlue result is missing {label}")
    current = value
    if hasattr(current, "detach"):
        current = current.detach()
    if hasattr(current, "cpu"):
        current = current.cpu()
    if hasattr(current, "tolist"):
        current = current.tolist()
    if not isinstance(current, list):
        raise RuntimeError(f"LightGlue result has unsupported {label}")
    return current


def _to_points(value: Any, label: str) -> list[tuple[float, float]]:
    rows = _to_nested_list(value, label)
    points: list[tuple[float, float]] = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise RuntimeError(f"{label} must have shape (N, 2)")
        points.append((float(row[0]), float(row[1])))
    return points


def _to_pairs(value: Any) -> list[tuple[int, int]]:
    rows = _to_nested_list(value, "matches")
    pairs: list[tuple[int, int]] = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise RuntimeError("matches must have shape (K, 2)")
        pairs.append((int(row[0]), int(row[1])))
    return pairs


def _to_floats(value: Any) -> list[float]:
    values = _to_nested_list(value, "scores")
    return [float(item) for item in values]
